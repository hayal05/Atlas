"""
Ingestion + retrieval pipeline.

Docs (markdown/txt/pdf/docx) are stored in the Neon `documents` table,
split into heading-aware chunks, embedded locally with fastembed's ONNX
all-MiniLM-L6-v2 model (no torch/transformers/chromadb dependency --
important for staying inside low-memory hosting tiers), and stored in
the Neon `chunks` table alongside their pgvector embedding. At query
time we embed the question and pull back the top-K most similar chunks
with their source citations via a plain SQL ORDER BY on vector distance.

Storing both the source documents and the vector index in Postgres
(rather than on local disk) means everything survives a Render redeploy,
restart, or move to a new instance -- Render's filesystem does not.
"""
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from fastembed import TextEmbedding

from app.config import settings
from app.db import get_pool, init_schema
from app.extract import extract_pages, ALLOWED_EXTENSIONS


@dataclass
class Chunk:
    text: str
    source: str
    heading: str
    chunk_id: str
    page: Optional[int] = None  # 1-based page number if the format has pages, else None
    chunk_number: int = 0     # 1-based position of this chunk within its source document


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # similarity, 0..1 (higher = more relevant)


# How many chunks get embedded (and inserted into Postgres) per batch during
# a rebuild. Keeping this bounded caps peak memory regardless of how many
# documents/chunks exist -- important on low-memory hosting tiers, where
# embedding thousands of chunks in a single call/insert can spike RSS.
EMBED_BATCH_SIZE = 32


def _split_into_sections(text: str) -> List[tuple]:
    """Split markdown-ish text on headings so each chunk stays under one
    section, which keeps citations meaningful ('Section 4.2: Expense
    Approval' beats 'chunk 17')."""
    lines = text.splitlines()
    sections = []
    current_heading = "General"
    current_lines: List[str] = []

    for line in lines:
        if re.match(r"^#{1,6}\s+", line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, b) for h, b in sections if b.strip()]


def _chunk_section(body: str, size: int, overlap: int) -> List[str]:
    if len(body) <= size:
        return [body]
    chunks = []
    start = 0
    while start < len(body):
        end = start + size
        chunks.append(body[start:end])
        start = end - overlap
    return chunks


def _fetch_documents() -> List[tuple]:
    """Returns (filename, content_bytes) for every document in the DB."""
    with get_pool().connection() as conn:
        return conn.execute(
            "SELECT filename, content FROM documents ORDER BY filename"
        ).fetchall()


def load_and_chunk_documents() -> List[Chunk]:
    """Pull every stored document out of Postgres, write each one to a
    scratch temp dir just long enough to run the existing
    extract-and-chunk logic (unchanged from the local-disk version), then
    discard the temp files. The DB row is the only persistent copy."""
    chunks: List[Chunk] = []
    with tempfile.TemporaryDirectory() as tmp:
        for filename, content in _fetch_documents():
            if os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS:
                continue
            path = os.path.join(tmp, filename)
            with open(path, "wb") as f:
                f.write(content)
            try:
                pages = extract_pages(path)
            except Exception as e:
                print(f"[ingest] skipping {filename}: {e}")
                continue
            chunk_number = 0  # 1-based position of a chunk within this source doc, for citations
            for page_num, page_text in pages:
                sections = _split_into_sections(page_text) or [("General", page_text)]
                for heading, body in sections:
                    for piece in _chunk_section(body, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
                        piece = piece.strip()
                        if not piece:
                            continue
                        chunk_number += 1
                        chunks.append(
                            Chunk(
                                text=piece,
                                source=filename,
                                heading=heading,
                                chunk_id=f"{filename}::p{page_num}::{heading}::{chunk_number}",
                                page=page_num,
                                chunk_number=chunk_number,
                            )
                        )
    return chunks


def ensure_docs_dir_seeded():
    """On first boot, the `documents` table is empty -- populate it from
    the read-only sample docs bundled with the repo. Once anything exists
    in the table (including an admin having deleted all the samples on
    purpose), this is a no-op: it never overwrites what's in the DB."""
    with get_pool().connection() as conn:
        (count,) = conn.execute("SELECT count(*) FROM documents").fetchone()
    if count:
        return
    if not os.path.isdir(settings.SEED_DOCS_DIR):
        return
    for name in sorted(os.listdir(settings.SEED_DOCS_DIR)):
        src = os.path.join(settings.SEED_DOCS_DIR, name)
        if os.path.isfile(src) and os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS:
            with open(src, "rb") as f:
                save_uploaded_document(name, f.read())


def list_documents() -> List[dict]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT filename, size_bytes, uploaded_at FROM documents ORDER BY filename"
        ).fetchall()
    return [
        {"filename": filename, "size_bytes": size_bytes, "modified": uploaded_at.timestamp()}
        for filename, size_bytes, uploaded_at in rows
    ]


def save_uploaded_document(filename: str, content: bytes) -> str:
    """Save an uploaded file's bytes into the `documents` table under a
    sanitized filename (overwriting any existing document with the same
    name). Returns the final filename used."""
    base = os.path.basename(filename).strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    if not base or os.path.splitext(base)[1].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported or invalid filename: {filename!r}")

    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (filename, content, size_bytes, uploaded_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (filename) DO UPDATE
                SET content = EXCLUDED.content,
                    size_bytes = EXCLUDED.size_bytes,
                    uploaded_at = now()
            """,
            (base, content, len(content)),
        )
    return base


def delete_document(filename: str) -> bool:
    base = os.path.basename(filename)
    with get_pool().connection() as conn:
        cur = conn.execute("DELETE FROM documents WHERE filename = %s", (base,))
        return cur.rowcount > 0


class ComplianceIndex:
    def __init__(self):
        # fastembed runs the same ONNX all-MiniLM-L6-v2 model Chroma's
        # DefaultEmbeddingFunction used (384-dim, matches EMBEDDING_DIM in
        # app/db.py), but without pulling in chromadb's full client/server
        # dependency tree (posthog telemetry, opentelemetry, kubernetes
        # client, grpcio, etc). That import graph alone can add hundreds of
        # MB of resident memory on process start -- easily enough to tip a
        # 512MB Render instance into an OOM kill even before any request
        # arrives, which is the most likely cause of memory-related
        # suspensions on a low-memory plan.
        self._embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        init_schema()

    def _embed(self, texts: List[str]) -> List[list]:
        """Returns one embedding (as a plain list of floats) per input text."""
        return [emb.tolist() for emb in self._embedder.embed(texts)]

    def rebuild(self) -> int:
        """Wipe and re-ingest all documents. Call this on startup and
        whenever procedure documents change. Embeds and inserts in bounded
        batches so peak memory doesn't scale with the total corpus size."""
        chunks = load_and_chunk_documents()

        with get_pool().connection() as conn:
            conn.execute("TRUNCATE TABLE chunks")
            with conn.cursor() as cur:
                for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                    batch = chunks[i : i + EMBED_BATCH_SIZE]
                    embeddings = self._embed([c.text for c in batch])
                    rows = [
                        (c.chunk_id, c.source, c.heading, c.chunk_number, c.page, c.text, emb)
                        for c, emb in zip(batch, embeddings)
                    ]
                    cur.executemany(
                        """
                        INSERT INTO chunks
                            (chunk_id, source, heading, chunk_number, page, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        rows,
                    )
        return len(chunks)

    def count(self) -> int:
        with get_pool().connection() as conn:
            (n,) = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return n

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.TOP_K
        n = self.count()
        if n == 0:
            return []

        query_embedding = self._embed([query])[0]
        with get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, source, heading, chunk_number, page, text,
                       embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, min(top_k, n)),
            ).fetchall()

        out: List[RetrievedChunk] = []
        for chunk_id, source, heading, chunk_number, page, text, dist in rows:
            # pgvector cosine distance is in [0, 2] (1 - cosine_similarity),
            # same convention the old Chroma "cosine" space used.
            similarity = max(0.0, 1.0 - dist / 2.0)
            out.append(
                RetrievedChunk(
                    chunk=Chunk(
                        text=text,
                        source=source,
                        heading=heading,
                        chunk_id=chunk_id,
                        page=page,
                        chunk_number=chunk_number,
                    ),
                    score=similarity,
                )
            )
        return out


# Single shared index instance for the process
index = ComplianceIndex()
