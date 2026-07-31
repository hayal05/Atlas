"""
Ingestion + retrieval pipeline.

Docs (markdown/txt/pdf/docx) in DOCS_DIR are split into heading-aware
chunks, embedded locally with Chroma's built-in ONNX MiniLM embedding
function (no torch/transformers dependency -- important for staying
inside low-memory hosting tiers), and stored in a persistent Chroma
collection. At query time we embed the question and pull back the
top-K most similar chunks with their source citations.
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings
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


def load_and_chunk_documents(docs_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    paths = sorted(
        p for p in glob.glob(os.path.join(docs_dir, "**", "*"), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in ALLOWED_EXTENSIONS
    )
    for path in paths:
        try:
            pages = extract_pages(path)
        except Exception as e:
            print(f"[ingest] skipping {path}: {e}")
            continue
        source_name = os.path.relpath(path, docs_dir)
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
                            source=source_name,
                            heading=heading,
                            chunk_id=f"{source_name}::p{page_num}::{heading}::{chunk_number}",
                            page=page_num,
                            chunk_number=chunk_number,
                        )
                    )
    return chunks


def ensure_docs_dir_seeded():
    """On first boot, DOCS_DIR (the persistent, writable location) is
    empty -- populate it from the read-only sample docs bundled with the
    repo. Once anything exists there (including an admin having deleted
    all the samples on purpose), this is a no-op."""
    os.makedirs(settings.DOCS_DIR, exist_ok=True)
    if os.listdir(settings.DOCS_DIR):
        return
    if not os.path.isdir(settings.SEED_DOCS_DIR):
        return
    import shutil

    for name in os.listdir(settings.SEED_DOCS_DIR):
        src = os.path.join(settings.SEED_DOCS_DIR, name)
        dst = os.path.join(settings.DOCS_DIR, name)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)


def list_documents() -> List[dict]:
    os.makedirs(settings.DOCS_DIR, exist_ok=True)
    out = []
    for name in sorted(os.listdir(settings.DOCS_DIR)):
        path = os.path.join(settings.DOCS_DIR, name)
        if os.path.isfile(path):
            stat = os.stat(path)
            out.append({"filename": name, "size_bytes": stat.st_size, "modified": stat.st_mtime})
    return out


def save_uploaded_document(filename: str, content: bytes) -> str:
    """Save an uploaded file's bytes into DOCS_DIR under a sanitized
    filename. Returns the final filename used."""
    base = os.path.basename(filename).strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    if not base or os.path.splitext(base)[1].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported or invalid filename: {filename!r}")

    os.makedirs(settings.DOCS_DIR, exist_ok=True)
    dest = os.path.join(settings.DOCS_DIR, base)
    with open(dest, "wb") as f:
        f.write(content)
    return base


def delete_document(filename: str) -> bool:
    base = os.path.basename(filename)
    path = os.path.join(settings.DOCS_DIR, base)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True


class ComplianceIndex:
    def __init__(self):
        self._embedder = embedding_functions.DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def rebuild(self, docs_dir: str = None) -> int:
        """Wipe and re-ingest all documents. Call this on startup and
        whenever procedure documents change."""
        docs_dir = docs_dir or settings.DOCS_DIR
        chunks = load_and_chunk_documents(docs_dir)

        try:
            self._client.delete_collection(settings.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

        if not chunks:
            return 0

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "heading": c.heading,
                    "chunk_number": c.chunk_number,
                    # Chroma metadata can't store None, so use 0 as "no page"
                    "page": c.page if c.page is not None else 0,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.TOP_K
        if self.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(top_k, self.count()))

        out: List[RetrievedChunk] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        ids = results["ids"][0]
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            # Chroma cosine "distance" -> similarity score in [0, 1]
            similarity = max(0.0, 1.0 - dist / 2.0)
            page = meta.get("page") or None  # 0 was stored as "no page" -> None
            out.append(
                RetrievedChunk(
                    chunk=Chunk(
                        text=doc,
                        source=meta["source"],
                        heading=meta["heading"],
                        chunk_id=cid,
                        page=page,
                        chunk_number=meta.get("chunk_number", 0),
                    ),
                    score=similarity,
                )
            )
        return out


# Single shared index instance for the process
index = ComplianceIndex()
