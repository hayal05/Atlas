"""
Neon Postgres connection + schema.

This is the persistence layer that replaces the old local-disk storage
(uploaded files under DOCS_DIR, Chroma's on-disk index under CHROMA_DIR).
Render's filesystem is wiped on every redeploy -- a Neon database is not,
so uploaded documents and the vector index now survive redeploys,
restarts, and moving to a new Render instance.

Embeddings are stored directly in Postgres using the pgvector extension
(Neon has this available -- no separate vector DB needed).
"""
import logging

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from app.config import settings

logger = logging.getLogger("atlas_ai")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 -- matches Chroma's DefaultEmbeddingFunction

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    conn.autocommit = True
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)


def _probe_connection() -> None:
    """Try ONE direct connection and run the EXACT same setup steps the
    pool's `configure` callback runs, before the pool gets involved at all.

    Uses print(flush=True) instead of the logger -- this runs during
    `import app.rag` at module load time, which happens BEFORE main.py
    gets to the lines that attach a handler to the "atlas_ai" logger, so
    logger.info/.exception calls here are silently dropped. print()
    always shows up in Render's log tab regardless of logging setup.

    Without this, a bad DATABASE_URL / a failing CREATE EXTENSION / a
    failing register_vector just shows up as a generic
    ``psycopg_pool.PoolTimeout: couldn't get a connection after 30.00 sec``
    -- the pool retries the same failing setup in the background and
    swallows the actual cause."""
    import traceback

    masked = settings.DATABASE_URL
    if "@" in masked:
        creds, _, rest = masked.partition("@")
        scheme, _, _ = creds.partition("://")
        masked = f"{scheme}://***:***@{rest}"

    try:
        with psycopg.connect(settings.DATABASE_URL, connect_timeout=10) as conn:
            conn.autocommit = True
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
        print(f"[db probe] connected OK to {masked}", flush=True)
    except Exception:
        print(f"[db probe] FAILED connecting to {masked}:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not settings.DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Point it at your Neon connection string "
                "(Render dashboard -> Environment -> DATABASE_URL, or .env locally). "
                "Get it from the Neon console: Project -> Connect."
            )
        _probe_connection()
        _pool = ConnectionPool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=5,
            configure=_configure,
            open=True,
        )
    return _pool


def init_schema() -> None:
    """Create the extension/tables if they don't exist yet. Safe to call
    on every startup -- IF NOT EXISTS everywhere, never drops data."""
    with get_pool().connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                filename    TEXT PRIMARY KEY,
                content     BYTEA NOT NULL,
                size_bytes  INTEGER NOT NULL,
                uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      TEXT PRIMARY KEY,
                source        TEXT NOT NULL,
                heading       TEXT NOT NULL,
                chunk_number  INTEGER NOT NULL,
                page          INTEGER,
                text          TEXT NOT NULL,
                embedding     VECTOR({EMBEDDING_DIM}) NOT NULL
            )
            """
        )
        # No ANN index (hnsw/ivfflat) on purpose: this app's corpus is a
        # handful of procedure documents, so an exact brute-force scan
        # over `chunks` is plenty fast and avoids pinning a specific
        # pgvector index-build version. Add one later (e.g.
        # `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`)
        # if the document set grows into the tens of thousands of chunks.
        pass
