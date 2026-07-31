"""
Central configuration, all overridable via environment variables so the
same code runs locally, on Render, or anywhere else.
"""
import os


class Settings:
    # --- Embedding: Chroma's built-in ONNX MiniLM embedding function is used
    # (see app/rag.py). It's fixed, not configurable here, deliberately --
    # it has no torch/transformers dependency, which matters for staying
    # inside low-memory hosting tiers (that's what OOM-killed earlier
    # sentence-transformers-based deploys with exit status 137). ---

    # --- Persistent storage: Neon Postgres. Both the uploaded documents
    # AND the vector index (via pgvector) live in this database, so they
    # survive redeploys, restarts, and moving to a new Render instance --
    # unlike Render's local disk, which is wiped on redeploy. Get this
    # connection string from the Neon console (Project -> Connect); set
    # it in the Render dashboard (or .env locally). ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # --- Read-only sample documents bundled with the repo. Copied into
    # the database on first boot only (if the `documents` table is
    # empty), so the app has something to answer questions about out of
    # the box, without overwriting anything an admin has since uploaded
    # or deleted. ---
    SEED_DOCS_DIR: str = os.getenv("SEED_DOCS_DIR", "./data/docs")

    # --- Admin auth. Required to upload/delete documents or trigger a
    # reindex. Set this in your environment (Render dashboard, or .env
    # locally) -- if it's left blank, admin endpoints are disabled
    # entirely rather than left open. ---
    ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "")

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "15"))

    # --- LLM (open-source model served behind an OpenAI-compatible API) ---
    # Render has no GPUs on its standard plans, so instead of hosting the
    # model's weights in this service, we call out to an inference host
    # that serves open source models over an OpenAI-compatible endpoint.
    # Groq, Together AI, Fireworks, and DeepInfra all work here unchanged --
    # just swap the base URL, key, and model name. Point this at a local
    # Ollama/vLLM server instead if you're self-hosting elsewhere.
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Minimum similarity to trust a retrieved chunk. Below this, the
    # system says it doesn't have grounded coverage rather than guessing.
    #
    # IMPORTANT -- this needs calibrating against your own embedding
    # model + corpus, not left at a guessed default. Chroma's default
    # ONNX MiniLM embeddings have a fairly high "noise floor": even a
    # totally unrelated query (e.g. "What is Australia" against a
    # travel-expense policy) can score ~0.50-0.52 similarity, not
    # anywhere near 0. If MIN_RELEVANCE sits below that floor, EVERY
    # question gets treated as a confident document match and general
    # Q&A never triggers -- which is exactly what "hello"/"what is
    # Australia" got misrouted by before this was raised.
    #
    # To calibrate for your own deployment: ask a couple of genuinely
    # off-topic questions and a couple of genuinely on-topic ones, note
    # the top_score the server logs for each (see the "ask:" log line in
    # app/main.py), and set MIN_RELEVANCE comfortably above the
    # off-topic scores and SOFT_RELEVANCE somewhere between the two
    # clusters. The value below is a conservative starting point, not a
    # verified-correct one -- override it via the MIN_RELEVANCE /
    # SOFT_RELEVANCE env vars once you have real numbers. ---
    MIN_RELEVANCE: float = float(os.getenv("MIN_RELEVANCE", "0.60"))

    # --- Uploaded documents are always given first crack at answering a
    # question. SOFT_RELEVANCE is a second, lower bar: chunks that don't
    # clear MIN_RELEVANCE but do clear this are still a *document* match,
    # just a weaker one, and are used in preference to the general-
    # knowledge fallback below (mode: "grounded_partial"). Only when
    # nothing clears even this bar does the assistant fall through to
    # general knowledge (or refuse, if that's disabled). Must be <=
    # MIN_RELEVANCE. See the MIN_RELEVANCE comment above -- calibrate
    # this the same way. ---
    SOFT_RELEVANCE: float = float(os.getenv("SOFT_RELEVANCE", "0.55"))

    # --- General Q&A fallback. When a question doesn't match any indexed
    # procedure documents (score < MIN_RELEVANCE), the assistant can either
    # (a) fall back to the model's general knowledge, clearly labeled as
    # ungrounded, or (b) refuse and point the user at a human, as before.
    # This calls the SAME external LLM host already configured above, so it
    # adds no extra memory/process footprint on a free-tier web service --
    # only an extra outbound HTTP call. Default on; set to "false" to
    # restore the original strict, docs-only behavior. ---
    ALLOW_GENERAL_QA: bool = os.getenv("ALLOW_GENERAL_QA", "true").strip().lower() in ("1", "true", "yes", "on")

    # --- Caps below keep each request cheap and fast, which matters on
    # Render's free tier: a single worker process, 512MB RAM, and a service
    # that spins down after 15 minutes idle (so the first request after a
    # cold start is already slow -- a hung or oversized LLM call on top of
    # that can tie up the only worker and time out the request). ---
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "600"))


settings = Settings()
