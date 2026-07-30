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

    # --- Persistent storage root. On Render this is the mounted disk from
    # render.yaml; locally it's a folder next to the code. Both the vector
    # index AND the uploaded documents live under here, so admin uploads
    # survive redeploys. ---
    PERSIST_ROOT: str = os.getenv(
        "PERSIST_ROOT", "/opt/render/project/data" if os.getenv("RENDER") else "./local_data"
    )

    # --- Vector store ---
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", os.path.join(PERSIST_ROOT, "chroma_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "procedures")

    # --- Source documents to ingest on startup. This is the WRITABLE,
    # persistent location -- admin uploads land here. ---
    DOCS_DIR: str = os.getenv("DOCS_DIR", os.path.join(PERSIST_ROOT, "docs"))

    # --- Read-only sample documents bundled with the repo. Copied into
    # DOCS_DIR on first boot only (if DOCS_DIR is empty), so the app has
    # something to answer questions about out of the box, without
    # overwriting anything an admin has since uploaded or deleted. ---
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
    # system says it doesn't have grounded coverage rather than guessing. ---
    MIN_RELEVANCE: float = float(os.getenv("MIN_RELEVANCE", "0.25"))


settings = Settings()
