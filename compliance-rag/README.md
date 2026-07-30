# Atlas AI

A retrieval-augmented (RAG) compliance assistant that answers questions about
corporate procedures using open source models — grounded strictly in your
policy documents, with citations, and deployable to [Render](https://render.com).

## Project structure

```
app/                     FastAPI backend
├── main.py              routes: /api/ask, /api/admin/*, /, /admin
├── config.py             env-var settings
├── rag.py                chunking, embedding, Chroma index, retrieval
├── llm.py                calls the open-source LLM
└── extract.py             .pdf/.docx/.md/.txt text extraction

templates/                Jinja2 page templates (structure only)
├── ask.html
└── admin.html

static/                   true static assets, served as-is
├── css/
│   ├── ask.css
│   └── admin.css
└── js/
    ├── ask.js
    └── admin.js

data/docs/                 bundled sample policy documents (seed data)
requirements.txt
render.yaml
README.md
```

At runtime, a separate writable location (`local_data/` locally, or the
mounted disk on Render) holds the live document set and vector index —
see "Admin: uploading documents" below.

## How it works

1. Policy documents in `data/docs/` (markdown, `.txt`, `.pdf`, `.docx`) are
   chunked by section on startup.
2. Chunks are embedded locally with Chroma's built-in ONNX MiniLM embedding
   function — no API key needed, runs on CPU, and has no torch/transformers
   dependency, which keeps this comfortably inside low-memory hosting tiers
   — and stored in a persistent Chroma vector index.
3. A question is embedded the same way and matched against the index to
   pull back the most relevant chunks.
4. Those chunks (not the raw question alone) are handed to an open source
   LLM with instructions to answer *only* from the provided text and cite
   the source section. If nothing relevant is found, it says so instead
   of guessing.

## Why the LLM call goes to an external endpoint

Render's standard plans are CPU-only, so self-hosting a capable open
source LLM's weights *inside this service* isn't practical. Instead, the
generation step calls out to an OpenAI-compatible endpoint that serves
open source models — you're still using an open source model (Llama,
Mixtral, Qwen, etc.), it's just hosted somewhere with a GPU. This is a
one-line config change (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`), so
you can point it at:

- **Groq** — fast, generous free tier, serves Llama 3.3, Mixtral, Gemma
- **Together AI** / **Fireworks** / **DeepInfra** — broader open source
  model catalogs
- **Your own vLLM or Ollama server** — if you have GPU infrastructure
  and want the model fully in-house, run it there and point `LLM_BASE_URL`
  at it (Ollama supports an OpenAI-compatible endpoint out of the box)

The embedding model, in contrast, runs directly inside this service on
CPU — no external calls, so document content never leaves your
infrastructure at the retrieval stage.

## Local development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: at minimum set LLM_API_KEY (a free Groq key works well)
export $(cat .env | grep -v '^#' | xargs)

uvicorn app.main:app --reload
```

Visit `http://localhost:8000` for the assistant, or `http://localhost:8000/admin`
to upload documents (see below).

Two sample policies in `data/docs/` are copied into the live, writable
`DOCS_DIR` the first time the app starts, so it works out of the box.
From then on, manage documents either by dropping files directly into
`DOCS_DIR` and restarting, or through the `/admin` upload page — no
restart needed for uploads made that way.

## Deploying to Render

This repo includes a `render.yaml`, so the easiest path is Render's
**Blueprint** deploy:

1. Push this repo to GitHub/GitLab.
2. In Render, choose **New > Blueprint** and point it at the repo.
3. Render will read `render.yaml` and provision a web service with a
   1GB persistent disk (for the Chroma index).
4. Set the `LLM_API_KEY` environment variable in the Render dashboard
   (it's marked `sync: false` in the blueprint so it's never committed).
5. Deploy. On first boot the app indexes everything in `data/docs/`.

If you'd rather configure manually instead of using the blueprint:
Runtime = Python 3, Build command = `pip install -r requirements.txt`,
Start command = `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

### About the persistent disk

Render's filesystem is ephemeral on redeploys unless you attach a disk
(the blueprint does this). The index also fully rebuilds from
`data/docs/` on every startup regardless, so the disk is a nice-to-have
for warm restarts, not a hard requirement — if you'd rather keep the
service disk-free, remove the `disk:` block from `render.yaml` and set
`CHROMA_DIR` to a temp path.

### If the deploy fails with "Exited with status 137"

That exit code means the process was killed for using too much memory
(OOM kill) — it happens before your app ever binds `$PORT`, so you'll
also see Render logging "No open ports detected" right before it.

This app is built to avoid that: embeddings run through Chroma's
built-in ONNX MiniLM function rather than `sentence-transformers`/torch,
which keeps the memory footprint small enough for Render's Starter
plan (512MB). If you've since added a heavier dependency (a bigger
local model, `torch`, etc.) and hit this again, either trim that
dependency or move to a plan with more RAM.

## Admin: uploading documents

Set `ADMIN_TOKEN` (any long random string) in your environment, then visit
`/admin` on your deployment. From there an admin can:

- Drag-and-drop or select a file to upload (`.md`, `.txt`, `.pdf`, `.docx`)
- See every currently indexed document and its size
- Remove a document
- Every upload or removal triggers an automatic reindex

If `ADMIN_TOKEN` is left unset, all `/api/admin/*` endpoints return `503`
and the admin page stays locked — there's no "open by default" state.

The token is sent as an `X-Admin-Token` header on each request (stored
in the browser's `sessionStorage`, not a cookie, so it clears when the
tab closes). This is a lightweight shared-secret scheme suitable for a
small number of trusted admins; if you need per-user accounts, audit
trails of *who* uploaded what, or SSO, put this behind your normal
company auth (e.g. an internal reverse proxy or a proper auth
middleware) instead of relying on the token alone.

Uploaded documents are stored on the persistent disk (`DOCS_DIR`), so
they survive redeploys. The two sample policies are copied there once
on first boot and are otherwise ordinary documents — an admin can
delete or replace them like anything else.

PDF and DOCX are converted to text on upload (page-by-page for PDFs,
heading-aware for DOCX) so retrieval and citations work the same way
as for markdown source files.

## API

- `GET /api/health` — status and indexed chunk count
- `POST /api/ask` — `{"question": "..."}` → grounded answer + citations
- `GET /api/admin/documents` — list indexed documents *(requires `X-Admin-Token`)*
- `POST /api/admin/documents` — upload a document, multipart `file=` field *(requires `X-Admin-Token`)*
- `DELETE /api/admin/documents/{filename}` — remove a document *(requires `X-Admin-Token`)*
- `POST /api/admin/reindex` — re-ingest without adding/removing files *(requires `X-Admin-Token`)*

## Taking this further

This is a working starting point, not a production compliance system.
Before relying on it for real decisions, you'll likely want:

- **Access control** — some procedures are role- or region-specific;
  add auth and filter retrieval accordingly.
- **An evaluation set** — a list of real questions with known-correct
  answers, checked whenever you change the chunking, embedding model,
  or LLM, so you catch retrieval regressions before employees do.
- **Audit logging** — store every question, retrieved sources, and
  answer for compliance review.
- **A real ingestion pipeline** — for more than a handful of documents,
  add PDF/DOCX parsing (see `unstructured` or similar) and a way to
  version documents as they're revised.
- **A confidence-based escalation path** — the `MIN_RELEVANCE` threshold
  is a coarse first pass; consider having genuinely ambiguous or
  high-stakes questions routed to a human compliance officer by default.
