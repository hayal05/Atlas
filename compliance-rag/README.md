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
   split by page (real pages for PDF; detected page breaks for DOCX; no
   page concept for markdown/txt), then chunked by section within each
   page on startup. Every chunk keeps its source filename, page number
   (if any), section heading, and a sequential **content number** — its
   1-based position among that document's chunks — so any excerpt can be
   pointed back to precisely.
2. Chunks are embedded locally with Chroma's built-in ONNX MiniLM embedding
   function — no API key needed, runs on CPU, and has no torch/transformers
   dependency, which keeps this comfortably inside low-memory hosting tiers
   — and stored in a persistent Chroma vector index.
3. A question is embedded the same way and matched against the index to
   pull back the most relevant chunks. **Uploaded documents always get
   first crack at answering** — general knowledge is only ever a
   fallback, in two tiers:
   - **Confident match** (`score >= MIN_RELEVANCE`) → answered from those
     excerpts, `mode: "grounded"`.
   - **Weak match** (`SOFT_RELEVANCE <= score < MIN_RELEVANCE`) → still a
     *document* match, just a shakier one, so it's still preferred over
     general knowledge — answered from those excerpts with an explicit
     "this is only a partial match" note, `mode: "grounded_partial"`.
   - **No match at all** (`score < SOFT_RELEVANCE` for everything) → only
     now does it fall back to general knowledge, or refuse — see below.
4. Grounded excerpts (either tier) are handed to an open source LLM with
   instructions to answer *only* from the provided text and cite the
   source section inline (e.g. "(Expense Policy, Section 4.2)"). The
   backend separately returns a deduplicated, structured `citations` list
   — document name, section heading, and page number where applicable —
   built in code from the retrieved chunks, not left to the model, so it
   can't be misquoted. This is the single source of truth for references
   and is what renders as chips under the answer in the UI; there's no
   second copy of the same list appended as text.
5. If nothing clears even the weak-match bar, and `ALLOW_GENERAL_QA=true`
   (the default), Atlas AI answers from the model's general knowledge
   instead of refusing outright — clearly labeled in the UI ("General
   knowledge · not from your policy documents") and in the API response
   (`mode: "general"`, `grounded: false`) so it's never mistaken for
   documented policy. Set `ALLOW_GENERAL_QA=false` to go back to the
   original strict, docs-only behavior (a message pointing the person to
   their compliance officer).

   This fallback reuses the *same* external LLM host already configured
   for grounded answers, so it costs one extra outbound API call, not one
   extra process or model — it doesn't change the service's memory
   footprint, which matters if you're on Render's free tier (see below).
   `LLM_TIMEOUT_SECONDS` and `LLM_MAX_TOKENS` cap how long/large any single
   LLM call can be, so a slow or oversized response can't tie up the
   service's only worker.

### A note on DOCX page numbers

PDF page numbers are exact (pypdf reads real pages). DOCX has no stored
page count — pagination is a rendering detail — so page numbers for DOCX
are only recovered when the author inserted explicit page breaks
(Ctrl+Enter in Word); Atlas AI detects those and numbers accordingly. A
DOCX with no explicit breaks (e.g. relying on Word's automatic reflow) has
no page numbers to cite, so its citations show section heading and
content number only, no page — same as markdown/`.txt`.

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
# edit .env: set LLM_API_KEY (a free Groq key works well) and DATABASE_URL
# (a free Neon project works well: https://neon.tech -- see below)
export $(cat .env | grep -v '^#' | xargs)

uvicorn app.main:app --reload
```

Visit `http://localhost:8000` for the assistant, or `http://localhost:8000/admin`
to upload documents (see below).

Two sample policies in `data/docs/` are copied into the `documents` table
in your Neon database the first time the app starts, so it works out of
the box. From then on, manage documents through the `/admin` upload
page — uploads and deletes are written straight to Neon and take effect
immediately, no restart needed.

## Deploying to Render

This repo includes a `render.yaml`, so the easiest path is Render's
**Blueprint** deploy:

1. Create a free Neon project at https://neon.tech and copy its connection
   string (Project -> Connect). It looks like
   `postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require`.
   Neon's pgvector extension is enabled automatically by the app on
   startup (`CREATE EXTENSION IF NOT EXISTS vector`) — nothing to do by
   hand in the Neon console.
2. Push this repo to GitHub/GitLab.
3. In Render, choose **New > Blueprint** and point it at the repo.
4. Render will read `render.yaml` and provision a web service — no disk
   needed, since documents and the vector index both live in Neon now.
5. Set `LLM_API_KEY` and `DATABASE_URL` in the Render dashboard (both are
   marked `sync: false` in the blueprint so neither is ever committed).
6. Deploy. On first boot the app seeds `data/docs/` into Neon (only if
   the `documents` table is empty) and builds the index from there.

If you'd rather configure manually instead of using the blueprint:
Runtime = Python 3, Build command = `pip install -r requirements.txt`,
Start command = `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

### Why redeploys no longer erase anything

Render's local filesystem is wiped on every redeploy, restart, and
instance move. Earlier versions of this app stored uploaded documents
and the Chroma vector index on that filesystem (optionally backed by a
Render persistent disk), so anything uploaded through `/admin` could be
lost the moment the disk config changed or the service moved instances.

Now both live in Neon Postgres instead: uploaded file bytes go in a
`documents` table, and chunk embeddings go in a `chunks` table using the
`pgvector` extension, queried with a plain `ORDER BY embedding <=> ...`
similarity search. Neon is external to Render entirely, so none of it is touched by a
redeploy — a redeploy just reconnects to the same database. There's no
in-memory index either; every `/api/ask` query runs its similarity
search directly against the `chunks` table in Neon. The one thing that
still happens on every startup is `index.rebuild()`, which re-chunks and
re-embeds all documents already in Neon — cheap, and it means a code
change to chunking/embedding takes effect on the next deploy without a
manual reindex.

### Running on Render's free tier

`render.yaml` as shipped uses `plan: starter`. To run on the free tier
instead, change `plan: starter` to `plan: free` — no other change is
needed, since storage no longer depends on a Render disk. Know the other
trade-offs that come with the free tier:

- **Spins down after ~15 minutes idle** — the first request after that
  triggers a cold start (service boot + reconnect to Neon + re-embedding
  whatever's indexed), which can take a while depending on how much
  you've indexed.
- **Single instance, limited CPU/RAM (512MB)** — this is why generation
  is offloaded to an external LLM host (see above) rather than
  self-hosted, and why `LLM_TIMEOUT_SECONDS`/`LLM_MAX_TOKENS` exist: one
  slow request shouldn't be able to block the only worker for other
  users.
- **Neon's own free-tier limits** — free Neon projects auto-suspend
  after inactivity too, and have storage/compute caps of their own;
  check the current limits at https://neon.tech/docs if you expect real
  traffic.

Everything else in this README (admin uploads, the general Q&A fallback,
etc.) works the same on free as on Starter.

### If the deploy fails with "Exited with status 137"

That exit code means the process was killed for using too much memory
(OOM kill) — it happens before your app ever binds `$PORT`, so you'll
also see Render logging "No open ports detected" right before it.

This app is built to avoid that: embeddings run through Chroma's
built-in ONNX MiniLM function rather than `sentence-transformers`/torch
(the vectors themselves are stored in Neon via pgvector, not in Chroma),
which keeps the memory footprint small enough for Render's Starter plan
(512MB). If you've since added a heavier dependency (a bigger local
model, `torch`, etc.) and hit this again, either trim that dependency or
move to a plan with more RAM.

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

Uploaded documents are stored in Neon (the `documents` table), so they
survive redeploys, restarts, and instance moves. The two sample policies
are copied there once on first boot and are otherwise ordinary
documents — an admin can delete or replace them like anything else.

PDF and DOCX are converted to text on upload (page-by-page for PDFs,
heading-aware for DOCX) so retrieval and citations work the same way
as for markdown source files.

## API

- `GET /api/health` — status, indexed chunk count, and whether the general
  Q&A fallback is enabled (`general_qa_enabled`)
- `POST /api/ask` — `{"question": "...", "top_k": 5}` → `{"answer", "citations",
  "grounded", "mode"}`, where:
  - `mode` is `"grounded"` (confident document match), `"grounded_partial"`
    (weaker document match, still preferred over general knowledge),
    `"general"` (no document match, answered from general knowledge,
    `grounded: false`), or `"unavailable"` (no match and the fallback is
    disabled or failed, `grounded: false`)
  - `citations` is a deduplicated list of `{source, heading, page}` for
    every document section actually used — `page` is `null` when the
    format/file has no page concept (markdown, `.txt`, or a DOCX with no
    explicit page breaks). Internal retrieval bookkeeping (chunk number,
    raw similarity score) isn't included; confidence is already conveyed
    by `mode`.
  - `answer` is just the model's text with inline citations (e.g.
    "(Expense Policy, Section 4.2)") — no separate appended reference
    list, since `citations` already covers that structurally without
    duplicating it as text
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
