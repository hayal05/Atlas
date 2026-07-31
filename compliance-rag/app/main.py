import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

from app.config import settings
from app.rag import (
    index,
    ensure_docs_dir_seeded,
    list_documents,
    save_uploaded_document,
    delete_document,
)
from app.llm import answer_question, answer_general_question

logger = logging.getLogger("atlas_ai")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_docs_dir_seeded()
    # Build the vector index from the procedure documents on startup.
    n = index.rebuild()
    print(f"[startup] Indexed {n} chunks from Neon Postgres")
    yield


def require_admin(x_admin_token: Optional[str] = Header(None)):
    """Guards every admin endpoint. If ADMIN_TOKEN isn't configured at
    all, admin endpoints are disabled outright rather than left open."""
    if not settings.ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin access is not configured on this deployment.")
    if not x_admin_token or x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


app = FastAPI(title="Atlas AI", description="Procedure compliance assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class Citation(BaseModel):
    source: str
    heading: str
    page: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    grounded: bool
    # "grounded": confident match against indexed procedure documents
    # "grounded_partial": only a weak document match -- still prioritized
    #                     over general knowledge, but flagged as weaker
    # "general": no matching document, answered from the model's general
    #            knowledge instead, clearly labeled as such
    # "unavailable": no matching document and general Q&A is disabled/failed
    mode: str = "grounded"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": index.count(),
        "general_qa_enabled": settings.ALLOW_GENERAL_QA,
    }


@app.get("/api/admin/documents")
def admin_list_documents(_: None = Depends(require_admin)):
    return {"documents": list_documents(), "indexed_chunks": index.count()}


@app.post("/api/admin/documents")
async def admin_upload_document(file: UploadFile = File(...), _: None = Depends(require_admin)):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.")
    try:
        saved_name = save_uploaded_document(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    n = index.rebuild()
    return {"filename": saved_name, "indexed_chunks": n}


@app.delete("/api/admin/documents/{filename}")
def admin_delete_document(filename: str, _: None = Depends(require_admin)):
    deleted = delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    n = index.rebuild()
    return {"deleted": filename, "indexed_chunks": n}


@app.post("/api/admin/reindex")
def admin_reindex(_: None = Depends(require_admin)):
    """Re-ingest documents already stored in Neon, without adding/removing any."""
    n = index.rebuild()
    return {"indexed_chunks": n}


def _citations_for(chunks) -> List[Citation]:
    """What was actually cited: document, section, and page. Deduplicated
    by (source, heading, page) rather than by underlying chunk, so two
    chunks from the same section of the same document collapse into one
    reference instead of showing twice. Internal retrieval bookkeeping
    (chunk number, raw similarity score) stays out -- it's not meaningful
    to someone reading the answer, and confidence is already conveyed at
    the response level via `mode`."""
    seen = set()
    out: List[Citation] = []
    for r in chunks:
        c = r.chunk
        key = (c.source, c.heading, c.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(source=c.source, heading=c.heading, page=c.page))
    return out


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        retrieved = index.retrieve(req.question, top_k=req.top_k)
    except Exception:
        # A DB hiccup (Neon waking from idle-suspend, a pool timeout, a
        # dropped connection, etc.) shouldn't surface as a bare 500 --
        # that comes back as non-JSON to the browser and breaks the
        # frontend's res.json() parsing, which is exactly what made this
        # indistinguishable from a generic "network error" client-side.
        # Log it so the real cause is visible in the server logs, and
        # return the same honest, well-formed "unavailable" response the
        # general-Q&A fallback already uses on its own failures.
        logger.exception("Retrieval failed for question: %r", req.question)
        return AskResponse(
            answer=(
                "I couldn't reach the procedure document index just now (this is "
                "usually a temporary database connection issue). Please try again "
                "in a moment, or check with your compliance officer if it persists."
            ),
            citations=[],
            grounded=False,
            mode="unavailable",
        )

    # Tier 1: confident document match -- uploaded documents always get
    # first crack at answering. Tier 2: no confident match, but a weaker
    # one -- still a document, and still preferred over general knowledge.
    # Only when neither tier finds anything does the assistant fall
    # through to general knowledge (or refuse).
    strong = [r for r in retrieved if r.score >= settings.MIN_RELEVANCE]
    weak = [r for r in retrieved if settings.SOFT_RELEVANCE <= r.score < settings.MIN_RELEVANCE]

    top_score = retrieved[0].score if retrieved else None
    logger.info(
        "ask: question=%r top_score=%s strong=%d weak=%d",
        req.question, f"{top_score:.3f}" if top_score is not None else None, len(strong), len(weak),
    )

    if strong:
        answer = answer_question(req.question, strong)
        return AskResponse(answer=answer, citations=_citations_for(strong), grounded=True, mode="grounded")

    if weak:
        answer = answer_question(req.question, weak, weak_match=True)
        return AskResponse(answer=answer, citations=_citations_for(weak), grounded=True, mode="grounded_partial")

    if settings.ALLOW_GENERAL_QA:
        try:
            answer = answer_general_question(req.question)
        except Exception:
            # An LLM host hiccup (timeout, rate limit, bad key, wrong
            # model name, etc.) shouldn't surface as a 500 to the user --
            # fall back to the same honest refusal as when general Q&A is
            # disabled. But DO log it: a bare silent catch here made past
            # failures indistinguishable from "working as intended",
            # which is exactly what made this hard to debug.
            logger.exception("General Q&A call failed for question: %r", req.question)
            return AskResponse(
                answer=(
                    "I couldn't find procedure documentation covering this question, "
                    "and the general-knowledge fallback is temporarily unavailable. "
                    "Please try again shortly, or check with your compliance officer "
                    "or the relevant policy owner directly."
                ),
                citations=[],
                grounded=False,
                mode="unavailable",
            )
        return AskResponse(answer=answer, citations=[], grounded=False, mode="general")

    return AskResponse(
        answer=(
            "I couldn't find procedure documentation that clearly covers this "
            "question. Please check with your compliance officer or the relevant "
            "policy owner directly, or rephrase the question."
        ),
        citations=[],
        grounded=False,
        mode="unavailable",
    )


# Static assets (CSS/JS) for the templated pages below
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})


@app.get("/admin")
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})
