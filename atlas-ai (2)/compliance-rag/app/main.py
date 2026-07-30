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
from app.llm import answer_question

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_docs_dir_seeded()
    # Build the vector index from the procedure documents on startup.
    n = index.rebuild()
    print(f"[startup] Indexed {n} chunks from {settings.DOCS_DIR}")
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
    relevance: float


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    grounded: bool


@app.get("/api/health")
def health():
    return {"status": "ok", "indexed_chunks": index.count()}


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
    """Re-ingest documents from DOCS_DIR without adding/removing any."""
    n = index.rebuild()
    return {"indexed_chunks": n}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved = index.retrieve(req.question, top_k=req.top_k)
    relevant = [r for r in retrieved if r.score >= settings.MIN_RELEVANCE]

    if not relevant:
        return AskResponse(
            answer=(
                "I couldn't find procedure documentation that clearly covers this "
                "question. Please check with your compliance officer or the relevant "
                "policy owner directly, or rephrase the question."
            ),
            citations=[],
            grounded=False,
        )

    answer = answer_question(req.question, relevant)
    citations = [
        Citation(source=r.chunk.source, heading=r.chunk.heading, relevance=round(r.score, 3))
        for r in relevant
    ]
    return AskResponse(answer=answer, citations=citations, grounded=True)


# Static assets (CSS/JS) for the templated pages below
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})


@app.get("/admin")
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})
