"""
Thin client around an OpenAI-compatible chat completions endpoint.

This is intentionally provider-agnostic: point LLM_BASE_URL at Groq,
Together AI, Fireworks, DeepInfra, a self-hosted vLLM/Ollama server, or
anything else that speaks the OpenAI chat completions schema, and set
LLM_MODEL to whichever open source model that host is serving
(e.g. llama-3.3-70b-versatile, mixtral-8x7b, qwen2.5-72b-instruct).
"""
from typing import List

from openai import OpenAI

from app.config import settings
from app.rag import RetrievedChunk

SYSTEM_PROMPT = """You are Atlas AI, a corporate procedure compliance assistant. You answer \
questions ONLY using the procedure excerpts provided in the context below. \

Rules you must follow:
1. Base your answer strictly on the provided excerpts. Do not use outside knowledge \
of laws, regulations, or "typical" corporate policy.
2. Every claim you make must be traceable to a specific excerpt. Reference the \
document and section by name inline, e.g. "(Expense Policy, Section 4.2)".
3. If the excerpts do not clearly answer the question, say so plainly and recommend \
the person contact their compliance officer or the policy owner. Do not guess or fill gaps.
4. If excerpts conflict, point out the conflict rather than picking one silently.
5. You are providing guidance based on documented procedure, not a legal or compliance \
determination. For ambiguous, high-stakes, or disciplinary matters, say the person should \
escalate to a human compliance officer.
6. Be concise and practical. Use short paragraphs or bullet points.
"""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(no relevant procedure excerpts found)"
    parts = []
    for rc in chunks:
        parts.append(
            f"--- Source: {rc.chunk.source} | Section: {rc.chunk.heading} "
            f"(relevance: {rc.score:.2f}) ---\n{rc.chunk.text}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, chunks: List[RetrievedChunk]) -> str:
    if not settings.LLM_API_KEY:
        return (
            "The LLM API key isn't configured yet. Set LLM_API_KEY (and optionally "
            "LLM_BASE_URL / LLM_MODEL) in your environment to enable answers. "
            f"In the meantime, here are the most relevant procedure excerpts I found:\n\n"
            + _build_context(chunks)
        )

    client = OpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)

    context = _build_context(chunks)
    user_prompt = f"""Procedure excerpts:
{context}

Employee question: {question}

Answer using only the excerpts above, with inline citations to document and section."""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
