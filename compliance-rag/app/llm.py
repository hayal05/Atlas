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

# Used only when no indexed procedure document is a good match for the
# question (see app/main.py). Kept deliberately separate from
# SYSTEM_PROMPT above so the strict, docs-only behavior for in-scope
# questions is never weakened -- this prompt governs a clearly different,
# clearly labeled mode.
GENERAL_SYSTEM_PROMPT = """You are Atlas AI, a workplace assistant. The user's question does NOT match \
any of this company's indexed procedure documents, so you are answering from general knowledge \
instead of documented company policy.

Rules you must follow:
1. Answer helpfully and accurately using your general knowledge.
2. Make it unmistakable that this is NOT sourced from the company's procedure documents -- \
open by briefly noting that, then answer.
3. If the question sounds like it's actually about this company's internal policy, procedure, \
disciplinary process, or compliance obligations, say plainly that you don't have that in the \
indexed documents and recommend they check with their compliance officer or the relevant policy \
owner, rather than guessing at what an internal policy might say.
4. Do not present general knowledge as if it were the company's official position.
5. Be concise and practical. Use short paragraphs or bullet points.
"""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(no relevant procedure excerpts found)"
    parts = []
    for rc in chunks:
        c = rc.chunk
        location = f"page {c.page}, " if c.page is not None else ""
        parts.append(
            f"--- Source: {c.source} | {location}Section: {c.heading} | "
            f"content #{c.chunk_number} (relevance: {rc.score:.2f}) ---\n{c.text}"
        )
    return "\n\n".join(parts)


def format_references(chunks: List[RetrievedChunk]) -> str:
    """Deterministic reference footer listing document name, page number,
    and content (chunk) number for every excerpt actually used -- built in
    code rather than left to the model, so it can't be misquoted or
    hallucinated. Appended to the bottom of grounded/partial answers."""
    if not chunks:
        return ""
    lines = ["", "---", "**References**"]
    # De-dupe in case retrieval returns the same chunk twice; keep order.
    seen = set()
    n = 0
    for rc in chunks:
        c = rc.chunk
        key = (c.source, c.page, c.chunk_number)
        if key in seen:
            continue
        seen.add(key)
        n += 1
        location = f"page {c.page}, " if c.page is not None else ""
        lines.append(f"{n}. {c.source} — {location}§ {c.heading} (content #{c.chunk_number})")
    return "\n".join(lines)


def answer_question(question: str, chunks: List[RetrievedChunk], weak_match: bool = False) -> str:
    if not settings.LLM_API_KEY:
        return (
            "The LLM API key isn't configured yet. Set LLM_API_KEY (and optionally "
            "LLM_BASE_URL / LLM_MODEL) in your environment to enable answers. "
            f"In the meantime, here are the most relevant procedure excerpts I found:\n\n"
            + _build_context(chunks)
        )

    client = OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )

    context = _build_context(chunks)
    weak_note = (
        "\n\nNote: these excerpts are only a WEAK match for the question (below the "
        "confident-match threshold). Open by flagging that plainly, answer only what "
        "they actually support, and say what's missing rather than filling gaps."
        if weak_match else ""
    )
    user_prompt = f"""Procedure excerpts:
{context}

Employee question: {question}

Answer using only the excerpts above, with inline citations to document and section.{weak_note}"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content
    return answer + format_references(chunks)


def answer_general_question(question: str) -> str:
    """Fallback used when no indexed document clearly covers the question
    (see app/main.py). Answers from the model's general knowledge instead
    of refusing outright. Reuses the same configured LLM host/client as
    answer_question -- no extra dependency or process, just one more
    outbound API call, so it's free-tier friendly."""
    if not settings.LLM_API_KEY:
        return (
            "The LLM API key isn't configured yet, so I can't answer questions outside "
            "the indexed procedure documents either. Set LLM_API_KEY in your environment "
            "to enable this."
        )

    client = OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
