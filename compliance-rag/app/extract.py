"""
Extract plain text from the document formats admins are allowed to upload.
Markdown/text pass through unchanged; PDF and DOCX are converted to text
so the same chunking logic in rag.py can treat everything uniformly.

Unlike a plain single-string extraction, this returns text broken out by
*page* wherever the format actually has pages, so citations can reference
"page 3" the same way a person reading the source PDF would -- see
extract_pages() below.
"""
import os
from typing import List, Optional, Tuple

ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


def extract_pages(path: str) -> List[Tuple[Optional[int], str]]:
    """Returns a list of (page_number, text) tuples in document order.

    page_number is 1-based and reflects a real page for formats that have
    them (PDF always; DOCX when explicit page breaks are present in the
    file). It's None for formats/files with no page concept (Markdown,
    .txt, and DOCX files with no detected page breaks) -- callers should
    treat None as "not applicable", not "page 1".
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".md", ".txt"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [(None, f.read())]

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    if ext == ".docx":
        return _extract_docx_pages(path)

    raise ValueError(f"Unsupported file type: {ext}")


def _extract_docx_pages(path: str) -> List[Tuple[Optional[int], str]]:
    """DOCX has no stored page count -- pagination depends on the reader's
    rendering. We can still recover *explicit* page breaks the author
    inserted (Ctrl+Enter in Word), which python-docx doesn't surface
    directly but does leave in each run's underlying XML as a
    <w:br w:type="page"/>. If we find at least one, we split on those and
    number the resulting pages; otherwise we fall back to a single
    page-less block, same as before."""
    import docx

    doc = docx.Document(path)

    page_breaks_found = False
    pages: List[List[str]] = [[]]

    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        prefix = "## " if (style.startswith("heading") or style == "title") else ""

        # A single paragraph can itself contain a page break mid-run, so we
        # walk runs rather than checking once per paragraph.
        para_has_break = False
        for run in para.runs:
            br_elements = run._element.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
            )
            for br in br_elements:
                if br.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type"
                ) == "page":
                    para_has_break = True

        text = para.text.strip()
        if text:
            pages[-1].append(f"{prefix}{text}")

        if para_has_break:
            page_breaks_found = True
            pages.append([])

    # Drop any trailing empty page created by a break at the very end.
    if pages and not pages[-1]:
        pages.pop()

    if not page_breaks_found:
        all_lines = [line for page_lines in pages for line in page_lines]
        return [(None, "\n\n".join(all_lines))]

    return [(i + 1, "\n\n".join(lines)) for i, lines in enumerate(pages) if lines]


def extract_text(path: str) -> str:
    """Back-compat helper: full document text with page breaks kept as
    '## Page N' markdown headings, for any caller that just wants a single
    string (e.g. quick inspection)."""
    pages = extract_pages(path)
    if len(pages) == 1 and pages[0][0] is None:
        return pages[0][1]
    return "\n\n".join(
        f"## Page {num}\n{text}" if num is not None else text for num, text in pages
    )
