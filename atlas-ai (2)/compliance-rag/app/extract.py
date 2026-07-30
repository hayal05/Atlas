"""
Extract plain text from the document formats admins are allowed to upload.
Markdown/text pass through unchanged; PDF and DOCX are converted to text
so the same chunking logic in rag.py can treat everything uniformly.
"""
import os

ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".md", ".txt"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Keep page breaks as headings so citations can point to a page.
            pages.append(f"## Page {i + 1}\n{text}")
        return "\n\n".join(pages)

    if ext == ".docx":
        import docx

        doc = docx.Document(path)
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style else ""
            if style.startswith("heading") or style == "title":
                lines.append(f"## {text}")
            else:
                lines.append(text)
        return "\n\n".join(lines)

    raise ValueError(f"Unsupported file type: {ext}")
