import fitz
from pathlib import Path


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract text page-by-page and preserve page numbers."""
    pages = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_number, text))
    return pages
