import pymupdf


def load_pdf(pdf_path: str) -> str:
    document = pymupdf.open(pdf_path)

    try:
        pages = [page.get_text() for page in document]
        return "\n".join(pages)
    finally:
        document.close()