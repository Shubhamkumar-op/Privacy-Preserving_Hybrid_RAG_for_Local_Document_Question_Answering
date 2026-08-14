from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    pdf_name: str
    page_number: int
    text: str


def chunk_pages(pages: list[tuple[int, str]], pdf_name: str, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    """Split page text into overlapping character chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for page_number, text in pages:
        for start in range(0, len(text), step):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(Chunk(pdf_name, page_number, chunk))
    return chunks
