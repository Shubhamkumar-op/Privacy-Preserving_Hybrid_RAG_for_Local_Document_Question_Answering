from src.ingestion.chunker import chunk_pages


def test_chunking_preserves_page_and_pdf_metadata():
    chunks = chunk_pages([(2, "abcdefghij")], "paper.pdf", chunk_size=6, overlap=2)
    assert chunks
    assert all(chunk.pdf_name == "paper.pdf" for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)
