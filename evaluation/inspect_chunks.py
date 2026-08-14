from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text


PDF_PATH = "LLM.pdf"

TARGET_CHUNKS = [
    0, 1, 2,
    7,
    8, 9,
    141, 142,
    151, 152, 154, 155, 156, 157
]


def main():
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    print("=" * 80)

    for index in TARGET_CHUNKS:
        if index >= len(chunks):
            print(f"\nChunk {index}: NOT FOUND")
            continue

        print(f"\n{'=' * 30} CHUNK {index} {'=' * 30}")
        print(chunks[index])


if __name__ == "__main__":
    main()