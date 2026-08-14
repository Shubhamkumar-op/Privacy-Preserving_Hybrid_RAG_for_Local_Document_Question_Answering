import pickle
from pathlib import Path

import faiss

from src.ingestion.chunker import Chunk


def save_faiss(index, chunks: list[Chunk], index_path: str | Path, metadata_path: str | Path):
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as file:
        pickle.dump(chunks, file)


def load_faiss(index_path: str | Path, metadata_path: str | Path):
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as file:
        chunks = pickle.load(file)
    return index, chunks
