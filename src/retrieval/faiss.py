import faiss
import numpy as np

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunk


class FAISSRetriever:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        embeddings = self.embedder.encode([c.text for c in chunks]).astype("float32")
        if not len(embeddings):
            raise ValueError("Cannot build FAISS index from empty chunks")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            return []
        vector = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(vector, min(top_k, len(self.chunks)))
        return [(self.chunks[i], float(d)) for d, i in zip(distances[0], indices[0]) if i >= 0]
