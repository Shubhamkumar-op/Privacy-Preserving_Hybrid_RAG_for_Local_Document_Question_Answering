from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.corpus = [chunk.text.lower().split() for chunk in chunks]
        self.index = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            return []
        scores = self.index.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(score)) for i, score in ranked]
