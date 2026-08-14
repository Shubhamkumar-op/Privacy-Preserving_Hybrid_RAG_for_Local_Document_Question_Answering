from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss import FAISSRetriever


class HybridRetriever:
    """Combine lexical BM25 and semantic FAISS rankings using reciprocal rank fusion."""

    def __init__(self, bm25: BM25Retriever, faiss_retriever: FAISSRetriever):
        self.bm25 = bm25
        self.faiss = faiss_retriever

    def search(self, query: str, top_k: int = 5, candidate_k: int = 10):
        bm25_results = self.bm25.search(query, candidate_k)
        faiss_results = self.faiss.search(query, candidate_k)
        scores = {}
        chunks = {}

        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            key = (chunk.pdf_name, chunk.page_number, chunk.text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            chunks[key] = chunk

        for rank, (chunk, _) in enumerate(faiss_results, start=1):
            key = (chunk.pdf_name, chunk.page_number, chunk.text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            chunks[key] = chunk

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(chunks[key], score) for key, score in ranked]
