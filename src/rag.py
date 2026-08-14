from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdf

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss import FAISSRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

from src.generation.ollama import LocalGenerator


class RAGPipeline:

    def __init__(self):
        self.embedder = Embedder()
        self.generator = LocalGenerator()
        self.reranker = Reranker()

        self.retriever = None
        self.chunks = []

    def ingest_many(self, pdf_paths):

        all_chunks = []

        for pdf_path in pdf_paths:
            pdf_name = pdf_path.rsplit("\\", 1)[-1]

            pages = load_pdf(pdf_path)

            chunks = chunk_pages(
                pages,
                pdf_name
            )

            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError(
                "No extractable text found in the supplied PDFs"
            )

        self.chunks = all_chunks

        faiss_retriever = FAISSRetriever(
            self.embedder
        )

        faiss_retriever.build(
            all_chunks
        )

        bm25_retriever = BM25Retriever(
            all_chunks
        )

        self.retriever = HybridRetriever(
            bm25_retriever,
            faiss_retriever
        )

    def ask(
        self,
        question,
        candidate_k=20,
        top_k=5
    ):

        if self.retriever is None:
            raise RuntimeError(
                "Ingest at least one PDF first"
            )

        # Hybrid retrieval
        candidates = self.retriever.search(
            question,
            top_k=candidate_k
        )

        # Extract (index, text) format for reranker
        reranker_candidates = []

        for i, (chunk, score) in enumerate(candidates):
            reranker_candidates.append(
                (
                    i,
                    chunk.text
                )
            )

        # Cross-encoder reranking
        reranked = self.reranker.rerank(
            question,
            reranker_candidates,
            top_k=top_k
        )

        # Convert back to chunks
        contexts = []

        for (candidate, rerank_score) in reranked:

            candidate_index = candidate[0]

            chunk, hybrid_score = candidates[
                candidate_index
            ]

            contexts.append(
                (
                    chunk,
                    float(rerank_score)
                )
            )

        # Generate answer
        answer = self.generator.generate(
            question,
            contexts
        )

        return answer, contexts