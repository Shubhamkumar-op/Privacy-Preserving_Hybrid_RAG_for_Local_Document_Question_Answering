from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_text
from src.ingestion.pdf_loader import load_pdf

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss import FAISSRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

from src.generation.ollama import OllamaGenerator


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.generator = OllamaGenerator()
        self.reranker = Reranker()

        self.retriever = None
        self.chunks = []

    def ingest_many(self, pdf_paths):
        all_chunks = []

        for pdf_path in pdf_paths:
            text = load_pdf(pdf_path)
            all_chunks.extend(chunk_text(text))

        if not all_chunks:
            raise ValueError("No extractable text found in the supplied PDFs")

        self.chunks = all_chunks

        embeddings = self.embedder.encode(all_chunks)
        faiss_retriever = FAISSRetriever(embeddings)
        bm25_retriever = BM25Retriever(all_chunks)

        self.retriever = HybridRetriever(
            bm25_retriever,
            faiss_retriever,
            self.embedder
        )

    def ask(self, question, candidate_k=20, top_k=5):
        if self.retriever is None:
            raise RuntimeError("Ingest at least one PDF first")

        candidates = self.retriever.search(
            question,
            top_k=candidate_k,
            candidate_k=candidate_k
        )

        reranker_candidates = [
            (index, self.chunks[index])
            for index, _score in candidates
        ]

        reranked = self.reranker.rerank(
            question,
            reranker_candidates,
            top_k=top_k
        )

        contexts = [
            (candidate[1], float(score))
            for candidate, score in reranked
        ]

        context_text = "\n\n---\n\n".join(
            text for text, _score in contexts
        )

        prompt = f"""Answer the question using only the provided context.
If the answer is not present in the context, say that the information is not available in the documents.

Context:
{context_text}

Question:
{question}

Answer:"""

        answer = self.generator.generate(prompt)

        return answer, contexts
