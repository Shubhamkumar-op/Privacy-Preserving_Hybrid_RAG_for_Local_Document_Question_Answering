from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss import FAISSRetriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.ollama import LocalGenerator


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.generator = LocalGenerator()
        self.retriever = None

    def ingest(self, pdf_path: str):
        pdf_name = pdf_path.rsplit("/", 1)[-1]
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, pdf_name)
        if not chunks:
            raise ValueError(f"No text found in {pdf_name}")
        faiss_retriever = FAISSRetriever(self.embedder)
        faiss_retriever.build(chunks)
        bm25_retriever = BM25Retriever(chunks)
        self.retriever = HybridRetriever(bm25_retriever, faiss_retriever)

    def ingest_many(self, pdf_paths: list[str]):
        all_chunks = []
        for pdf_path in pdf_paths:
            pdf_name = pdf_path.rsplit("/", 1)[-1]
            all_chunks.extend(chunk_pages(load_pdf(pdf_path), pdf_name))
        if not all_chunks:
            raise ValueError("No extractable text found in the supplied PDFs")
        faiss_retriever = FAISSRetriever(self.embedder)
        faiss_retriever.build(all_chunks)
        self.retriever = HybridRetriever(BM25Retriever(all_chunks), faiss_retriever)

    def ask(self, question: str, top_k: int = 5):
        if self.retriever is None:
            raise RuntimeError("Ingest at least one PDF before asking a question")
        contexts = self.retriever.search(question, top_k=top_k)
        answer = self.generator.generate(question, contexts)
        return answer, contexts
