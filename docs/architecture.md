# Architecture

## Data flow

1. PDFs are uploaded locally.
2. PyMuPDF extracts text while preserving page numbers.
3. Text is split into overlapping chunks.
4. SentenceTransformer creates semantic embeddings.
5. BM25 performs lexical retrieval.
6. FAISS performs semantic retrieval.
7. Reciprocal Rank Fusion combines both retrieval signals.
8. Ollama/Mistral generates an answer from retrieved evidence only.
9. Optional Argos Translate performs offline English-to-Hindi translation.

## Privacy boundary

No external LLM API is required by the application. Document processing, embeddings, retrieval, generation, and optional translation are designed to run locally.

## Research components

- `src/retrieval/bm25.py`: lexical baseline
- `src/retrieval/faiss.py`: semantic baseline
- `src/retrieval/hybrid.py`: proposed hybrid retrieval
- `evaluation/`: reproducible retrieval evaluation
- `experiments/`: notebook-based research history
