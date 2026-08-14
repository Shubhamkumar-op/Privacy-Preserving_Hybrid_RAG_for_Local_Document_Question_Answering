# Privacy-Preserving Hybrid RAG for Local Document Question Answering

A fully local Retrieval-Augmented Generation (RAG) system for question answering over private PDF documents.

## Highlights

- 🔒 Local-first architecture: documents and inference remain on the local machine.
- 🔎 Hybrid retrieval using **BM25 + FAISS**.
- 🧠 Local embeddings with `all-MiniLM-L6-v2`.
- 🤖 Local LLM inference through **Ollama**.
- 📄 PDF/page-aware evidence retrieval.
- 🌐 Optional offline English-to-Hindi translation using Argos Translate.
- 📊 Evaluation utilities for Precision@K and Recall@K.
- 🧪 Unit tests for core components.

## Architecture

```text
PDFs → Extraction → Chunking → Embeddings
                         │
                 ┌───────┴────────┐
                 ▼                ▼
               BM25             FAISS
                 └───────┬────────┘
                         ▼
                  Hybrid Retrieval
                         ▼
                   Local Ollama
                         ▼
                  Grounded Answer
```

## Project Structure

```text
src/
├── app.py
├── config.py
├── rag.py
├── ingestion/       # PDF extraction and chunking
├── embeddings/      # SentenceTransformer embeddings
├── retrieval/       # BM25, FAISS and hybrid retrieval
├── generation/      # Local Ollama generation and prompts
├── storage/         # SQLite and FAISS persistence
└── translation/     # Offline translation

evaluation/          # Retrieval evaluation
tests/               # Unit tests
```

The original notebooks and legacy implementation are retained in the repository as research history while the modular implementation is developed under `src/`.

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama

Make sure Ollama is running locally and the configured model is available:

```bash
ollama pull mistral:latest
```

### 3. Run the application

```bash
streamlit run src/app.py
```

### 4. Run tests

```bash
pytest
```

## Research Direction

The system is designed to compare lexical retrieval, semantic retrieval, and hybrid retrieval while preserving document locality. Future evaluation can include retrieval precision/recall, answer faithfulness, response latency, and semantic similarity.

## License

See [LICENSE](LICENSE).
