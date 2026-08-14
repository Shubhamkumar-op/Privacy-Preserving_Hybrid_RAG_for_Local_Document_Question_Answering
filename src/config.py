from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
CHUNK_SIZE = 500
TOP_K = 5

for directory in (DOCUMENTS_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)
