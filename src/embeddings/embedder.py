from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
