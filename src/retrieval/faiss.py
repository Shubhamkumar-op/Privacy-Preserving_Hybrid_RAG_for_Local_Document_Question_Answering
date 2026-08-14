import faiss
import numpy as np


class FAISSRetriever:
    def __init__(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]