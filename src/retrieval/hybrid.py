import numpy as np


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever,
        faiss_retriever,
        embedder,
        bm25_weight=0.5,
        faiss_weight=0.5
    ):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.embedder = embedder

        total = bm25_weight + faiss_weight

        if total <= 0:
            raise ValueError("Weights must be greater than zero.")

        self.bm25_weight = bm25_weight / total
        self.faiss_weight = faiss_weight / total

    def _normalize_scores(self, scores):
        scores = np.asarray(scores, dtype=np.float32)

        if len(scores) == 0:
            return scores

        minimum = np.min(scores)
        maximum = np.max(scores)

        if maximum == minimum:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)

    def _normalize_embedding(self, embedding):
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True
        )

        norm[norm == 0] = 1.0

        return embedding / norm

    def search(self, query, top_k=5, candidate_k=20):

        bm25_results = self.bm25.search(
            query,
            candidate_k
        )

        query_embedding = self.embedder.encode(
            [query]
        )

        query_embedding = self._normalize_embedding(
            query_embedding
        )

        faiss_scores, faiss_indices = self.faiss.search(
            query_embedding,
            candidate_k
        )

        bm25_indices = []
        bm25_scores = []

        for document, score, index in bm25_results:
            bm25_indices.append(int(index))
            bm25_scores.append(float(score))

        bm25_scores = self._normalize_scores(
            bm25_scores
        )

        faiss_indices = np.asarray(
            faiss_indices,
            dtype=np.int64
        )

        faiss_scores = self._normalize_scores(
            faiss_scores
        )

        combined_scores = {}

        for index, score in zip(
            bm25_indices,
            bm25_scores
        ):
            combined_scores[index] = (
                self.bm25_weight * float(score)
            )

        for index, score in zip(
            faiss_indices,
            faiss_scores
        ):
            index = int(index)

            combined_scores[index] = (
                combined_scores.get(index, 0.0)
                + self.faiss_weight * float(score)
            )

        ranked_indices = sorted(
            combined_scores.keys(),
            key=lambda index: combined_scores[index],
            reverse=True
        )

        ranked_indices = ranked_indices[:top_k]

        return [
            (
                index,
                combined_scores[index]
            )
            for index in ranked_indices
        ]