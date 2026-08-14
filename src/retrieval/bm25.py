from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_documents = [
            document.lower().split()
            for document in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def search(self, query, top_k=5):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            (self.documents[i], scores[i], i)
            for i in ranked_indices
        ]