from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):

        texts = []

        for document in documents:

            # (index, text)
            if isinstance(document, tuple):
                text = document[1]

            # {"index": ..., "text": ...}
            elif isinstance(document, dict):
                text = (
                    document.get("text")
                    or document.get("chunk")
                    or document.get("content")
                )

                if text is None:
                    raise ValueError(
                        f"Cannot find text in document: {document}"
                    )

            # plain string
            else:
                text = document

            texts.append(str(text))

        # CrossEncoder 5.7 expects list pairs
        pairs = [
            [query, text]
            for text in texts
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return ranked[:top_k]