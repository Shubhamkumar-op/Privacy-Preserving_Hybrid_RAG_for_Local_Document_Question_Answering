from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):

        # ---------------------------------------------------------
        # Extract plain text from retrieved candidates
        # ---------------------------------------------------------
        texts = []

        for document in documents:

            # Candidate format: (index, text)
            if isinstance(document, tuple):
                text = document[1]

            # Candidate format: dictionary
            elif isinstance(document, dict):

                text = (
                    document.get("text")
                    or document.get("chunk")
                    or document.get("content")
                )

                if text is None:
                    raise ValueError(
                        f"Cannot find text in candidate: {document}"
                    )

            # Candidate already contains plain text
            else:
                text = document

            texts.append(str(text))

        # ---------------------------------------------------------
        # CrossEncoder expects:
        # [
        #   [query, document_text],
        #   [query, document_text],
        #   ...
        # ]
        # ---------------------------------------------------------
        pairs = [
            [query, text]
            for text in texts
        ]

        # ---------------------------------------------------------
        # Get CrossEncoder scores
        # ---------------------------------------------------------
        scores = self.model.predict(pairs)

        # ---------------------------------------------------------
        # Keep original candidates together with scores
        # ---------------------------------------------------------
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return ranked[:top_k]