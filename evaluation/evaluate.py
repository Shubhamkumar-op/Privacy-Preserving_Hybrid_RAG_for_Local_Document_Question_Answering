"""Entry point for reproducible RAG evaluation.

Add benchmark questions and gold evidence IDs here as the research dataset is finalized.
"""

from evaluation.metrics import precision_at_k, recall_at_k


def evaluate_case(retrieved_ids, relevant_ids, k=5):
    return {
        "precision@k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall@k": recall_at_k(retrieved_ids, relevant_ids, k),
    }


if __name__ == "__main__":
    print(evaluate_case([1, 2, 3, 4, 5], [2, 5], k=5))
