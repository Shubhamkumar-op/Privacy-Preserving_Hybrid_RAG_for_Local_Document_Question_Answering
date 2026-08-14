import math


def precision_at_k(retrieved, relevant, k):
    retrieved = retrieved[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for item in retrieved
        if item in relevant
    )

    return relevant_count / len(retrieved)


def recall_at_k(retrieved, relevant, k):
    retrieved = retrieved[:k]

    if not relevant:
        return 0.0

    relevant_count = sum(
        1 for item in retrieved
        if item in relevant
    )

    return relevant_count / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def mrr(results):
    if not results:
        return 0.0

    return sum(results) / len(results)