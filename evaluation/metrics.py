def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved & relevant) / k if k else 0.0
