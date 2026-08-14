import time
import numpy as np

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss import FAISSRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker


GROUND_TRUTH = {
    "What are Large Language Models?": [0, 1, 2],
    "What factors have contributed to the development of Large Language Models?": [7],
    "What are some applications of Large Language Models?": [141, 142],
    "What are multimodal Large Language Models?": [8, 9],
    "What are some challenges associated with Large Language Models?": [
        151, 152, 154, 155, 156, 157
    ],
}


def get_index(result):
    if isinstance(result, (int, np.integer)):
        return int(result)

    if isinstance(result, np.ndarray):
        if result.ndim == 0:
            return int(result.item())

        if result.size == 1:
            return int(result.flatten()[0])

        return int(result.flatten()[0])

    if isinstance(result, dict):
        for key in ("index", "id", "chunk_index"):
            if key in result:
                return int(result[key])

    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            raise TypeError("Empty result")

        for item in result:
            if isinstance(item, (int, np.integer)):
                return int(item)

            if isinstance(item, np.ndarray):
                try:
                    return get_index(item)
                except TypeError:
                    pass

            if isinstance(item, dict):
                try:
                    return get_index(item)
                except TypeError:
                    pass

    raise TypeError(
        f"Unsupported result format: {type(result)} -> {result}"
    )


def normalize_indices(results):
    if results is None:
        return []

    if isinstance(results, tuple):
        if len(results) == 2:
            first, second = results

            if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
                return [int(x) for x in np.asarray(second).flatten()]

        results = list(results)

    if isinstance(results, np.ndarray):
        return [int(x) for x in results.flatten()]

    return [get_index(result) for result in results]


def precision_at_k(retrieved, ground_truth, k=5):
    retrieved = retrieved[:k]

    if k == 0:
        return 0.0

    relevant = sum(
        1 for x in retrieved
        if x in ground_truth
    )

    return relevant / k


def recall_at_k(retrieved, ground_truth, k=5):
    retrieved = retrieved[:k]

    if not ground_truth:
        return 0.0

    relevant = sum(
        1 for x in retrieved
        if x in ground_truth
    )

    return relevant / len(ground_truth)


def reciprocal_rank(retrieved, ground_truth):
    for rank, index in enumerate(retrieved, start=1):
        if index in ground_truth:
            return 1.0 / rank

    return 0.0


def average_results(results):
    if not results:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "time": 0.0
        }

    return {
        "precision": np.mean(
            [x["precision"] for x in results]
        ),
        "recall": np.mean(
            [x["recall"] for x in results]
        ),
        "mrr": np.mean(
            [x["rr"] for x in results]
        ),
        "time": np.mean(
            [x["time"] for x in results]
        )
    }


def evaluate_hybrid_weights(
    questions,
    ground_truth,
    bm25,
    faiss,
    embedder
):
    weights = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2)
    ]

    results = []

    print()
    print("=" * 70)
    print("HYBRID WEIGHT EXPERIMENT")
    print("=" * 70)

    for bm25_weight, faiss_weight in weights:

        hybrid = HybridRetriever(
            bm25,
            faiss,
            embedder,
            bm25_weight=bm25_weight,
            faiss_weight=faiss_weight
        )

        metrics = []

        for question in questions:

            start = time.perf_counter()

            raw_results = hybrid.search(
                question,
                top_k=5,
                candidate_k=20
            )

            elapsed = time.perf_counter() - start

            indices = normalize_indices(raw_results)

            gt = ground_truth[question]

            metrics.append({
                "precision": precision_at_k(
                    indices,
                    gt,
                    5
                ),
                "recall": recall_at_k(
                    indices,
                    gt,
                    5
                ),
                "rr": reciprocal_rank(
                    indices,
                    gt
                ),
                "time": elapsed
            })

        avg = average_results(metrics)

        results.append({
            "bm25": bm25_weight,
            "faiss": faiss_weight,
            **avg
        })

        print(
            f"BM25={bm25_weight:.1f} "
            f"FAISS={faiss_weight:.1f} | "
            f"Precision={avg['precision']:.3f} "
            f"Recall={avg['recall']:.3f} "
            f"MRR={avg['mrr']:.3f} "
            f"Time={avg['time']:.3f}s"
        )

    best = max(
        results,
        key=lambda x: (
            x["mrr"],
            x["recall"],
            x["precision"]
        )
    )

    print()
    print("=" * 70)
    print("BEST HYBRID CONFIGURATION")
    print("=" * 70)

    print(f"BM25 Weight : {best['bm25']:.1f}")
    print(f"FAISS Weight: {best['faiss']:.1f}")
    print(f"Precision@5 : {best['precision']:.3f}")
    print(f"Recall@5    : {best['recall']:.3f}")
    print(f"MRR         : {best['mrr']:.3f}")
    print(f"Latency     : {best['time']:.3f}s")

    return best


def build_reranker_candidates(
    hybrid,
    question,
    chunks,
    candidate_k=20
):
    raw_results = hybrid.search(
        question,
        top_k=candidate_k,
        candidate_k=candidate_k
    )

    indices = normalize_indices(raw_results)

    candidates = []

    seen = set()

    for index in indices:

        if index in seen:
            continue

        if index < 0 or index >= len(chunks):
            continue

        seen.add(index)

        candidates.append(
            (
                int(index),
                chunks[index]
            )
        )

    return candidates


def evaluate_reranker(
    questions,
    ground_truth,
    chunks,
    bm25,
    faiss,
    embedder,
    bm25_weight,
    faiss_weight,
    reranker,
    top_k=5,
    candidate_k=20
):
    results = []

    hybrid = HybridRetriever(
        bm25,
        faiss,
        embedder,
        bm25_weight=bm25_weight,
        faiss_weight=faiss_weight
    )

    print()
    print("=" * 70)
    print("HYBRID + CROSS-ENCODER RERANKER")
    print("=" * 70)

    for question in questions:

        start = time.perf_counter()

        candidates = build_reranker_candidates(
            hybrid,
            question,
            chunks,
            candidate_k
        )

        reranked = reranker.rerank(
            question,
            candidates,
            top_k=top_k
        )

        elapsed = time.perf_counter() - start

        indices = []

        for item, score in reranked:
            if isinstance(item, tuple):
                indices.append(int(item[0]))
            elif isinstance(item, dict):
                indices.append(int(item["index"]))
            else:
                indices.append(int(item))

        gt = ground_truth[question]

        precision = precision_at_k(indices, gt, top_k)
        recall = recall_at_k(indices, gt, top_k)
        rr = reciprocal_rank(indices, gt)

        results.append({
            "precision": precision,
            "recall": recall,
            "rr": rr,
            "time": elapsed
        })

        print(
            f"Hybrid + Reranker    "
            f"Retrieved={indices} "
            f"Precision@5={precision:.3f} "
            f"Recall@5={recall:.3f} "
            f"RR={rr:.3f} "
            f"Time={elapsed:.3f}s"
        )

    return results

def evaluate():

    text = load_pdf("LLM.pdf")

    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    embedder = Embedder()

    embeddings = embedder.encode(
        chunks
    )

    bm25 = BM25Retriever(
        chunks
    )

    faiss = FAISSRetriever(
        embeddings
    )

    reranker = Reranker()

    questions = list(
        GROUND_TRUTH.keys()
    )

    print()
    print("=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    bm25_results = []
    faiss_results = []
    hybrid_results = []

    for question in questions:

        print()
        print("=" * 70)
        print(f"QUESTION: {question}")
        print(
            f"GROUND TRUTH: "
            f"{GROUND_TRUTH[question]}"
        )

        print()
        print("Retrieved Results:")

        start = time.perf_counter()

        bm25_raw = bm25.search(
            question,
            5
        )

        bm25_time = time.perf_counter() - start

        bm25_indices = normalize_indices(
            bm25_raw
        )

        gt = GROUND_TRUTH[question]

        bm25_metric = {
            "precision": precision_at_k(
                bm25_indices,
                gt,
                5
            ),
            "recall": recall_at_k(
                bm25_indices,
                gt,
                5
            ),
            "rr": reciprocal_rank(
                bm25_indices,
                gt
            ),
            "time": bm25_time
        }

        bm25_results.append(
            bm25_metric
        )

        print(
            f"BM25                 "
            f"Retrieved={bm25_indices} "
            f"Precision@5="
            f"{bm25_metric['precision']:.3f} "
            f"Recall@5="
            f"{bm25_metric['recall']:.3f} "
            f"RR="
            f"{bm25_metric['rr']:.3f} "
            f"Time={bm25_time:.3f}s"
        )

        start = time.perf_counter()

        query_embedding = embedder.encode(
            [question]
        )

        faiss_scores, faiss_indices = faiss.search(
            query_embedding,
            5
        )

        faiss_time = time.perf_counter() - start

        faiss_indices = [
            int(x)
            for x in np.asarray(
                faiss_indices
            ).flatten()
        ]

        faiss_metric = {
            "precision": precision_at_k(
                faiss_indices,
                gt,
                5
            ),
            "recall": recall_at_k(
                faiss_indices,
                gt,
                5
            ),
            "rr": reciprocal_rank(
                faiss_indices,
                gt
            ),
            "time": faiss_time
        }

        faiss_results.append(
            faiss_metric
        )

        print(
            f"FAISS                "
            f"Retrieved={faiss_indices} "
            f"Precision@5="
            f"{faiss_metric['precision']:.3f} "
            f"Recall@5="
            f"{faiss_metric['recall']:.3f} "
            f"RR="
            f"{faiss_metric['rr']:.3f} "
            f"Time={faiss_time:.3f}s"
        )

        hybrid = HybridRetriever(
            bm25,
            faiss,
            embedder,
            bm25_weight=0.5,
            faiss_weight=0.5
        )

        start = time.perf_counter()

        hybrid_raw = hybrid.search(
            question,
            top_k=5,
            candidate_k=20
        )

        hybrid_time = time.perf_counter() - start

        hybrid_indices = normalize_indices(
            hybrid_raw
        )

        hybrid_metric = {
            "precision": precision_at_k(
                hybrid_indices,
                gt,
                5
            ),
            "recall": recall_at_k(
                hybrid_indices,
                gt,
                5
            ),
            "rr": reciprocal_rank(
                hybrid_indices,
                gt
            ),
            "time": hybrid_time
        }

        hybrid_results.append(
            hybrid_metric
        )

        print(
            f"Hybrid               "
            f"Retrieved={hybrid_indices} "
            f"Precision@5="
            f"{hybrid_metric['precision']:.3f} "
            f"Recall@5="
            f"{hybrid_metric['recall']:.3f} "
            f"RR="
            f"{hybrid_metric['rr']:.3f} "
            f"Time={hybrid_time:.3f}s"
        )

    print()
    print("=" * 70)
    print("WEIGHT OPTIMIZATION")
    print("=" * 70)

    best = evaluate_hybrid_weights(
        questions,
        GROUND_TRUTH,
        bm25,
        faiss,
        embedder
    )

    reranker_results = evaluate_reranker(
        questions,
        GROUND_TRUTH,
        chunks,
        bm25,
        faiss,
        embedder,
        best["bm25"],
        best["faiss"],
        reranker,
        top_k=5,
        candidate_k=20
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    summaries = {
        "BM25": average_results(
            bm25_results
        ),
        "FAISS": average_results(
            faiss_results
        ),
        "Hybrid 0.5/0.5": average_results(
            hybrid_results
        ),
        "Hybrid + Reranker": average_results(
            reranker_results
        )
    }

    for name, metrics in summaries.items():

        print(
            f"{name:<22}"
            f"Precision@5="
            f"{metrics['precision']:.3f} "
            f"Recall@5="
            f"{metrics['recall']:.3f} "
            f"MRR="
            f"{metrics['mrr']:.3f} "
            f"AvgTime="
            f"{metrics['time']:.3f}s"
        )


if __name__ == "__main__":
    evaluate()