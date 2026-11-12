
import json
import math
from pathlib import Path

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    retrieved = retrieved_ids[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant_count = min(len(relevant_ids), k)

    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg


def main():
    # 1. Load evaluation questions.
    evaluation_path = Path("data/evaluation/rag_eval.json")

    with evaluation_path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError("Evaluation dataset is empty.")

    # 2. Connect to the existing vector database.
    vector_store = VectorStore()
    embedder = MultilingualEmbedder()

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    if vector_store.count() == 0:
        raise RuntimeError(
            "The vector database is empty. Run main.py first "
            "to ingest and index the documents."
        )

    # 3. Configure evaluation.
    top_k = 3

    metric_totals = {
        "hit_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "source_hit_rate": 0.0,
    }

    question_results = []

    print(f"Total evaluation questions: {len(test_cases)}")
    print("Evaluation metrics: Hit Rate, Precision, Recall, MRR, NDCG")
    print(f"K = {top_k}\n")

    # 4. Evaluate every question.
    for test_case in test_cases:
        query = process_query(test_case["question"])

        relevant_ids = set(
            test_case["relevant_chunk_ids"]
        )

        expected_source = test_case.get("expected_source")

        results = retriever.search(
            query=query,
            top_k=top_k,
        )

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        retrieved_sources = [
            result["metadata"].get("source")
            for result in results
        ]

        # Calculate metrics for this question.
        hit = int(
            any(chunk_id in relevant_ids for chunk_id in retrieved_ids)
        )

        precision = precision_at_k(
            retrieved_ids, relevant_ids, top_k
        )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, top_k
        )

        mrr = reciprocal_rank_at_k(
            retrieved_ids, relevant_ids, top_k
        )

        ndcg = ndcg_at_k(
            retrieved_ids, relevant_ids, top_k
        )

        source_hit = int(
            expected_source in retrieved_sources
        ) if expected_source else 0

        # Store individual scores.
        question_metrics = {
            "hit_rate": hit,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "source_hit_rate": source_hit,
        }

        # Add individual scores to overall totals.
        for metric_name, score in question_metrics.items():
            metric_totals[metric_name] += score

        question_results.append({
            "question_id": test_case["id"],
            "question": test_case["question"],
            "answer_language": test_case.get("answer_language"),
            "relevant_chunk_ids": sorted(relevant_ids),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_sources": retrieved_sources,
            "metrics": question_metrics,
        })

        print(f"Question {test_case['id']}")
        print(f"Question: {query}")
        print(f"Relevant chunk IDs: {sorted(relevant_ids)}")
        print(f"Retrieved chunk IDs: {retrieved_ids}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Hit Rate: {hit:.2f}")
        print(f"Precision@{top_k}: {precision:.3f}")
        print(f"Recall@{top_k}: {recall:.3f}")
        print(f"MRR@{top_k}: {mrr:.3f}")
        print(f"NDCG@{top_k}: {ndcg:.3f}")
        print("-" * 60)

    # 5. Calculate average metrics.
    total_questions = len(test_cases)

    average_metrics = {
        metric_name: total / total_questions
        for metric_name, total in metric_totals.items()
    }

    print("\nRetrieval Evaluation Summary")
    print(f"Questions tested: {total_questions}")

    for metric_name, average in average_metrics.items():
        display_name = metric_name.replace("_", " ").title()
        print(f"{display_name}: {average:.3f} ({average:.2%})")

    # 6. Build the evaluation report.
    report = {
        "evaluation_type": "Hybrid Retrieval Evaluation",
        "top_k": top_k,
        "total_questions": total_questions,
        "individual_results": question_results,
        "average_metrics": average_metrics,
    }

    # 7. Save the report.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "retrieval_evaluation.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
