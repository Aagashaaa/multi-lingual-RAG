
import json
from pathlib import Path

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query


def main():
    # 1. Load evaluation questions
    evaluation_path = Path("data/evaluation/rag_eval.json")

    with evaluation_path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # 2. Connect to the existing vector database
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

    # 3. Evaluate retrieval for every question
    top_k = 3
    hits = 0

    print(f"Total evaluation questions: {len(test_cases)}")
    print(f"Evaluation metric: Hit Rate@{top_k}\n")

    for test_case in test_cases:
        query = process_query(test_case["question"])
        expected_source = test_case["expected_source"]

        results = retriever.search(
            query=query,
            top_k=top_k,
        )

        retrieved_sources = [
            result["metadata"].get("source")
            for result in results
        ]

        hit = expected_source in retrieved_sources

        if hit:
            hits += 1

        status = "PASS" if hit else "FAIL"

        print(f"Question {test_case['id']}: {status}")
        print(f"Question: {test_case['question']}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {retrieved_sources}")
        print("-" * 50)

    # 4. Calculate the overall retrieval score
    hit_rate = hits / len(test_cases) if test_cases else 0.0

    print("\nRetrieval Evaluation Summary")
    print(f"Questions tested: {len(test_cases)}")
    print(f"Successful retrievals: {hits}")
    print(f"Hit Rate@{top_k}: {hit_rate:.2%}")


if __name__ == "__main__":
    main()
