
import json
from pathlib import Path

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.reranking.reranker import Reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_generator import LLMGenerator


def main():
    # 1. Load evaluation questions
    evaluation_path = Path("data/evaluation/rag_eval.json")

    with evaluation_path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError("The evaluation dataset is empty.")

    # 2. Connect to existing components
    vector_store = VectorStore()

    if vector_store.count() == 0:
        raise RuntimeError(
            "The vector database is empty. Run main.py first."
        )

    embedder = MultilingualEmbedder()

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    reranker = Reranker()
    generator = LLMGenerator()

    passed = 0

    print(f"Total evaluation questions: {len(test_cases)}")
    print("Evaluation metric: Expected keyword coverage\n")

    # 3. Generate and evaluate each answer
    for test_case in test_cases:
        query = process_query(test_case["question"])
        answer_language = test_case["answer_language"]
        expected_keywords = test_case["expected_keywords"]

        candidates = retriever.search(
            query=query,
            top_k=5,
        )

        final_results = reranker.rerank(
            query=query,
            results=candidates,
            top_k=3,
        )

        prompt = build_prompt(
            query=query,
            retrieved_chunks=final_results,
            answer_language=answer_language,
        )

        answer = generator.generate(prompt)

        answer_lower = answer.casefold()

        missing_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.casefold() not in answer_lower
        ]

        success = len(missing_keywords) == 0

        if success:
            passed += 1

        status = "PASS" if success else "FAIL"

        print(f"Question {test_case['id']}: {status}")
        print(f"Question: {test_case['question']}")
        print(f"Generated answer: {answer}")
        print(f"Expected keywords: {expected_keywords}")
        print(f"Missing keywords: {missing_keywords}")
        print("-" * 60)

    # 4. Summarize results
    score = passed / len(test_cases)

    print("\nGeneration Evaluation Summary")
    print(f"Questions tested: {len(test_cases)}")
    print(f"Answers passing keyword checks: {passed}")
    print(f"Keyword pass rate: {score:.2%}")


if __name__ == "__main__":
    main()
