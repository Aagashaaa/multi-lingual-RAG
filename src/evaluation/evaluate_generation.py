
import json
import re
from pathlib import Path

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.reranking.reranker import Reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_generator import LLMGenerator


def calculate_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0

    answer_lower = answer.casefold()

    matched = sum(
        1
        for keyword in expected_keywords
        if keyword.casefold() in answer_lower
    )

    return matched / len(expected_keywords)


def check_language_compliance(
    answer: str,
    expected_language: str,
) -> bool:
    tamil_count = sum(
        1 for char in answer
        if "\u0B80" <= char <= "\u0BFF"
    )

    latin_count = sum(
        1 for char in answer
        if ("a" <= char.lower() <= "z")
    )

    language = expected_language.casefold()

    if language == "tamil":
        return tamil_count > 0 and tamil_count >= latin_count

    if language == "english":
        return latin_count > 0 and latin_count > tamil_count

    raise ValueError(
        f"Unsupported evaluation language: {expected_language}"
    )


def build_judge_prompt(
    question: str,
    expected_answer: str,
    generated_answer: str,
    retrieved_context: str,
    answer_language: str,
) -> str:
    return f"""
You are an impartial evaluator assessing a Retrieval-Augmented
Generation (RAG) system.

Evaluate the generated answer using the question, expected answer,
and retrieved context.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CONTEXT:
{retrieved_context}

REQUIRED ANSWER LANGUAGE:
{answer_language}

Give each metric a score from 1 to 5.

1. correctness:
Does the generated answer convey the correct information compared
with the expected answer?

2. faithfulness:
Are the factual claims in the generated answer supported by the
retrieved context? Penalize unsupported or contradictory claims.

3. relevance:
Does the generated answer directly answer the question without
unnecessary or unrelated information?

Scoring guide:
1 = Very poor
2 = Poor
3 = Partially satisfactory
4 = Good
5 = Excellent

Be strict. Do not give a high score simply because the answer
contains similar words to the expected answer.

Return ONLY valid JSON using this exact structure:

{{
    "correctness": 1,
    "faithfulness": 1,
    "relevance": 1,
    "reason": "Brief explanation of the scores"
}}
"""


def parse_judge_response(response: str) -> dict:
    match = re.search(r"\{.*\}", response, re.DOTALL)

    if not match:
        raise ValueError(
            f"Could not find JSON in judge response: {response}"
        )

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON from judge: {response}"
        ) from error

    required_metrics = [
        "correctness",
        "faithfulness",
        "relevance",
    ]

    for metric in required_metrics:
        if metric not in result:
            raise ValueError(
                f"Judge response is missing: {metric}"
            )

        score = result[metric]

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 1 <= score <= 5
        ):
            raise ValueError(
                f"Invalid {metric} score: {score}"
            )

    return result


def main():
    # 1. Load evaluation questions.
    evaluation_path = Path("data/evaluation/rag_eval.json")

    with evaluation_path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError("The evaluation dataset is empty.")

    # 2. Connect to existing RAG components.
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

    retrieval_top_k = 5
    rerank_top_k = 3

    # 3. Prepare evaluation tracking.
    question_results = []

    metric_totals = {
        "keyword_coverage": 0.0,
        "correctness": 0.0,
        "faithfulness": 0.0,
        "relevance": 0.0,
        "language_compliance": 0.0,
        "overall_pass_rate": 0.0,
    }

    print(f"Total evaluation questions: {len(test_cases)}")
    print("Evaluation metrics:")
    print("- Keyword coverage")
    print("- Correctness (LLM judge)")
    print("- Faithfulness (LLM judge)")
    print("- Relevance (LLM judge)")
    print("- Language compliance\n")

    # 4. Generate and evaluate every answer.
    for test_case in test_cases:
        query = process_query(test_case["question"])

        expected_answer = test_case["expected_answer"]
        answer_language = test_case["answer_language"]
        expected_keywords = test_case["expected_keywords"]

        # Retrieve candidate chunks.
        candidates = retriever.search(
            query=query,
            top_k=retrieval_top_k,
        )

        # Rerank retrieved chunks.
        final_results = reranker.rerank(
            query=query,
            results=candidates,
            top_k=rerank_top_k,
        )

        # Build the generation prompt.
        prompt = build_prompt(
            query=query,
            retrieved_chunks=final_results,
            answer_language=answer_language,
        )

        # Generate the answer.
        answer = generator.generate(prompt)

        # Calculate keyword coverage.
        keyword_coverage = calculate_keyword_coverage(
            answer,
            expected_keywords,
        )

        # Check requested answer language.
        language_compliance = check_language_compliance(
            answer,
            answer_language,
        )

        # Prepare retrieved context for the LLM judge.
        retrieved_context = "\n\n".join(
            result["text"]
            for result in final_results
        )

        # Ask the LLM judge to evaluate the answer.
        judge_prompt = build_judge_prompt(
            question=query,
            expected_answer=expected_answer,
            generated_answer=answer,
            retrieved_context=retrieved_context,
            answer_language=answer_language,
        )

        judge_response = generator.generate(judge_prompt)

        judge_scores = parse_judge_response(judge_response)

        correctness = judge_scores["correctness"]
        faithfulness = judge_scores["faithfulness"]
        relevance = judge_scores["relevance"]

        # An answer passes only if all three judge scores
        # are at least 4 and the requested language is used.
        success = (
            correctness >= 4
            and faithfulness >= 4
            and relevance >= 4
            and language_compliance
        )

        if success:
            metric_totals["overall_pass_rate"] += 1

        metric_totals["keyword_coverage"] += keyword_coverage
        metric_totals["correctness"] += correctness
        metric_totals["faithfulness"] += faithfulness
        metric_totals["relevance"] += relevance
        metric_totals["language_compliance"] += int(
            language_compliance
        )

        # Store individual question results.
        question_results.append({
            "question_id": test_case["id"],
            "question": test_case["question"],
            "answer_language": answer_language,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "retrieved_chunk_ids": [
                result.get("id")
                for result in final_results
            ],
            "keyword_coverage": keyword_coverage,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "language_compliance": language_compliance,
            "judge_reason": judge_scores.get("reason", ""),
            "passed": success,
        })

        # Display individual results.
        status = "PASS" if success else "FAIL"

        print(f"Question {test_case['id']}: {status}")
        print(f"Question: {test_case['question']}")
        print(f"Generated answer: {answer}")
        print(f"Keyword coverage: {keyword_coverage:.2%}")
        print(f"Correctness: {correctness}/5")
        print(f"Faithfulness: {faithfulness}/5")
        print(f"Relevance: {relevance}/5")
        print(f"Language compliance: {language_compliance}")
        print(f"Judge reason: {judge_scores.get('reason', '')}")
        print("-" * 60)

    # 5. Calculate average evaluation metrics.
    total_questions = len(test_cases)

    average_metrics = {
        metric_name: total / total_questions
        for metric_name, total in metric_totals.items()
    }

    print("\nGeneration Evaluation Summary")
    print(f"Questions tested: {total_questions}")

    for metric_name, average in average_metrics.items():
        display_name = metric_name.replace("_", " ").title()
        print(f"{display_name}: {average:.3f} ({average:.2%})")

    # 6. Build the evaluation report.
    report = {
        "evaluation_type": "RAG Generation Evaluation",
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k": rerank_top_k,
        "total_questions": total_questions,
        "individual_results": question_results,
        "average_metrics": average_metrics,
        "pass_criteria": {
            "correctness_minimum": 4,
            "faithfulness_minimum": 4,
            "relevance_minimum": 4,
            "language_compliance_required": True,
        },
    }

    # 7. Save the report.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "generation_evaluation.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nGeneration evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
