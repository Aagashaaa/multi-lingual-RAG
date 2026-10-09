
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_result = result.copy()
            reranked_result["rerank_score"] = float(score)
            reranked_results.append(reranked_result)

        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked_results[:top_k]