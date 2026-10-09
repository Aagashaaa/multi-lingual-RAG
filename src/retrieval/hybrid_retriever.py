
import re

from rank_bm25 import BM25Okapi

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: MultilingualEmbedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def search(
        self,
        query: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        # Load stored documents and their metadata.
        stored = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        ids = stored["ids"]
        documents = stored["documents"] or []
        metadatas = stored["metadatas"] or []

        if not ids:
            return []

        corpus = [
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
            }
            for i in range(len(ids))
        ]

        corpus_by_id = {
            item["id"]: item for item in corpus
        }

        # 1. Semantic retrieval using ChromaDB.
        query_embedding = self.embedder.embed_text(query)
        candidate_k = min(top_k * 3, len(corpus))

        semantic_results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        semantic_ids = semantic_results["ids"][0]
        semantic_distances = semantic_results["distances"][0]

        # 2. Keyword retrieval using BM25.
        tokenized_corpus = [
            self.tokenize(item["text"])
            for item in corpus
        ]

        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = self.tokenize(query)
        keyword_scores = bm25.get_scores(query_tokens)

        keyword_ranked_indices = sorted(
            range(len(corpus)),
            key=lambda i: keyword_scores[i],
            reverse=True,
        )

        # 3. Combine rankings using Reciprocal Rank Fusion.
        fused_scores = {}
        semantic_distance_by_id = {}

        for rank, chunk_id in enumerate(semantic_ids, start=1):
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )
            semantic_distance_by_id[chunk_id] = (
                semantic_distances[rank - 1]
            )

        for rank, index in enumerate(
            keyword_ranked_indices, start=1
        ):
            if keyword_scores[index] <= 0:
                continue

            chunk_id = ids[index]
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        # 4. Return the highest-scoring combined results.
        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        results = []

        for chunk_id in ranked_ids[:top_k]:
            item = corpus_by_id[chunk_id]

            results.append(
                {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": fused_scores[chunk_id],
                    "semantic_distance": (
                        semantic_distance_by_id.get(chunk_id)
                    ),
                }
            )

        return results