
from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore


class SemanticRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: MultilingualEmbedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k,
                self.vector_store.count(),
            ),
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        for i, document in enumerate(results["documents"][0]):
            retrieved_chunks.append(
                {
                    "text": document,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved_chunks