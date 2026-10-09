
from src.ingestion.loader import load_document
from src.ingestion.text_cleaner import clean_document
from src.chunking.text_chunker import chunk_document
from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import Reranker


def main():
    file_path = "data/raw/leave_policy.txt"

    # 1. Load and clean documents
    documents = load_document(file_path)
    cleaned_documents = [
        clean_document(document)
        for document in documents
    ]

    # 2. Chunk documents
    chunks = []

    for document in cleaned_documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=100,
                overlap=20,
            )
        )

    print(f"Total chunks: {len(chunks)}")

    # 3. Generate embeddings
    embedder = MultilingualEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    # 4. Store chunks and vectors
    vector_store = VectorStore()
    vector_store.add_chunks(chunks, embeddings)

    print(f"Chunks in database: {vector_store.count()}")

    # 5. Hybrid retrieval
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    query = "How many annual leave days do employees receive?"
    candidates = retriever.search(query, top_k=5)

    print("\nHybrid retrieval results:")

    for rank, result in enumerate(candidates, start=1):
        print(f"\nResult {rank}: {result['text']}")
        print(f"RRF score: {result['rrf_score']:.6f}")

    # 6. Rerank the retrieved candidates
    reranker = Reranker()
    final_results = reranker.rerank(
        query=query,
        results=candidates,
        top_k=3,
    )

    print("\nReranked results:")

    for rank, result in enumerate(final_results, start=1):
        print(f"\nResult {rank}: {result['text']}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Reranker score: {result['rerank_score']:.4f}")


if __name__ == "__main__":
    main()