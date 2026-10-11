
from src.ingestion.loader import load_document
from src.ingestion.text_cleaner import clean_document
from src.chunking.text_chunker import chunk_document
from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.reranking.reranker import Reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_generator import LLMGenerator


def main():
    file_path = "data/raw/leave_policy.txt"

    # 1. Load and clean documents
    documents = load_document(file_path)

    cleaned_documents = [
        clean_document(document)
        for document in documents
    ]

    # 2. Split documents into chunks
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

    # 5. Process the query
    raw_query = (
        "   How many annual leave days do employees receive?   "
    )
    query = process_query(raw_query)

    print(f"\nProcessed query: {query}")

    # 6. Retrieve candidates using hybrid retrieval
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    candidates = retriever.search(
        query=query,
        top_k=5,
    )

    # 7. Rerank the retrieved candidates
    reranker = Reranker()

    final_results = reranker.rerank(
        query=query,
        results=candidates,
        top_k=3,
    )

    # 8. Build the RAG prompt
    prompt = build_prompt(
        query=query,
        retrieved_chunks=final_results,
        answer_language="same as the question",
    )

    # 9. Generate the answer using Groq
    generator = LLMGenerator()
    answer = generator.generate(prompt)

    print("\nGenerated answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
