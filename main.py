
from src.ingestion.loader import load_document
from src.ingestion.text_cleaner import clean_document
from src.chunking.text_chunker import chunk_document
from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore


def main():
    file_path = "data/raw/leave_policy.txt"

    # Step 1: Load documents
    documents = load_document(file_path)

    # Step 2: Clean documents
    cleaned_documents = [
        clean_document(document)
        for document in documents
    ]

    # Step 3: Split documents into chunks
    chunks = []

    for document in cleaned_documents:
        document_chunks = chunk_document(
            document,
            chunk_size=100,
            overlap=20,
        )
        chunks.extend(document_chunks)

    print(f"Total chunks: {len(chunks)}")

    # Step 4: Generate embeddings
    embedder = MultilingualEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    print(f"Total embeddings: {len(embeddings)}")

    # Step 5: Store embeddings in ChromaDB
    vector_store = VectorStore()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"Chunks stored in database: {vector_store.count()}")
    print("Vector database setup completed successfully.")


if __name__ == "__main__":
    main()