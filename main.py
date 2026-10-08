# def main():
#     print("Multi-lingual RAG Project")

# if __name__ == "__main__":
#     main()


from src.ingestion.loader import load_document
from src.ingestion.text_cleaner import clean_document
from src.chunking.text_chunker import chunk_document


def main():
    file_path = "data/raw/leave_policy.txt"

    # Step 1: Load document
    documents = load_document(file_path)

    print("===== LOADED DOCUMENT =====\n")

    for document in documents:
        print(document.text)

    # Step 2: Clean document
    cleaned_documents = [
        clean_document(document)
        for document in documents
    ]

    print("\n===== CLEANED DOCUMENT =====\n")

    for document in cleaned_documents:
        print(document.text)

    # Step 3: Chunk document
    chunks = []

    for document in cleaned_documents:
        document_chunks = chunk_document(
            document,
            chunk_size=100,
            overlap=20,
        )

        chunks.extend(document_chunks)

    print("\n===== CHUNKS =====\n")

    for chunk in chunks:
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Text: {chunk.text}")
        print("-" * 60)

    print(f"\nTotal chunks: {len(chunks)}")


if __name__ == "__main__":
    main()