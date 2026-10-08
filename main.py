# def main():
#     print("Multi-lingual RAG Project")

# if __name__ == "__main__":
#     main()


from src.ingestion.loader import load_document
from src.ingestion.text_cleaner import clean_document


def main():
    file_path = "data/raw/leave_policy.txt"

    # Load document
    documents = load_document(file_path)

    for document in documents:
        print(document.text)

    # Clean document
    cleaned_documents = [
        clean_document(document)
        for document in documents
    ]

    print("\n\n CLEANED DOCUMENT \n")
    
    for document in cleaned_documents:
        print(document.text)

        print("\nMetadata:")
        print("Source:", document.source)
        print("File type:", document.file_type)
        print("Page:", document.page_number)
        print("Language:", document.language)


if __name__ == "__main__":
    main()