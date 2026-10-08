import re
import unicodedata

from src.ingestion.document import Document


def clean_text(text: str) -> str:
    """
    Clean and normalize document text.
    """

    if not text:
        return ""

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove spaces/tabs at the beginning and end of each line
    text = "\n".join(
        line.strip()
        for line in text.split("\n")
    )

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3 or more consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def clean_document(document: Document) -> Document:
    """
    Clean the text while preserving document metadata.
    """

    cleaned_text = clean_text(document.text)

    return Document(
        text=cleaned_text,
        source=document.source,
        file_type=document.file_type,
        page_number=document.page_number,
        language=document.language,
    )