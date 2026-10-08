from dataclasses import dataclass
from typing import Optional

from src.ingestion.document import Document


@dataclass
class Chunk:
    text: str
    source: str
    file_type: str
    page_number: Optional[int] = None
    language: Optional[str] = None
    chunk_id: Optional[int] = None


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Split a document into overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=document.source,
                    file_type=document.file_type,
                    page_number=document.page_number,
                    language=document.language,
                    chunk_id=chunk_id,
                )
            )

            chunk_id += 1

        start = end - overlap

    return chunks