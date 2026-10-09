
from pathlib import Path

import chromadb

from src.chunking.text_chunker import Chunk


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "data/vector_db",
        collection_name: str = "multilingual_documents",
    ):
        self.persist_directory = str(Path(persist_directory))
        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return

        ids = [
            f"{chunk.source}_{chunk.page_number or 0}_{chunk.chunk_id}_{i}"
            for i, chunk in enumerate(chunks)
        ]

        documents = [chunk.text for chunk in chunks]

        metadatas = [
            {
                "source": chunk.source,
                "file_type": chunk.file_type,
                "page_number": chunk.page_number or 0,
                "language": chunk.language or "unknown",
                "chunk_id": chunk.chunk_id if chunk.chunk_id is not None else i,
            }
            for i, chunk in enumerate(chunks)
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()