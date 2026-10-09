from sentence_transformers import SentenceTransformer

from src.chunking.text_chunker import Chunk


class MultilingualEmbedder:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        
        """
        Convert a single text into an embedding vector.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """
        Convert multiple chunks into embedding vectors.
        """

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()