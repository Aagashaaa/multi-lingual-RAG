
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.reranking.reranker import Reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_generator import LLMGenerator


# Configure application logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Multilingual RAG API",
    description="Ask questions about documents in multiple languages.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    answer_language: str = "same as the question"


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


# Initialize RAG components once and reuse them.
vector_store = VectorStore()
embedder = MultilingualEmbedder()

retriever = HybridRetriever(
    vector_store=vector_store,
    embedder=embedder,
)

reranker = Reranker()
generator = LLMGenerator()


def format_citations(results: list[dict]) -> list[str]:
    citations = []
    seen = set()

    for result in results:
        metadata = result.get("metadata", {})
        source = metadata.get("source")

        if not source:
            continue

        page_number = metadata.get("page_number", 0)

        if page_number:
            citation = f"{source}, page {page_number}"
        else:
            citation = source

        if citation not in seen:
            citations.append(citation)
            seen.add(citation)

    return citations


@app.get("/")
def home():
    return {
        "message": "Multilingual RAG API is running",
        "docs": "/docs",
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    total_start = time.perf_counter()

    logger.info("Received a RAG request.")

    try:
        query = process_query(request.question)
    except (TypeError, ValueError) as error:
        logger.warning("Invalid query received: %s", error)

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    if vector_store.count() == 0:
        logger.error("Vector database is empty.")

        raise HTTPException(
            status_code=503,
            detail="The vector database is empty. Index documents first.",
        )

    try:
        # 1. Retrieve relevant chunks
        retrieval_start = time.perf_counter()

        candidates = retriever.search(
            query=query,
            top_k=5,
        )

        retrieval_time = time.perf_counter() - retrieval_start

        logger.info(
            "Retrieval completed in %.3f seconds. Candidates: %d",
            retrieval_time,
            len(candidates),
        )

        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant document chunks were retrieved.",
            )

        # 2. Rerank retrieved chunks
        reranking_start = time.perf_counter()

        final_results = reranker.rerank(
            query=query,
            results=candidates,
            top_k=3,
        )

        reranking_time = time.perf_counter() - reranking_start

        logger.info(
            "Reranking completed in %.3f seconds. Results: %d",
            reranking_time,
            len(final_results),
        )

        # 3. Build prompt and generate answer
        prompt = build_prompt(
            query=query,
            retrieved_chunks=final_results,
            answer_language=request.answer_language,
        )

        generation_start = time.perf_counter()

        answer = generator.generate(prompt)

        generation_time = time.perf_counter() - generation_start

        logger.info(
            "LLM generation completed in %.3f seconds.",
            generation_time,
        )

        # 4. Format citations
        sources = format_citations(final_results)

        total_time = time.perf_counter() - total_start

        logger.info(
            "RAG request completed successfully in %.3f seconds.",
            total_time,
        )

        return AnswerResponse(
            question=query,
            answer=answer,
            sources=sources,
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error during RAG request.")
        raise
