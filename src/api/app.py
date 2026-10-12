
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.reranking.reranker import Reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_generator import LLMGenerator


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


# Initialize the RAG components once and reuse them across requests.
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
    try:
        query = process_query(request.question)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    if vector_store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="The vector database is empty. Index documents first.",
        )

    # 1. Retrieve relevant chunks
    candidates = retriever.search(
        query=query,
        top_k=5,
    )

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No relevant document chunks were retrieved.",
        )

    # 2. Rerank the retrieved chunks
    final_results = reranker.rerank(
        query=query,
        results=candidates,
        top_k=3,
    )

    # 3. Build the prompt and generate the answer
    prompt = build_prompt(
        query=query,
        retrieved_chunks=final_results,
        answer_language=request.answer_language,
    )

    answer = generator.generate(prompt)

    # 4. Format source citations from retrieved metadata
    sources = format_citations(final_results)

    return AnswerResponse(
        question=query,
        answer=answer,
        sources=sources,
    )
