# Multilingual RAG System

A multilingual Retrieval-Augmented Generation (RAG) system that answers questions using information retrieved from documents. The system combines multilingual embeddings, semantic search, keyword-based retrieval, reranking, and LLM-powered answer generation.

It exposes a REST API using FastAPI and supports containerization with Docker.

## Features

- **Document ingestion:** Load TXT, PDF, and DOCX files.
- **Text cleaning:** Normalize and clean extracted document text.
- **Text chunking:** Split documents into smaller overlapping chunks.
- **Multilingual embeddings:** Generate embeddings using `BAAI/bge-m3`.
- **Vector database:** Store and retrieve document embeddings using ChromaDB.
- **Semantic retrieval:** Retrieve relevant document chunks based on semantic similarity.
- **Hybrid retrieval:** Combine semantic search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Reranking:** Reorder retrieved results using a cross-encoder.
- **LLM generation:** Generate answers using the Groq API.
- **Source citations:** Return source document names and available page numbers.
- **Evaluation:** Evaluate retrieval and generated answers using a small test dataset.
- **REST API:** Expose the RAG pipeline through FastAPI.
- **Interactive documentation:** Test API endpoints through Swagger UI.
- **Docker support:** Run the application in a container.

## Architecture

```text
                DOCUMENT INGESTION
                         |
                 TXT / PDF / DOCX
                         |
                         v
                   Text Cleaning
                         |
                         v
                    Text Chunking
                         |
                         v
               Multilingual Embeddings
                     (BGE-M3)
                         |
                         v
                      ChromaDB
                  Vector Database


                     USER QUERY
                         |
                         v
                   Query Processing
                         |
                         v
                   Hybrid Retrieval
                  /               \
                 v                 v
          Semantic Search      BM25 Search
                  \               /
                   v             v
               Reciprocal Rank Fusion
                         |
                         v
                      Reranking
                         |
                         v
                  Prompt Construction
                         |
                         v
                     Groq LLM
                         |
                         v
                 Generated Answer
                         |
                         v
                  Source Citations
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Embedding model | BAAI/bge-m3 |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Ranking fusion | Reciprocal Rank Fusion |
| Reranking | Sentence Transformers CrossEncoder |
| LLM provider | Groq |
| API framework | FastAPI |
| API validation | Pydantic |
| API documentation | Swagger UI |
| Containerization | Docker |

## Project Structure

```text
multi-lingual-RAG/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   └── vector_db/             # Local ChromaDB persistence
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── evaluation/
│   └── api/
├── tests/
├── notebooks/
├── results/
├── config/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── main.py
```

## Getting Started

### Prerequisites

- Python 3.12
- Git
- A Groq API key
- Docker Desktop (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/Aagashaaa/multi-lingual-RAG.git
cd multi-lingual-RAG
```


### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The first run may download the embedding and reranking models, so initial startup can take longer.

### 4. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your actual API key.

**Security:** Never commit `.env` or expose your API key in source code, screenshots, logs, or public repositories.

### 5. Add Documents

Place your documents in:

```text
data/raw/
```

Supported formats:

- `.txt`
- `.pdf`
- `.docx`

The current demonstration uses `data/raw/leave_policy.txt`.

## Running the RAG Demonstration

Run the main script from the project root:

```powershell
python main.py
```

The current `main.py` runs the complete demonstration pipeline:

1. Loads the sample leave-policy document.
2. Cleans and normalizes the text.
3. Splits the document into overlapping chunks.
4. Generates multilingual embeddings.
5. Stores chunks and embeddings in ChromaDB.
6. Processes a Tamil question.
7. Retrieves relevant chunks using hybrid retrieval.
8. Reranks the retrieved results.
9. Builds a prompt using the retrieved context.
10. Generates an answer through the Groq API.
11. Prints the answer and source citations.

**Important:** This is a demonstration pipeline, not a dedicated indexing-only command. The script currently uses a predefined Tamil question and the sample document.

## Running the FastAPI Server

Start the API from the project root:

```powershell
python -m uvicorn src.api.app:app --reload
```

Open the interactive API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The API loads its RAG components when the application starts. Make sure the ChromaDB database contains indexed documents before asking questions.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API status |
| POST | `/ask` | Ask a question about indexed documents |

### Example Request

Use `POST /ask` in Swagger UI.

Request body:

```json
{
  "question": "How many annual leave days do employees receive?",
  "answer_language": "English"
}
```

### Example Response

The response follows the API's `AnswerResponse` schema:

```json
{
  "question": "How many annual leave days do employees receive?",
  "answer": "Employees receive 20 days of annual leave every calendar year.",
  "sources": [
    "leave_policy.txt"
  ]
}
```

The generated wording may vary. The example illustrates the response format and expected answer for the current sample document.

## Multilingual and Cross-Lingual Retrieval

The system uses `BAAI/bge-m3`, a multilingual embedding model, to represent queries and document chunks in a shared semantic space.

This enables experiments in which a question in one language retrieves relevant information from a document written in another language.

**Example**

Source document in English:

> Employees receive 20 days of annual leave every calendar year.

Question in Tamil:

> ஊழியர்களுக்கு ஆண்டுக்கு எத்தனை நாட்கள் விடுப்பு கிடைக்கும்?

The retrieval pipeline searches for relevant document chunks, and the LLM generates an answer in the requested language.

The embedding model supports multilingual retrieval, but overall cross-lingual quality also depends on the reranker, prompt, and LLM. Multilingual performance requires further evaluation.

## Retrieval and Generation Evaluation

The evaluation dataset is stored at:

```text
data/evaluation/rag_eval.json
```

It currently contains four test questions covering:

- Annual leave entitlement
- Carry-forward leave limits
- Manager approval
- Tamil-language retrieval and answer generation

### Run Retrieval Evaluation

```powershell
python -m src.evaluation.evaluate_rag
```

### Run Generation Evaluation

```powershell
python -m src.evaluation.evaluate_generation
```

### Initial Results

| Metric | Result |
|---|---:|
| Retrieval hit rate at 3 | 100% |
| Generation keyword-check pass rate | 100% |
| Evaluation questions | 4 |

These are initial results on a small test dataset, not evidence of general or production-level accuracy.

The retrieval metric checks whether the expected source appears among the top three retrieved results. The generation evaluation uses expected-keyword checks rather than a comprehensive assessment of factual correctness, completeness, or fluency.

A larger dataset with more languages, documents, and difficult questions is needed for a meaningful evaluation.

## Running with Docker

Docker has been tested locally with the FastAPI application.

### 1. Build the Image

Run from the project root:

```powershell
docker build -t multilingual-rag .
```

### 2. Start the Container

Make sure the `.env` file exists and contains `GROQ_API_KEY`.

```powershell
docker run -d --name multilingual-rag-api -p 8000:8000 --env-file .env -v "${PWD}/data/vector_db:/app/data/vector_db" multilingual-rag
```

This maps port `8000` on your computer to port `8000` in the container and mounts the local ChromaDB directory.

The database must already contain indexed documents, or the documents must be indexed through the application's demonstration pipeline.

### 3. Open the API Documentation

[http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Check Container Status

```powershell
docker ps -a
```

### 5. View Logs

```powershell
docker logs --tail 100 multilingual-rag-api
```

### 6. Stop and Restart

Stop the container:

```powershell
docker stop multilingual-rag-api
```

Start the existing container again:

```powershell
docker start multilingual-rag-api
```

Remove the container when it is no longer needed:

```powershell
docker rm multilingual-rag-api
```

Removing a container does not remove the image. Avoid deleting the mounted database directory if you want to preserve your indexed data.

## Security Considerations

- Store API keys in environment variables.
- Keep `.env` out of version control.
- Exclude virtual environments, caches, and local generated database files from Git.
- Avoid exposing the API publicly without authentication and appropriate access controls.
- Apply suitable resource limits and monitoring before production deployment.

## Current Limitations

- The evaluation dataset currently contains only four questions.
- Generation evaluation uses basic keyword checks.
- The current reranker, `cross-encoder/ms-marco-MiniLM-L-6-v2`, is primarily English-focused and may not rank multilingual results optimally.
- Character-based chunking can split sentences or words.
- The demonstration script uses a predefined document and question.
- The application has not been validated for production-scale traffic or reliability.

## Future Improvements

- Expand evaluation across more documents and languages.
- Improve chunking with sentence-aware boundaries.
- Evaluate multilingual reranking models.
- Add stronger retrieval and answer-quality metrics.
- Add automated unit and integration tests.
- Improve API authentication, error handling, and observability.
- Evaluate performance under larger document collections and concurrent requests.
- Improve deployment and configuration management.

## License

No license has been specified yet. Add an appropriate open-source license file before distributing the project under a particular license.
