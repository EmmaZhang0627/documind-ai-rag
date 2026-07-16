# DocuMind

DocuMind is a FastAPI-based RAG document question-answering system for PDFs. It lets users upload and parse documents, indexes page-aware chunks with embeddings and hybrid retrieval, and returns source-grounded answers with citations, status, and trace IDs for debugging.

## Problem Statement

Teams often need to ask questions over internal PDFs, but raw LLM calls are not grounded, cannot cite sources, and may hallucinate when the answer is not present. DocuMind addresses this by building a controlled RAG pipeline with retrieval, reranking fallback, confidence gating, citation metadata, observability traces, and a lightweight evaluation runner.

## Key Features

- PDF upload and page-aware parsing
- Text chunking with `document_id`, `source_file`, `page_number`, and `chunk_index`
- OpenAI embedding generation
- In-memory vector store for local MVP development
- Hybrid retrieval using embedding similarity and BM25 keyword score fusion
- Optional cross-encoder reranking with fallback ranking when local dependencies fail
- Confidence gate for low-evidence questions
- LLM answer generation with source citations
- Thin FastAPI routers and service-layer orchestration
- Centralized runtime configuration
- `trace_id` per chat request with JSONL trace logging
- Lightweight local evaluation dataset and regression runner

## Tech Stack

Frontend:

- Vue 3
- TypeScript
- Vite
- Ant Design Vue

Backend:

- Python
- FastAPI
- Pydantic
- PyMuPDF / `fitz`
- OpenAI API
- `rank-bm25`
- `sentence-transformers` for optional cross-encoder reranking
- Local JSONL logs for lightweight observability

## High-Level Architecture

```text
Client / Swagger
      |
      v
FastAPI Routers
  - documents.py
  - chat.py
  - traces.py
      |
      v
Dependency Provider
  - get_settings()
  - get_rag_service()
      |
      v
RAGService Orchestrator
      |
      +--> EmbeddingService
      +--> RetrievalService + InMemoryVectorStore
      +--> RerankService
      +--> LLMService
      +--> Trace Logger
      +--> Evaluation Runner
```

## RAG Pipeline

```text
PDF upload / parse
      |
      v
Page-aware text extraction
      |
      v
Chunking with page metadata
      |
      v
Embedding generation
      |
      v
Hybrid retrieval
  - embedding similarity
  - BM25 keyword score
  - weighted fusion
      |
      v
Rerank or fallback ranking
      |
      v
Confidence gate
      |
      +--> low confidence refusal
      |
      v
LLM generation
      |
      v
Answer + sources + status + trace_id
```

## Run Locally

From the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set the required OpenAI key:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Start the backend:

```powershell
$env:PYTHONPATH="."
uvicorn app.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

## Test The API

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Upload or parse a PDF through Swagger:

```text
POST /api/documents/upload
POST /api/documents/parse-pdf
```

Ask a grounded question after parsing a PDF:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/chat `
  -ContentType "application/json" `
  -Body '{"question":"Ask a question answered by the uploaded PDF.","top_k":3}'
```

Ask an unrelated question to test low-confidence behavior:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/chat `
  -ContentType "application/json" `
  -Body '{"question":"What is the CEO birthday on Mars?","top_k":3}'
```

View recent local traces:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/traces/latest?limit=5"
Get-Content .\logs\rag_traces.jsonl -Tail 5
```

## Run Evaluation

The local evaluation runner reads cases from `eval/documind_eval_cases.json`, calls `RAGService.ask()`, and writes results to `eval/eval_results_latest.json`.

```powershell
python eval/run_eval.py
```

The current sample cases include placeholders. Replace the grounded placeholder with a real question from a stable evaluation PDF before treating failures as product regressions.

## Documentation

- [Architecture](docs/architecture.md)
- [RAG Pipeline](docs/rag_pipeline.md)
- [Evaluation And Observability](docs/evaluation_and_observability.md)
- [Interview Notes](docs/interview_notes.md)

## Current Limitations

- The vector store is in-memory and not production-ready.
- Indexed documents are not persisted across process restarts.
- Chroma, Qdrant, or pgvector would be better future storage backends.
- Cross-encoder reranking is optional and may fall back locally because PyTorch / Windows dependencies can fail.
- JSONL trace logging is an MVP observability solution, not a production monitoring stack.
- The evaluation dataset is lightweight and should grow with real user questions and domain-reviewed cases.
- The current eval runner is local-only and can later be upgraded with RAGAS, LLM-as-judge, CI, or dashboard reporting.
- Authentication, authorization, tenant isolation, database persistence, and deployment hardening are not implemented in this phase.

## Future Roadmap

- Replace in-memory vector store with Chroma, Qdrant, or pgvector
- Persist documents, chunks, embeddings, and metadata
- Add authentication and access control
- Improve frontend workflows for upload, chat, and trace inspection
- Add production observability with metrics, latency, token usage, and dashboards
- Expand the eval dataset with real domain cases and bug-derived regression cases
- Add CI-based eval checks for critical cases
- Improve reranker deployment reliability outside the local Windows MVP environment
