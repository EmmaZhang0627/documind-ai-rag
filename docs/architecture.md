# DocuMind Architecture

DocuMind is organized as a layered FastAPI RAG application. The main design goal is to keep HTTP handling, configuration, object construction, RAG orchestration, component implementation, observability, and evaluation separated.

## Architecture Diagram

```text
User / Swagger / Frontend
        |
        v
API Layer
  backend/app/api/
  - documents.py
  - chat.py
  - traces.py
        |
        v
Dependency Provider
  backend/app/dependencies/rag_dependencies.py
  - get_settings()
  - get_rag_service()
        |
        v
Config Layer
  backend/app/config/settings.py
  - OpenAI key
  - model names
  - top_k defaults
  - confidence threshold
  - hybrid weights
  - reranker flag
        |
        v
Service Layer
  backend/app/services/
        |
        v
RAGService Orchestrator
  - embedding
  - retrieval
  - reranking / fallback
  - confidence gate
  - LLM generation
  - citations
  - trace logging
        |
        +--> EmbeddingService
        +--> RetrievalService
        +--> RerankService
        +--> LLMService
        +--> Observability / Trace Logger
        +--> Evaluation Runner
```

## API Layer

The API layer is implemented with FastAPI routers:

- `backend/app/api/documents.py`
- `backend/app/api/chat.py`
- `backend/app/api/traces.py`

Router responsibilities are intentionally small:

- receive HTTP requests
- validate request input
- map expected errors into HTTP responses
- call service-layer methods
- shape API responses

Routers do not implement embedding, retrieval, reranking, confidence gating, context building, or LLM generation.

## Dependency Provider

`backend/app/dependencies/rag_dependencies.py` builds and caches the service graph.

It provides:

- `get_settings()`
- `get_rag_service()`

This keeps heavy service construction out of request handlers. It also creates a single place to swap providers later, such as replacing the in-memory vector store with Chroma, Qdrant, or pgvector.

## Config Layer

`backend/app/config/settings.py` centralizes runtime configuration using `AppSettings.from_env()`.

It includes:

- `OPENAI_API_KEY`
- embedding model name
- chat model name
- OpenAI timeout
- retrieval top_k default
- answer top_k default
- confidence threshold
- embedding score weight
- BM25 score weight
- reranker enabled flag
- reranker model name

The application should not scatter these values across routers or services.

## Service Layer

The service layer lives under `backend/app/services/`.

It contains:

- `RAGService`: orchestrates the full RAG workflow
- `EmbeddingService`: wraps embedding generation
- `RetrievalService`: wraps vector store search and ingest
- `RerankService`: wraps reranking logic
- `LLMService`: wraps answer generation
- `InMemoryVectorStore`: local MVP vector store adapter
- `rag_types.py`: shared typed interfaces

This structure makes each capability replaceable without rewriting the entire pipeline.

## RAGService Orchestrator

`RAGService` is the pipeline controller. It does not parse HTTP requests and does not own FastAPI concerns.

For chat requests, it:

1. generates a `trace_id`
2. embeds the query
3. retrieves candidates
4. reranks candidates or uses fallback ranking
5. checks the confidence gate
6. refuses low-confidence questions
7. builds context for confident questions
8. calls the LLM
9. returns answer, sources, status, trace, and trace_id
10. writes structured trace records

For ingestion, it embeds chunks and adds them to the retrieval layer.

## EmbeddingService

`EmbeddingService` receives an embedding provider function and exposes:

```python
embed(text: str) -> list[float]
```

The current provider uses OpenAI embeddings. A future provider can be injected without changing `RAGService`.

## RetrievalService

`RetrievalService` wraps the vector store interface:

```python
add(chunks)
retrieve(query_embedding, query_text, top_k)
```

The current implementation uses an in-memory store and hybrid search. It is useful for a local MVP but not production persistence.

## RerankService

`RerankService` wraps reranking:

```python
rerank(query, candidates)
```

The current reranker can use a cross-encoder when available. If local dependencies fail, the system falls back to retrieval-score ordering so the API remains usable.

## LLMService

`LLMService` wraps answer generation:

```python
generate(query, context)
```

The current implementation uses OpenAI chat generation. A future provider can be wired through the dependency provider.

## Observability / Trace Logger

Observability lives under `backend/app/observability/`.

It provides:

- trace schema definitions
- JSONL trace writing
- latest trace reading for local debugging

Each chat request receives a `trace_id`. The trace logger writes one JSON object per line to:

```text
logs/rag_traces.jsonl
```

Trace logging is failure-safe: logging failure should not break chat responses.

## Evaluation Runner

The evaluation runner lives under `eval/`.

It provides:

- `documind_eval_cases.json`: local evaluation cases
- `run_eval.py`: lightweight regression runner
- `eval_results_latest.json`: latest generated result

The runner imports `RAGService` through the dependency provider and checks expected behavior such as answered-with-sources or low-confidence refusal.

## Replaceability

The architecture is intentionally designed around stable boundaries:

- Replace OpenAI embeddings by changing the embedding provider.
- Replace in-memory vector store by implementing a new vector store adapter.
- Replace reranker by injecting a different rerank function.
- Replace OpenAI LLM by changing `LLMService` or its provider.
- Replace JSONL tracing later with a real observability platform.
- Replace local eval later with RAGAS, LLM-as-judge, or CI-based scoring.
