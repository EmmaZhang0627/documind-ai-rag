# Phase 2 Baseline Audit

Date: 2026-07-20

This audit describes the current DocuMind implementation before adding new retrieval features. It is based on the repository state in `D:\HuaweiMoveData\Users\Emma\Desktop\codex-career\documind\documind-ai-rag`.

## 1. Current Repository Structure

### Backend

- `backend/app/main.py`
  - FastAPI application entry point.
  - Registers the document, chat, and trace routers.
  - Exposes `GET /health`.

- `backend/app/api/documents.py`
  - Document upload and PDF parsing API.
  - `POST /api/documents/upload` saves a PDF to `backend/uploads` but does not parse or index it.
  - `POST /api/documents/parse-pdf` reads the uploaded PDF, extracts page text with PyMuPDF, chunks pages, embeds chunks, and indexes them.

- `backend/app/api/chat.py`
  - Chat API.
  - `POST /api/chat` calls `RAGService.ask()` and returns `trace_id`, `answer`, `sources`, `trace`, `status`, and optional `fallback_reason`.

- `backend/app/api/traces.py`
  - Trace inspection API.
  - `GET /api/traces/latest` reads recent JSONL trace records.

- `backend/app/dependencies/rag_dependencies.py`
  - Builds the shared cached `RAGService`.
  - Wires settings, OpenAI embedding provider, in-memory retrieval, rerank wrapper, and OpenAI LLM provider.

- `backend/app/config/settings.py`
  - Environment-backed settings for OpenAI models, retrieval top-k values, confidence threshold, hybrid weights, and reranker configuration.

- `backend/app/services/chunker.py`
  - Chunking implementation.
  - `split_pages_into_chunks()` is used by the main PDF parsing flow.
  - `split_text_into_chunks()` exists but is not used by the current `parse-pdf` path.

- `backend/app/services/embedding.py`
  - OpenAI embedding provider.
  - `create_openai_embedding_model()` creates a lazy OpenAI client and calls `client.embeddings.create()`.

- `backend/app/services/embedding_service.py`
  - Thin wrapper around the configured embedding callable.

- `backend/app/services/vector_db.py`
  - Current vector database implementation, despite the filename.
  - Uses process-local Python lists: `vector_store` and `corpus`.
  - Implements cosine similarity, BM25 indexing, BM25 score normalization, hybrid score fusion, optional cross-encoder reranking helper, confidence check, and a clear helper.

- `backend/app/services/retrieval_service.py`
  - Retrieval abstraction.
  - `InMemoryVectorStore` delegates to `add_chunks_to_db()` and `retrieve_candidates()`.
  - `RetrievalService` is the service wrapper used by `RAGService`.

- `backend/app/services/rerank_service.py`
  - Thin wrapper around the configured rerank callable.

- `backend/app/services/rag.py`
  - Main RAG orchestration.
  - Handles ingestion, query embedding, retrieval, reranking, confidence/refusal logic, context construction, LLM generation, source conversion, and observability trace creation.

- `backend/app/services/llm_service.py`
  - OpenAI answer generation provider.
  - Prompts the model to answer only from supplied context.

- `backend/app/observability/trace_models.py`
  - Typed trace schema definitions.

- `backend/app/observability/trace_logger.py`
  - Writes JSONL traces to `logs/rag_traces.jsonl`.
  - Reads recent traces for the trace API.

### Frontend

- `frontend/src/App.vue`
  - Current main Vue component.
  - Supports backend health check and PDF upload/parse.
  - Does not currently implement a chat UI or source/citation display.

- `frontend/src/services/app.ts`
  - Axios client.
  - Implements `healthCheck()` and `uploadDocument()`.
  - Calls `POST /api/documents/parse-pdf`.

- `frontend/src/services/api.ts`
  - Re-exports the API helpers from `app.ts`.

### Evaluation And Docs

- `eval/run_eval.py`
  - Local evaluation runner.
  - Imports the shared RAG service and calls `RAGService.ask()` for each case.
  - Writes results to `eval/eval_results_latest.json`.

- `eval/documind_eval_cases.json`
  - Local evaluation cases.
  - Currently includes one grounded placeholder case and fallback/refusal cases.

- `docs/architecture.md`, `docs/rag_pipeline.md`, `docs/evaluation_and_observability.md`, `docs/responsible_fallback.md`
  - Existing architecture, pipeline, observability, and fallback documentation.

## 2. Real Document Indexing Flow

### Call Path

PDF upload and indexing currently happens through `POST /api/documents/parse-pdf`, not `POST /api/documents/upload`.

1. `backend/app/main.py`
   - Registers `documents_router`.

2. `backend/app/api/documents.py`
   - `parse_pdf(file, rag_service)`
   - Validates `file.content_type == "application/pdf"`.
   - Reads the uploaded file bytes.
   - Writes the file to `uploads/{uuid}.pdf` as a temporary local file.
   - Opens the file with `fitz.open(temp_path)`.

3. `backend/app/api/documents.py`
   - Iterates through pages with `enumerate(doc)`.
   - Extracts page text with `page.get_text()`.
   - Appends non-empty pages as dictionaries containing:
     - `page_number`
     - `text`
   - Builds a `full_text` preview string.
   - Creates a fresh `document_id`.

4. `backend/app/services/chunker.py`
   - `split_pages_into_chunks(pages, document_id, source_file)`
   - Splits each page into character-based chunks with default `chunk_size=800` and `overlap=100`.
   - Preserves:
     - `document_id`
     - `source_file`
     - `page_number`
     - `chunk_index`
     - `start_char`
     - `end_char`
     - `content`

5. `backend/app/api/documents.py`
   - Calls `await run_in_threadpool(rag_service.ingest_document, chunks)`.

6. `backend/app/services/rag.py`
   - `RAGService.ingest_document(chunks)`
   - For each chunk, calls `self.embedder.embed(chunk["content"])`.
   - Stores the returned embedding on `chunk["embedding"]`.
   - Calls `self.retriever.add(chunks)`.

7. `backend/app/services/embedding_service.py`
   - `EmbeddingService.embed(text)` delegates to the configured embedding model.

8. `backend/app/services/embedding.py`
   - `create_openai_embedding_model(settings)` returns `embed(text)`.
   - `embed(text)` requires `OPENAI_API_KEY`.
   - Lazily creates an OpenAI client and calls `client.embeddings.create(model=settings.embedding_model_name, input=text)`.

9. `backend/app/services/retrieval_service.py`
   - `RetrievalService.add(chunks)` delegates to `InMemoryVectorStore.add(chunks)`.
   - `InMemoryVectorStore.add(chunks)` calls `add_chunks_to_db(chunks)`.

10. `backend/app/services/vector_db.py`
    - `add_chunks_to_db(chunks)`
    - Appends each chunk to the process-local `vector_store`.
    - Stores chunk text in `corpus`.
    - Preserves metadata:
      - `document_id`
      - `source_file`
      - `chunk_index`
      - `page_number`
    - Rebuilds the BM25 index with `_rebuild_bm25_index()`.

## 3. Real Question-Answering Flow

### Call Path

1. `backend/app/main.py`
   - Registers `chat_router`.

2. `backend/app/api/chat.py`
   - `chat(request, rag_service)`
   - Validates the Pydantic `ChatRequest`.
   - Calls `rag_service.ask(request.question, top_k=request.top_k)`.

3. `backend/app/services/rag.py`
   - `RAGService.ask(query, top_k)`
   - Creates a new `trace_id`.
   - Computes:
     - `answer_top_k = max(1, top_k or answer_top_k_default)`
     - `retrieval_top_k = max(retrieval_top_k_default, answer_top_k)`

4. `backend/app/services/rag.py`
   - Pre-retrieval guardrails:
     - `_detect_sensitive_input(query)`
     - `_detect_out_of_scope_decision_request(query)`
   - If either guardrail fires, the service returns a fallback response before embedding, retrieval, or LLM generation.

5. `backend/app/services/embedding_service.py` and `backend/app/services/embedding.py`
   - `self.embedder.embed(query)` creates the query embedding through OpenAI.

6. `backend/app/services/retrieval_service.py`
   - `self.retriever.retrieve(query_embedding, query, top_k=retrieval_top_k)`
   - Delegates to `InMemoryVectorStore.search()`.

7. `backend/app/services/vector_db.py`
   - `retrieve_candidates(query_embedding, query_text)`
   - Computes BM25 scores if `bm25_model` exists and `query_text` is not blank.
   - Normalizes BM25 scores with `_normalize_scores()`.
   - Computes embedding cosine similarity for every in-memory chunk.
   - Computes hybrid score:
     - `retrieval_score = embedding_score_weight * embedding_score + bm25_score_weight * bm25_score`
   - Sorts all candidates by `retrieval_score` descending.

8. `backend/app/services/rag.py`
   - `self.reranker.rerank(query, candidates)` reranks the retrieved candidates.

9. `backend/app/services/rerank_service.py` and `backend/app/services/vector_db.py`
   - `RerankService.rerank()` delegates to `vector_db.rerank()`.
   - `vector_db.rerank()` attempts to load `sentence_transformers.CrossEncoder`.
   - If disabled, unavailable, or failing, `_fallback_rerank()` sorts by retrieval score and marks `rerank_enabled=False`.

10. `backend/app/services/rag.py`
    - Selects `top_chunks = ranked[:answer_top_k]`.
    - Computes `top1_score` from `candidate["score"]` if present, otherwise `candidate["retrieval_score"]`.
    - Builds the in-response trace with `_build_trace()`.
    - Checks confidence with `is_confident(top1_score, threshold=self.confidence_threshold)`.

11. `backend/app/services/rag.py`
    - Post-retrieval fallback logic:
      - `_detect_conflicting_sources(top_chunks)`
      - `_has_usable_evidence(top_chunks)`
      - low-confidence gate
    - Fallback responses are built by `_build_fallback_response()` and log observability traces with `llm_called=False`.

12. `backend/app/services/rag.py`
    - If confidence passes:
      - Builds context with `"\n\n".join([chunk["document"] for chunk in top_chunks])`.
      - Calls `self.llm.generate(query, context)`.

13. `backend/app/services/llm_service.py`
    - `LLMService.generate(query, context)` delegates to the OpenAI Responses API.
    - The prompt instructs the model to answer only from the supplied context.

14. `backend/app/services/rag.py`
    - Converts top chunks into sources with `_candidate_to_source()`.
    - Adds `source_snippet` to source metadata.
    - Logs a structured JSONL trace with `_log_observability_trace()`.
    - Returns:
      - `trace_id`
      - `answer`
      - `sources`
      - `trace`
      - `status`
      - `fallback_reason`

15. `backend/app/api/chat.py`
    - Wraps the result in `ChatResponse` without changing the response structure.

## 4. Feature Status Table

| Feature | Status | Evidence | Notes / Risks |
|---|---|---|---|
| Page-level metadata | COMPLETE | `documents.py` extracts page numbers; `chunker.py` stores `page_number`; `vector_db.py` stores metadata; `rag.py` returns sources. | Only basic page number is captured; no section heading, bounding box, or PDF coordinate metadata. |
| Vector retrieval | COMPLETE | `vector_db.py` computes cosine similarity over stored embeddings; main flow calls it through `RetrievalService`. | In-memory only; no persistence or isolation. |
| BM25 retrieval | COMPLETE | `rank_bm25.BM25Okapi` is used in `vector_db.py`; BM25 index rebuilds on ingest. | Tokenization is simple `lower().split()`. |
| Hybrid retrieval | COMPLETE | `retrieve_candidates()` fuses embedding and BM25 scores with configured weights. | Weights are configurable but not tuned against real eval data. |
| Score normalisation | PARTIALLY COMPLETE | BM25 scores are max-normalized. | Embedding scores are not normalized; rerank scores are raw cross-encoder scores when reranker is active, which may not be comparable with the confidence threshold. |
| Query rewriting | NOT IMPLEMENTED | No query rewrite module, prompt, or call path found. | Good Phase 2 candidate after baseline eval exists. |
| Metadata filtering | NOT IMPLEMENTED | No retrieval-time filter arguments or metadata filter logic found. | Conflict detection looks for fields such as `document_status` and `version`, but current chunk metadata does not populate them. |
| Parent-child retrieval | NOT IMPLEMENTED | No parent document / child chunk hierarchy found. | Current retrieval unit is the chunk only. |
| Reranking | PARTIALLY COMPLETE | `vector_db.rerank()` supports CrossEncoder and fallback ranking; `RAGService.ask()` calls `self.reranker.rerank()`. | Model load and runtime behavior were not verified; active rerank scores may break confidence threshold semantics. |
| Low-confidence refusal | COMPLETE | `RAGService.ask()` calls `is_confident()` and returns `status="low_confidence"` without LLM call when the gate fails. | Threshold quality is unverified on real documents; threshold may behave differently if raw reranker scores are used. |
| Citations | PARTIALLY COMPLETE | Chat response includes source metadata and `source_snippet`. | No inline answer citations or frontend citation display currently exist. |
| Trace ID | COMPLETE | `RAGService.ask()` creates `trace_id`; response and JSONL traces include it. | There is also an older `vector_db.search()` trace helper that is not used in the main `RAGService.ask()` flow. |
| Retrieval score logging | COMPLETE | JSONL traces include top candidates with embedding, BM25, final, and optional rerank scores. | Only top ranked candidates are written, not the full candidate list. |
| Vector database persistence | NOT IMPLEMENTED | `vector_store` and `corpus` are process-local lists. | Data is lost on process restart. |
| Duplicate document handling | NOT IMPLEMENTED | `parse_pdf()` always creates a fresh UUID and appends chunks. | Re-uploading the same file creates duplicate chunks. |
| Automated retrieval evaluation | BROKEN OR UNVERIFIED | `eval/run_eval.py` exists, but current cases include a grounded placeholder and latest results show 2/4 passed with OpenAI connection errors for retrieval-dependent cases. | Running the eval mutates `eval/eval_results_latest.json` and may call OpenAI, so it was not rerun during this safe audit. |
| Docker deployment | NOT IMPLEMENTED | No `Dockerfile` or `docker-compose` files found. | Deployment is local/manual at this stage. |
| Redis integration | NOT IMPLEMENTED | No Redis dependency, configuration, or code path found. | No caching/session/background queue integration exists. |
| Frontend chat and source display | NOT IMPLEMENTED | `App.vue` only implements health check and PDF upload/parse preview. | Backend chat API exists, but the Vue UI does not expose it. |

## 5. Safe Verification Results

No destructive checks were run. No documents were deleted, the vector store was not reset, and environment variables were not changed.

### Commands Executed

- `rg --files`
- `git status --short`
- Targeted `rg -n` searches across `backend`, `frontend`, `eval`, `docs`, and `README.md`
- `rg -n "^" ...` on key backend, frontend, and eval files
- `python -m compileall backend\app`
- `npm run build` from `frontend`
- `python -c "import sys; sys.path.insert(0, 'backend'); import app.main; ..."` with global Python
- `python -m pip show rank-bm25` with global Python
- `python -c "import fastapi, openai, fitz; ..."` with global Python
- `backend\.venv\Scripts\python.exe -m compileall backend\app`
- `backend\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'backend'); import app.main; ..."`
- `backend\.venv\Scripts\python.exe -m pip show rank-bm25`
- `backend\.venv\Scripts\python.exe -c "import fastapi, openai, fitz, rank_bm25; ..."`

### Verification Outcome

- Backend syntax check passed with global Python.
- Frontend production build passed.
- Global Python import check failed because the global interpreter does not have `fastapi` or `rank_bm25`.
- Backend virtualenv import check passed.
- Backend virtualenv service construction check passed and returned `RAGService`.
- `rank-bm25==0.2.2` is installed in `backend/.venv`.
- Existing eval runner was not executed because it writes to `eval/eval_results_latest.json` and may call OpenAI.

## 6. Known Technical Risks

1. The main retrieval store is in-memory only.
   - Documents, chunks, embeddings, and BM25 state disappear on backend restart.
   - There is no multi-user isolation or production persistence.

2. Duplicate uploads are not handled.
   - The same PDF can be uploaded repeatedly and indexed as separate chunks.
   - This can distort both embedding retrieval and BM25 scoring.

3. Reranker scoring may be semantically incompatible with the confidence gate.
   - When CrossEncoder reranking succeeds, `top1_score` may become a raw rerank score.
   - The low-confidence threshold was designed around retrieval score behavior, not necessarily cross-encoder logits.

4. Evaluation is not yet grounded in a real stable document.
   - The grounded eval case is still a placeholder.
   - Latest saved eval results show retrieval-dependent failures caused by OpenAI connection errors.

5. Metadata model is too thin for enterprise retrieval.
   - Current chunk metadata has document, file, page, and chunk identity.
   - Conflict detection references richer fields such as `version` and `document_status`, but current ingestion does not populate them.

6. Frontend is not aligned with backend RAG capability.
   - Backend chat and source response shape exists.
   - Frontend currently has no chat input, answer view, trace ID display, or source display.

7. `POST /api/documents/upload` and `POST /api/documents/parse-pdf` are separate and easy to confuse.
   - `upload` saves only.
   - `parse-pdf` performs the actual indexing flow.

8. Temporary upload files are not cleaned up by `parse_pdf()`.
   - This avoids data deletion during current development, but can accumulate files over time.

9. Active environment matters.
   - The backend virtualenv imports correctly.
   - The global Python environment cannot import required backend dependencies.

## 7. Recommended Next Three Implementation Tasks

1. Build a real Phase 2 retrieval evaluation baseline.
   - Replace the grounded placeholder case with one or more questions from a stable local evaluation PDF.
   - Add expected `source_file`, `page_number`, answer keywords, and evidence keywords.
   - Run the eval only after the document is indexed and OpenAI connectivity is confirmed.

2. Stabilize retrieval scoring before adding new retrieval features.
   - Separate `retrieval_score`, `rerank_score`, and `confidence_score`.
   - Keep the API response structure unchanged.
   - Ensure low-confidence gating uses a calibrated score with predictable range.

3. Add duplicate-document detection at ingestion.
   - Compute a file hash or content hash during `parse_pdf()`.
   - Store enough metadata to detect repeated uploads.
   - Preserve current behavior until the duplicate policy is explicit.

## 8. Files Likely To Be Modified For The Next Task

For the recommended evaluation baseline task:

- `eval/documind_eval_cases.json`
- `eval/run_eval.py`
- `eval/README.md`
- `docs/evaluation_and_observability.md`

For retrieval scoring stabilization:

- `backend/app/services/rag.py`
- `backend/app/services/vector_db.py`
- `backend/app/services/rag_types.py`
- `backend/app/observability/trace_models.py`
- `docs/rag_pipeline.md`
- `docs/evaluation_and_observability.md`

For duplicate document handling:

- `backend/app/api/documents.py`
- `backend/app/services/rag.py`
- `backend/app/services/vector_db.py`
- `backend/app/services/rag_types.py`
- `docs/rag_pipeline.md`
