# DocuMind RAG Pipeline

This document explains the RAG pipeline used by DocuMind from PDF ingestion to source-grounded answer generation.

## Pipeline Overview

```text
PDF
 |
 v
Page-aware parsing
 |
 v
Chunking
 |
 v
Embedding
 |
 v
Hybrid retrieval
 |
 v
Rerank or fallback ranking
 |
 v
Confidence gate
 |
 +--> low-confidence refusal
 |
 v
Context building
 |
 v
LLM generation
 |
 v
Answer + sources + status + trace_id
```

## 1. PDF Ingestion

PDF ingestion starts in `POST /api/documents/parse-pdf`.

The router validates that the uploaded file is a PDF, reads the file, saves a temporary copy, and parses it with `fitz` / PyMuPDF.

Router responsibilities stay limited to:

- file validation
- PDF request handling
- parsing coordination
- response formatting

Embedding and vector-store writes are delegated to `RAGService.ingest_document()`.

## 2. Page-Aware Parsing

The parser extracts text page by page and stores page metadata:

```text
page_number
text
```

This matters because answer citations should point back to the original document page. Without page-level metadata, the system could answer correctly but fail to explain where the evidence came from.

## 3. Chunking

Parsed pages are passed into the chunker. Each chunk preserves:

```text
document_id
source_file
page_number
chunk_index
content
```

Chunks are the retrieval units. They should be large enough to preserve meaning but small enough for focused retrieval.

Trade-off:

- larger chunks preserve more context but may add noise
- smaller chunks retrieve more precisely but may lose surrounding context

## 4. Embedding

During ingestion, each chunk receives an embedding:

```text
chunk content -> OpenAI embedding vector
```

During chat, the query is embedded:

```text
user question -> OpenAI embedding vector
```

Embedding similarity helps find semantically related chunks even when the user does not use the exact same wording as the document.

Failure modes:

- missing `OPENAI_API_KEY`
- OpenAI network/API failure
- timeout
- embedding model mismatch across indexing and querying

## 5. Hybrid Retrieval

DocuMind uses hybrid retrieval:

```text
embedding similarity + BM25 keyword score
```

Embedding similarity captures semantic meaning. BM25 captures keyword overlap and exact terms. The two scores are fused with configurable weights:

```text
final retrieval score =
  embedding_weight * embedding_score
  + bm25_weight * bm25_score
```

Current defaults are configured in `AppSettings`:

```text
embedding_score_weight = 0.7
bm25_score_weight = 0.3
```

Trade-off:

- too much embedding weight may miss exact keyword constraints
- too much BM25 weight may miss paraphrases
- weights should be tuned with evaluation cases

## 6. In-Memory Vector Store

The current vector store is an in-memory implementation.

It is useful for local MVP development because it is simple and avoids external infrastructure. It is not production-ready because indexed data is lost on process restart and it does not provide persistence, multi-user isolation, or scalable retrieval.

Future replacements:

- Chroma
- Qdrant
- pgvector
- FAISS with persistence

## 7. Reranking And Fallback

After retrieval, candidates can be reranked.

The intended reranker is a cross-encoder:

```text
(query, candidate document) -> rerank_score
```

Cross-encoders are often more precise than first-stage retrieval because they evaluate query-document pairs directly.

However, local Windows environments can have PyTorch / model-loading issues. For that reason, reranking has fallback behavior:

The score fields have separate meanings:

```text
retrieval_score = weighted hybrid embedding/BM25 score
rerank_score = raw CrossEncoder relevance score, only when reranking runs
confidence_score = selected top candidate's retrieval_score
```

When the CrossEncoder is active, candidates are sorted by `rerank_score` while
their original `retrieval_score` values are preserved. When it is disabled,
unavailable, or prediction fails, candidates are sorted by `retrieval_score`,
`rerank_enabled` is `false`, and `rerank_score` is `null`. A retrieval score is
not copied into the rerank field because the two values have different meanings.

This keeps the API available even when the optional reranker is unavailable.

Trade-off:

- cross-encoder reranking improves precision but adds latency and dependency risk
- fallback ranking is more robust locally but may be less precise

## 8. Confidence Gate

After ranking, DocuMind selects the top-ranked candidate and checks its stable
hybrid retrieval score against the confidence threshold:

```text
selected_candidate = ranked_candidates[0]
confidence_score = selected_candidate.retrieval_score

if confidence_score >= confidence_threshold:
    answer
else:
    refuse
```

The goal is to prevent the LLM from answering when retrieval evidence is weak.
Reranking controls which candidate is selected, but its raw score does not pass
through the confidence gate. This remains the design until reranker calibration
is supported by enough evaluation data.

Low-confidence behavior:

```text
answer = "I cannot find relevant information in the documents."
sources = []
status = "low_confidence"
llm_called = false in trace logs
```

Trade-off:

- threshold too low can allow hallucinations
- threshold too high can block valid answers
- threshold should be tuned with evaluation cases and trace data

## 9. Context Building

For confident requests, DocuMind takes the top ranked chunks and joins their document text into a context string.

Only selected chunks are sent to the LLM. The whole PDF is not sent.

Benefits:

- lower token cost
- less irrelevant context
- clearer source grounding
- easier debugging

## 10. LLM Generation

`LLMService` receives:

```text
query
context
```

It returns the final answer. The API response also includes the source metadata for the chunks used as evidence.

Failure modes:

- OpenAI API/network error
- timeout
- weak prompt constraints
- correct context but poor generation

## 11. Citation Output

Each successful answer returns sources:

```text
document_id
source_file
page_number
chunk_index
```

This allows users and engineers to verify where the answer came from.

## 12. Low-Confidence Refusal

If evidence is weak, DocuMind refuses rather than forcing an answer.

This is important because a RAG system should not answer every question. It should answer when grounded evidence exists and refuse when it does not.

## Failure Modes And Debugging

Common failure modes:

- no chunks indexed
- PDF parsing extracted little or no text
- embeddings failed
- vector store is empty after process restart
- BM25 score is weak for paraphrased questions
- embedding score is weak for exact keyword-heavy questions
- reranker dependency failed and fallback was used
- confidence threshold is too high or too low
- retrieved context is correct but LLM answer is poor

Debugging path:

1. inspect API `status`
2. find the request `trace_id`
3. inspect `logs/rag_traces.jsonl`
4. check retrieved candidate count
5. check top candidate metadata
6. compare embedding, BM25, final, and rerank scores
7. check confidence decision
8. if LLM was called and context is correct, inspect generation and prompt behavior
