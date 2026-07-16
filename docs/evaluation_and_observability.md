# Evaluation And Observability

RAG systems are difficult to debug from the final answer alone. A wrong answer can come from ingestion, chunking, embedding, retrieval, reranking, confidence gating, context construction, or LLM generation. DocuMind adds lightweight observability and evaluation to make these failure points visible.

## Why Observability Is Needed

Without traces, a user might only see:

```text
answer: ...
status: answered
sources: [...]
```

That is not enough to know whether the system retrieved the right evidence, reranked correctly, passed the confidence gate for the right reason, or called the LLM when it should not have.

Observability gives engineers a way to answer:

- Did retrieval find candidates?
- Were the candidates from the right document and page?
- Did reranking run or fall back?
- Did the confidence gate pass?
- Was the LLM called?
- What status did the system return?

## Trace ID Design

Each chat request receives a unique `trace_id` inside `RAGService.ask()`.

The same `trace_id` is returned in the API response and written to local JSONL logs.

This makes the trace ID a correlation ID:

```text
API response trace_id
        |
        v
logs/rag_traces.jsonl record
        |
        v
retrieval / rerank / confidence / LLM debugging
```

## JSONL Trace Logging

Trace logs are written to:

```text
logs/rag_traces.jsonl
```

JSONL means one JSON object per line:

```jsonl
{"trace_id":"...","final_status":"answered"}
{"trace_id":"...","final_status":"low_confidence"}
```

This format is useful for local MVP logging because new traces can be appended without rewriting the whole file. It can also be read line by line when the file grows.

Trace writing is failure-safe. If logging fails, the chat endpoint should still return normally.

## Logged Fields

Each trace record includes:

- `trace_id`
- `timestamp`
- `query`
- `query_length`
- `retrieval_top_k`
- `retrieved_candidate_count`
- `top_candidates`
- candidate metadata:
  - `document_id`
  - `source_file`
  - `page_number`
  - `chunk_index`
- scores:
  - `embedding_score`
  - `bm25_score`
  - `final_score`
  - `rerank_score`
- `confidence_threshold`
- `top1_score`
- `confidence_decision`
- `llm_called`
- `final_status`
- `error_message`

Long document text is not logged. If snippets are logged, they are capped to a short length to reduce noise and avoid exposing too much content.

## Low-Confidence Request Logging

When the confidence gate fails:

```text
confidence_decision = low_confidence
llm_called = false
final_status = low_confidence
```

This confirms that the system refused because retrieval evidence was weak and that it avoided an unnecessary LLM call.

A low-confidence trace is useful for tuning:

- confidence threshold
- retrieval weights
- chunking strategy
- reranking behavior

## Answered Request Logging

When the confidence gate passes:

```text
confidence_decision = confident
llm_called = true
final_status = answered
```

The trace also records the top candidate metadata and scores so engineers can verify that the answer came from the expected document and page.

## Latest Trace Endpoint

For local debugging, DocuMind exposes:

```text
GET /api/traces/latest?limit=20
```

This endpoint reads recent JSONL records and returns them through the API.

This is a local MVP debugging tool. It should not be exposed in production without authentication, authorization, redaction, pagination, and audit controls.

## Evaluation Dataset Design

Evaluation cases live in:

```text
eval/documind_eval_cases.json
```

Each case includes:

- `id`
- `question`
- `expected_behavior`
- `expected_keywords`
- `expected_source_file`
- `expected_page_number`
- `notes`

Supported expected behaviors:

```text
answer_with_sources
low_confidence_refusal
```

Example:

```json
{
  "id": "case_002_unrelated_refusal",
  "question": "What is the CEO's birthday on Mars?",
  "expected_behavior": "low_confidence_refusal",
  "expected_keywords": [],
  "expected_source_file": null,
  "expected_page_number": null,
  "notes": "Unrelated question should be refused by the confidence gate."
}
```

## Regression Test Logic

The runner lives at:

```text
eval/run_eval.py
```

It:

1. loads evaluation cases
2. imports `RAGService` through `get_rag_service()`
3. calls `rag_service.ask(question)` for each case
4. checks behavior expectations
5. writes results to `eval/eval_results_latest.json`

For `answer_with_sources`, it checks:

- status indicates answered / success
- sources are not empty
- expected keywords appear when provided
- expected source file matches when provided
- expected page number matches when provided

For `low_confidence_refusal`, it checks:

- status indicates low confidence / refused / insufficient evidence
- confidence gate did not unexpectedly pass when that signal is available

## How Evaluation Prevents Regressions

RAG behavior can change when engineers adjust:

- chunk size
- chunk overlap
- embedding model
- BM25 weight
- embedding similarity weight
- retrieval top_k
- reranker
- confidence threshold
- prompt
- LLM model
- context construction

The eval runner catches regressions such as:

- a grounded question no longer returns `answered`
- sources disappear
- source file changes unexpectedly
- page citation changes unexpectedly
- expected keywords disappear
- unrelated questions are no longer refused
- threshold tuning blocks valid questions
- reranking pushes the correct evidence out of the top results

## Current Limitations

- JSONL logging is a lightweight MVP observability solution.
- The trace endpoint is intended for local debugging only.
- The eval dataset currently contains placeholder and seed cases.
- Evaluation is rule-based and does not measure semantic quality deeply.
- Future upgrades can include RAGAS, LLM-as-judge, CI integration, dashboards, and domain-expert review workflows.
