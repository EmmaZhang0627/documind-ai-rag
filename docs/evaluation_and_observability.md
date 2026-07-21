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
- `expected_status`
- `expected_fallback_reason`
- `expected_keywords`
- `expected_evidence_keywords`
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
  "expected_evidence_keywords": [],
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
5. computes explicit RAG quality metrics
6. writes results to `eval/eval_results_latest.json`

For `answer_with_sources`, it checks:

- status indicates answered / success
- sources are not empty
- expected keywords appear when provided
- expected evidence keywords appear in cited source snippets when provided
- expected source file matches when provided
- expected page number matches when provided

For `low_confidence_refusal`, it checks:

- status indicates low confidence / refused / insufficient evidence
- confidence gate did not unexpectedly pass when that signal is available

Responsible fallback cases also check expected fallback statuses such as
`human_review_required` and `sensitive_input_detected`, plus
`fallback_reason` when the case provides one.

Phase 4 introduced these checks as pass/fail regression guards. Phase 5 keeps
that behavior and adds named metrics so failures are easier to diagnose and
aggregate over time.

## Evaluation Metrics

Each case result contains a `metrics` object:

- `retrieval_hit`: true when the expected `source_file` and/or `page_number`
  appears in returned sources. It is `not_applicable` when a case has no
  expected source fields.
- `source_accuracy`: true when the final returned sources match the expected
  `source_file` and/or `page_number` fields that are provided. It is
  `not_applicable` when no expected source fields are configured.
- `keyword_coverage`: compares `expected_keywords` against the answer text and
  reports `matched_keywords`, `missing_keywords`, and
  `keyword_coverage_ratio`. It is `not_applicable` when the expected keyword
  list is empty.
- `refusal_accuracy`: true when `low_confidence_refusal` cases return a
  low-confidence/refusal/insufficient-evidence status, and when
  `answer_with_sources` cases return an answered/success status.
- `citation_presence`: true when `answer_with_sources` cases return at least
  one source. Low-confidence refusal cases do not require citations, so this
  metric is `not_applicable` for those cases.
- `citation_correctness`: checks whether expected evidence keywords appear in
  the combined cited source snippets. It reports
  `expected_evidence_keywords`, `matched_evidence_keywords`,
  `missing_evidence_keywords`, `evidence_coverage_ratio`, and
  `citation_correctness_passed`. It is `not_applicable` for refusal cases or
  cases without expected evidence keywords.

Citation presence alone is not enough because a response can include a citation
that points to the wrong chunk or to a weakly related page. That is false
grounding: the answer looks grounded because it has a source, but the cited text
does not actually support the claim. DocuMind keeps this check lightweight by
returning a short `source_snippet` with each cited source and matching expected
evidence phrases against those snippets.

Example source object:

```json
{
  "document_id": "policy.pdf",
  "source_file": "policy.pdf",
  "page_number": 2,
  "chunk_index": 4,
  "source_snippet": "Manager approval is required before refunds can be processed. The approval window is 3 days."
}
```

This is still a rule-based keyword check. It can miss paraphrases, synonyms, and
subtle contradictions. Future versions can add LLM-as-judge, RAGAS
faithfulness, or domain-expert review for deeper semantic evaluation.

The output JSON has this shape:

```json
{
  "generated_at": "2026-07-16T00:00:00+00:00",
  "summary": {
    "total_cases": 2,
    "passed_cases": 1,
    "failed_cases": 1,
    "pass_rate": 0.5,
    "retrieval_hit_rate": 1.0,
    "source_accuracy_rate": 1.0,
    "average_keyword_coverage": 0.75,
    "refusal_accuracy_rate": 1.0,
    "citation_presence_rate": 1.0,
    "citation_correctness_rate": 1.0
  },
  "results": [
    {
      "case_id": "case_001_grounded_placeholder",
      "question": "...",
      "expected_behavior": "answer_with_sources",
      "actual_status": "answered",
      "passed": true,
      "failed_checks": [],
      "trace_id": "...",
      "metrics": {
        "retrieval_hit": true,
        "source_accuracy": true,
        "keyword_coverage": {
          "status": "applicable",
          "matched_keywords": ["keyword"],
          "missing_keywords": [],
          "keyword_coverage_ratio": 1.0
        },
        "refusal_accuracy": true,
        "citation_presence": true,
        "citation_correctness": {
          "expected_evidence_keywords": ["manager approval", "3 days"],
          "matched_evidence_keywords": ["manager approval", "3 days"],
          "missing_evidence_keywords": [],
          "evidence_coverage_ratio": 1.0,
          "citation_correctness_passed": true
        }
      },
      "top_sources": []
    }
  ]
}
```

For aggregate rates, `not_applicable` cases are excluded from the denominator.
If every case is `not_applicable` for a metric, the summary value is `null`.

## Running Evaluation

Run from the project root:

```bash
python eval/run_eval.py
```

The latest result is written to:

```text
eval/eval_results_latest.json
```

If the command exits with a non-zero status, open the JSON and inspect
`failed_checks`, `metrics`, and `trace_id`.

Responsible fallback behavior is documented in:

```text
docs/responsible_fallback.md
```

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

The metrics make those failures more specific:

- retrieval regressions lower `retrieval_hit_rate`
- citation regressions lower `citation_presence_rate` or
  `source_accuracy_rate`
- false-grounding regressions lower `citation_correctness_rate`
- answer-content regressions lower `average_keyword_coverage`
- refusal or confidence-threshold regressions lower `refusal_accuracy_rate`

## Current Limitations

- JSONL logging is a lightweight MVP observability solution.
- The trace endpoint is intended for local debugging only.
- The eval dataset currently contains placeholder and seed cases.
- Evaluation is rule-based and does not measure semantic quality deeply.
- Future upgrades can include RAGAS, LLM-as-judge, CI integration, dashboards, and domain-expert review workflows.
