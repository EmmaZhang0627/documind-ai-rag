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
  - `retrieval_score`
  - `rerank_score`
- `rerank_enabled`
- `confidence_threshold`
- `top1_score`
- `confidence_score`
- `confidence_decision`
- `llm_called`
- `final_status`
- `error_message`

Long document text is not logged. If snippets are logged, they are capped to a short length to reduce noise and avoid exposing too much content.

Score semantics are intentionally distinct:

- `retrieval_score` is the weighted hybrid embedding/BM25 score and is preserved
  through reranking.
- `rerank_score` is the raw CrossEncoder score when reranking succeeds; it is
  `null` when reranking is disabled or falls back.
- `confidence_score` exists on the selected top candidate and equals that
  candidate's `retrieval_score`.
- `rerank_enabled` shows whether CrossEncoder scores determined ordering.
- legacy `final_score` and `top1_score` fields are retained in JSONL logs for
  compatibility and now consistently alias the hybrid retrieval-based confidence
  score.

Reranking therefore controls ordering, while confidence gating uses the selected
candidate's hybrid retrieval score. Raw CrossEncoder scores are not normalized or
treated as probabilities. They will not be used for confidence until calibration
is backed by sufficient evaluation data.

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

For grounded answer cases, the runner checks:

- status indicates answered / success
- sources are not empty
- expected keywords appear when provided
- expected evidence keywords appear in full internal retrieved chunks
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
does not actually support the claim.

`source_snippet` is a presentation field, not a complete evidence record. It is
intentionally truncated to keep API responses compact and to reduce document
exposure. Evaluation must not treat absence from this preview as absence from the
retrieved chunk.

The self-contained runner supplies an evaluation-only in-process candidate sink
to `RAGService.ask`. The same retrieval and reranking pass used for the answer
places ranked candidates in that local sink. Full text is used only for matching
and is not added to `/api/chat`, public source objects, or the evaluation result
JSON. Results retain candidate metadata, scores, matched keywords, and rank so
failures remain explainable without publishing complete document text.

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
subtle contradictions. Future versions can add calibrated semantic evaluation
without changing the public source response.

## Layered Failure Evaluation

An end-to-end failure does not necessarily mean retrieval failed. Each grounded
case records independent layers:

- `source_match`: expected source exists in the internal candidate set.
- `page_match`: expected page exists in the internal candidate set.
- `full_evidence_match`: expected evidence exists in full candidate text.
- `retrieval_pass`: source, page, and full evidence checks all pass.
- `evidence_rank`: first ranked candidate containing all expected evidence
  keywords; `null` means no single retrieved candidate contains them.
- `confidence_pass`: returned answer/fallback behavior matches the case after
  confidence gating.
- `generation_pass`: when an answer is generated, its expected answer keywords
  are present.
- `overall_pass`: retrieval, expected confidence behavior, and applicable
  generation checks pass. `evidence_rank` remains visible as ranking quality
  information; a lower-ranked evidence chunk does not fail an otherwise correct
  top-k answer by itself.

Failure stages are assigned in causal order:

```text
evidence absent from candidate set       -> retrieval_failure
evidence present below rank 1            -> ranking_failure
rank 1 evidence rejected by gate         -> confidence_failure
gate passes but answer keywords are wrong -> generation_failure
system is correct but the check is wrong -> expectation_or_evaluation_failure
evaluation cannot be prepared or run     -> setup_error
```

For example, `evidence_rank = 3` means retrieval found the supporting chunk, but
two other candidates were ranked above it. If that ordering causes the system to
reject or answer incorrectly, it is a ranking failure rather than a retrieval
miss. If the top-k context still produces the expected answer, the case can pass
while retaining rank 3 as a comparison signal. A low-confidence response with
`evidence_rank = 1` is a confidence failure because the correct evidence was
already selected.

## Retrieval Strategy Comparison

`eval/run_retrieval_comparison.py` evaluates retrieval and ranking separately
from confidence gating and answer generation. It uses the same fixed fixture,
production page extraction, chunking, embedding service, in-memory ingestion,
hybrid formula, and CrossEncoder as the end-to-end runner, but it does not call
the confidence gate or LLM.

The three evaluation-only strategies are:

```text
vector_only
  order the candidate pool by embedding_score

hybrid
  order by:
  embedding_score_weight * embedding_score
  + bm25_score_weight * normalized_bm25_score

hybrid_rerank
  start with the hybrid candidate pool
  then order with the configured CrossEncoder rerank_score
```

No retrieval mode is added to the public API, and production defaults are not
changed.

### Retrieval Metrics

- **Top-1 Evidence Hit**: the first candidate contains all expected evidence
  keywords.
- **Hit@K**: at least one of the first K candidates contains the expected
  evidence. For example, Hit@3 means the evidence is available to a top-3
  context builder even if it is not ranked first.
- **Reciprocal Rank**: `1 / evidence_rank`. Rank 1 contributes `1.0`, rank 2
  contributes `0.5`, rank 3 contributes about `0.333`, and missing evidence
  contributes `0`.
- **MRR**: the mean reciprocal rank across all grounded cases. It rewards
  strategies that place correct evidence nearer the top.
- **Average Evidence Rank When Found**: average rank over cases where evidence
  was found; missing cases are reported separately rather than silently removed.

Source-file and page-number hit rates are weak comparison metrics for the current
fixture because it contains only one document and one page. A strategy can return
the wrong chunk while still matching both source and page. Full-text evidence
keywords and evidence rank provide the meaningful signal for this baseline.

### Reranking Effect

For each case, the runner compares hybrid evidence rank with CrossEncoder
reranked evidence rank:

```text
reranked rank is lower  -> improved
same rank               -> unchanged
reranked rank is higher -> worsened
evidence absent         -> evidence_not_found
```

This comparison reveals both reranking gains and regressions. A higher aggregate
MRR does not guarantee every question improved, so worsened cases must be
inspected individually.

The retrieval comparison and end-to-end evaluation answer different questions:

```text
retrieval comparison:
  Did a strategy find and rank the correct evidence?

end-to-end evaluation:
  After retrieval, did confidence, context construction, generation,
  citations, and fallback behavior produce the expected user outcome?
```

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

## OCR Fallback And Table-Aware Ingestion Evaluation

### Why this belongs before retrieval

Retrieval can only rank text that ingestion recovered. If a scanned page
contains pixels but no useful PDF text layer, missing evidence is an
`ingestion_failure`, not a retrieval failure. Likewise, if fixed-character
chunking separates table headers from their rows, retrieval may return text
whose values no longer have a reliable column meaning.

The evaluation-only document-intelligence path is:

```text
PDF page
-> native PyMuPDF extraction
-> selective page-level OCR fallback
-> optional structured table extraction
-> page-aware chunks
-> existing embedding and storage
```

OCR and table processing are feature-gated and disabled by default, preserving
the existing Corpus V2 text benchmark and production-compatible flat chunk
boundaries.

### OCR decision and fallback

The current deterministic OCR trigger is:

```text
native non-whitespace characters < 200
AND
(estimated image coverage >= 0.5 OR image object count >= 3)
```

PyMuPDF's `page.get_textpage_ocr()` invokes local Tesseract using bundled
English language data. OCR output replaces native text only when it is
materially larger. Empty, invalid, or failed OCR safely retains native text so
one problematic page does not crash the complete document.

Important PyMuPDF relationships:

- `fitz.Document` represents the PDF;
- `fitz.Page` represents one page and owns `get_image_info()` and
  `get_textpage_ocr()`;
- `fitz.Rect` represents a rectangular page region;
- `fitz.Rect(image_bbox) & page.rect` computes the image/page intersection,
  preventing out-of-page coordinates from inflating coverage.

### Table representation

Detected tables are serialized into deterministic retrieval text containing a
caption or stable fallback label, a header, and rows. If a large table is split,
every resulting table chunk repeats its header. This keeps values connected to
their column meanings when a chunk is retrieved independently.

Table chunks preserve `content_type=table`, `table_index`, `table_caption`,
`page_number`, `source_file`, `document_id`, and `extraction_method` metadata.
They coexist with ordinary text chunks in the current MVP, improving structural
evidence availability but also creating duplicate evidence and ranking
competition.

### Focused verification

The reserved OCR document produced:

- 4 pages;
- 542 raw characters from ordinary PyMuPDF extraction;
- 11,321 characters after local OCR;
- 4 improved pages;
- 4 native fragments versus 18 OCR chunks;
- representative visible-text evidence ranks of 1, 3, and 1 after previously
  being evidence-not-found.

For the two table papers, 20 useful tables were detected in total. All six
focused table cases preserved the expected header/row evidence in structured
chunks. Evaluation-only BM25 ranking improved one case, left four unchanged,
and worsened one nearby-row case. Structural preservation therefore passed, but
the experiment does not support claiming general ranking improvement.

### Parent-Child compatibility limitation

Current chunk construction is mutually exclusive:

```text
Parent-Child enabled -> Parent-Child chunks
else table enabled   -> flat text plus structured table chunks
else                 -> existing flat chunks
```

If both flags are enabled, Parent-Child takes precedence and structured table
chunks are not created. A proper combined version needs typed text/table
children connected to deterministic parents. Simply appending both outputs
would risk duplicate evidence, conflicting identities, repeated LLM context,
and one fact occupying several TopK positions.

### Diagnosis and future adjustments

- inaccurate OCR triggering: add text-quality signals or maximum single-image
  coverage before considering exact geometric image union;
- multilingual scans: configure OCR languages per manifest/document;
- OCR noise: retain raw text and apply conservative normalization without
  guessing factual corrections;
- complex or cross-page tables: preserve structured cells and join pages only
  when caption, column geometry, and boundary signals agree;
- table candidate competition: remove duplicated flat table regions or
  deduplicate candidates by table identity;
- oversized parent context: reconstruct a window around matched children and
  enforce a context budget.

These improvements should be introduced only after failures are classified as
extraction, chunking, retrieval, ranking, or context-composition problems.
Changing several stages together would hide the cause of regressions.

## Current Limitations

- JSONL logging is a lightweight MVP observability solution.
- The trace endpoint is intended for local debugging only.
- The eval dataset currently contains placeholder and seed cases.
- Evaluation is rule-based and does not measure semantic quality deeply.
- Future upgrades can include RAGAS, LLM-as-judge, CI integration, dashboards, and domain-expert review workflows.
- OCR currently uses English language data and heuristic image/text signals.
- Table detection is strongest for practical ruled tables; merged cells,
  borderless tables, charts, and cross-page structures remain limited.
- Parent-Child and structured table chunk construction are not yet composable.
