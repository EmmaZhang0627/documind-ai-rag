# DocuMind RAG Evaluation

This folder contains a lightweight local regression runner for the DocuMind RAG
pipeline. It does not use RAGAS, pytest, LangChain, LangGraph, a database, or
CI.

## Why The Eval Ingests A Fixture First

DocuMind currently stores indexed chunks, embeddings, and BM25 state in
process-local memory. A document uploaded through the running FastAPI backend is
therefore visible only to that backend process.

`eval/run_eval.py` runs as a separate Python process, so it must build its own
retrieval state before asking questions. The runner now:

1. clears only the current eval process's in-memory vector store;
2. loads the fixture PDF from `eval/fixtures`;
3. extracts text page by page using the same behavior as `POST /api/documents/parse-pdf`;
4. chunks pages with `split_pages_into_chunks()`;
5. ingests chunks through `RAGService.ingest_document()`;
6. runs the cases with `RAGService.ask()`;
7. writes detailed JSON results to `eval/eval_results_latest.json`.

It does not delete uploaded PDFs, logs, environment files, or user data.

## Fixture PDF

Put exactly one stable evaluation PDF in:

```text
eval/fixtures/
```

The runner discovers the actual `.pdf` filename from that directory. Do not
reference a made-up filename in cases. If no PDF exists, the runner stops. If
more than one PDF exists, the runner stops so the evaluation remains
repeatable.

Current fixture:

```text
eval/fixtures/Study Plan - MSc Computer Science.pdf
```

## Environment Variables

Grounded evaluation cases use the existing production embedding and answer
generation path, so the same backend environment is required:

```text
OPENAI_API_KEY
```

Optional variables are read by the normal backend settings:

```text
EMBEDDING_MODEL_NAME
CHAT_MODEL_NAME
OPENAI_TIMEOUT_SECONDS
RETRIEVAL_TOP_K_DEFAULT
ANSWER_TOP_K_DEFAULT
CONFIDENCE_THRESHOLD
EMBEDDING_SCORE_WEIGHT
BM25_SCORE_WEIGHT
RERANKER_ENABLED
RERANKER_MODEL_NAME
```

Do not commit or print real API key values.

## Run

From the project root, prefer the backend virtual environment:

```powershell
backend\.venv\Scripts\python.exe eval\run_eval.py
```

The runner writes:

```text
eval/eval_results_latest.json
```

If setup fails before cases can run, for example because the fixture is missing
or the OpenAI embedding call fails, the runner still writes a structured result
with `setup_error`, failed case records, and redacted error text.

It also prints a concise terminal summary:

- total cases
- passed cases
- failed cases
- source hit count
- page hit count
- fallback correctness count

## Case Schema

Grounded cases should use this simple format:

```json
{
  "id": "case_001_study_plan_duration",
  "question": "What is the indicative study duration for the MSc Computer Science programme?",
  "expected_status": "answered",
  "expected_source_file": "Study Plan - MSc Computer Science.pdf",
  "expected_page_numbers": [1],
  "expected_evidence_keywords": ["Indicative Study Duration", "24 Months"],
  "expected_answer_keywords": ["24", "months"],
  "top_k": 3,
  "notes": "Grounded case from the evaluation fixture PDF."
}
```

Fallback cases may omit source, page, evidence, and answer expectations:

```json
{
  "id": "case_003_unrelated_refusal",
  "question": "What is the CEO's birthday on Mars?",
  "expected_status": "low_confidence",
  "expected_fallback_reason": "low_confidence",
  "top_k": 3,
  "notes": "Unrelated question should be refused by the confidence gate."
}
```

## Checks

Each case result records:

- `actual_status`: status returned by `RAGService.ask()`.
- `fallback_reason`: fallback reason returned by the RAG service, if any.
- `returned_sources`: source metadata returned by the RAG service.
- `retrieved_page_numbers`: unique page numbers from returned sources.
- `checks.status`: whether `actual_status` equals `expected_status`.
- `checks.fallback`: whether fallback status and expected fallback reason match.
- `checks.source_match`: whether the expected PDF filename appears in sources.
- `checks.page_match`: whether all expected page numbers appear in sources.
- `checks.evidence_keywords`: whether expected evidence keywords appear in returned source snippets.
- `checks.answer_keywords`: whether expected answer keywords appear in the LLM answer.
- `passed`: true only when all applicable checks pass.
- `trace_id`: trace ID from the RAG response.
- `retrieval_trace`: in-response trace with retrieval, rerank, and decision data.
  If setup failed before retrieval, this is empty.

Keyword matching is case-insensitive. The eval does not require exact LLM answer
string equality.

## Interpreting Failures

Use `failed_checks` as the quick label:

- `status`: response status changed.
- `fallback`: refusal or fallback behavior changed.
- `source_match`: the expected fixture file was not cited.
- `page_match`: the expected page was not cited.
- `evidence_keywords`: returned source snippets no longer contain expected supporting evidence.
- `answer_keywords`: the answer no longer contains expected answer terms.

These checks are intentionally lightweight. They are useful for a repeatable
baseline, but they do not prove full semantic correctness.
