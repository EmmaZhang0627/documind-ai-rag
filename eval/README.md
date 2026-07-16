# DocuMind RAG Evaluation

This folder contains a lightweight local regression runner for the RAG pipeline.

It does not use RAGAS, a database, or CI. It imports the existing `RAGService`
through the dependency provider and runs a small list of questions against it.

## Run

```bash
python eval/run_eval.py
```

The runner writes the latest result to:

```text
eval/eval_results_latest.json
```

## Add A Case

Edit `eval/documind_eval_cases.json` and add a new object:

```json
{
  "id": "case_003_policy_question",
  "question": "What does the document say about the refund policy?",
  "expected_behavior": "answer_with_sources",
  "expected_keywords": ["refund", "policy"],
  "expected_source_file": "policy.pdf",
  "expected_page_number": 2,
  "notes": "Grounded question from the policy document."
}
```

Use `answer_with_sources` when the question should be answered from the indexed
documents. Use `low_confidence_refusal` when the system should refuse because
the documents do not contain enough evidence.

## Interpret Results

Each result includes:

- `case_id`
- `expected_behavior`
- `actual_status`
- `passed`
- `failed_checks`
- `trace_id`
- `top_sources`

A regression failure means a case that used to pass now fails. Common examples:

- A grounded question no longer returns `answered`.
- A grounded question returns no sources.
- Expected keywords disappear from the answer.
- Expected `source_file` or `page_number` is no longer cited.
- An unrelated question no longer returns `low_confidence`.

The `trace_id` can be used with `logs/rag_traces.jsonl` or
`GET /api/traces/latest` to inspect retrieval, rerank, confidence, and LLM
behavior for that case.
