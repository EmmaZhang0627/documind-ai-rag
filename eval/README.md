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
  "expected_evidence_keywords": ["refund policy", "manager approval"],
  "expected_source_file": "policy.pdf",
  "expected_page_number": 2,
  "notes": "Grounded question from the policy document."
}
```

Use `answer_with_sources` when the question should be answered from the indexed
documents. Use `low_confidence_refusal` when the system should refuse because
the documents do not contain enough evidence.

## Interpret Results

The output file contains:

- `summary`
- `results`

`summary` includes:

- `total_cases`
- `passed_cases`
- `failed_cases`
- `pass_rate`
- `retrieval_hit_rate`
- `source_accuracy_rate`
- `average_keyword_coverage`
- `refusal_accuracy_rate`
- `citation_presence_rate`
- `citation_correctness_rate`

Each item in `results` includes:

- `case_id`
- `question`
- `expected_behavior`
- `actual_status`
- `passed`
- `failed_checks`
- `trace_id`
- `metrics`
- `top_sources`

The `metrics` object includes:

- `retrieval_hit`: whether the expected source file and/or page appeared in
  returned sources. Cases without expected source fields are `not_applicable`.
- `source_accuracy`: whether the final returned sources match the expected
  source fields when those fields are provided.
- `keyword_coverage`: matched keywords, missing keywords, and coverage ratio.
  Cases with no expected keywords are `not_applicable`.
- `refusal_accuracy`: whether the response status matches the expected behavior.
- `citation_presence`: whether answer-with-sources cases returned citations.
  Refusal cases do not require citations.
- `citation_correctness`: whether `expected_evidence_keywords` appear in the
  returned source snippets. This catches false grounding, where a response has a
  citation but the cited text does not actually support the answer.

`not_applicable` metric values are excluded from aggregate denominators.

A regression failure means a case that used to pass now fails. Common examples:

- A grounded question no longer returns `answered`.
- A grounded question returns no sources.
- Expected keywords disappear from the answer.
- Expected `source_file` or `page_number` is no longer cited.
- Expected evidence keywords are missing from returned `source_snippet` values.
- An unrelated question no longer returns `low_confidence`.

Use `failed_checks` as the quick failure label, then use `metrics` to see which
quality dimension moved. For example, `retrieval_hit=false` points to retrieval
or indexing, while `citation_presence=false` means the final answer did not
return citations. If `citation_correctness=false`, inspect `top_sources` and
check whether the short snippets actually support the expected answer.

The `trace_id` can be used with `logs/rag_traces.jsonl` or
`GET /api/traces/latest` to inspect retrieval, rerank, confidence, and LLM
behavior for that case.

Citation correctness is intentionally lightweight. It uses keyword or phrase
matching against cited snippets rather than RAGAS or an LLM judge, so it is
cheap and local but will not catch every semantic mismatch.
