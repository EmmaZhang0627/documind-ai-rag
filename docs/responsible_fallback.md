# Responsible Fallback

DocuMind should not always generate an answer. A RAG system can retrieve weak
evidence, miss indexed evidence, receive sensitive input, or be asked to make a
business decision that should belong to a human reviewer. Responsible fallback
keeps these cases explicit instead of forcing the LLM to answer.

## Response Schema

Chat responses keep the existing fields and add `fallback_reason`:

```json
{
  "trace_id": "...",
  "question": "...",
  "answer": "...",
  "sources": [],
  "trace": {},
  "status": "low_confidence",
  "fallback_reason": "low_confidence"
}
```

Supported statuses:

- `answered`
- `low_confidence`
- `insufficient_evidence`
- `conflicting_sources`
- `out_of_scope`
- `sensitive_input_detected`
- `human_review_required`
- `error`

## Fallback Decision Layer

`RAGService.ask()` now has a rule-based decision layer before LLM generation.
Sensitive input and out-of-scope final-decision requests are checked before
embedding so obvious secrets, personal identifiers, or approval decisions are
not sent to external model calls. Evidence-quality and source-conflict fallback
decisions run after retrieval and reranking, when scores and source metadata are
available.

## Low Confidence

If the top retrieved score is below the confidence threshold, DocuMind skips the
LLM call and returns:

```json
{
  "status": "low_confidence",
  "fallback_reason": "low_confidence",
  "sources": []
}
```

This protects against hallucination when the indexed documents do not provide
strong enough evidence.

## Insufficient Evidence

If retrieval returns candidates but the selected chunks do not contain usable
evidence text, DocuMind returns:

```json
{
  "status": "insufficient_evidence",
  "fallback_reason": "insufficient_evidence"
}
```

This is separate from low confidence: candidates exist, but they cannot support
a grounded answer.

## Out-Of-Scope Decision Requests

Simple pattern checks catch requests that ask the assistant to make final
business, risk, approval, loan, or compliance decisions. Examples include
phrases such as `approve`, `reject`, `final decision`, `loan approval`, and
`should we approve`.

When detected, DocuMind returns:

```json
{
  "status": "human_review_required",
  "fallback_reason": "out_of_scope_decision_request"
}
```

The assistant may retrieve relevant guidance, but it must not make the final
decision.

## Sensitive Input

Rule-based checks detect obvious sensitive inputs such as ID-card-like numbers,
phone-number-like strings, bank-account-like numbers, API keys, tokens, or
secrets. When detected, DocuMind skips embedding and LLM calls and asks the user
to redact the input.

```json
{
  "status": "sensitive_input_detected",
  "fallback_reason": "sensitive_input_detected",
  "sources": []
}
```

## Conflicting Sources

The current chunk metadata does not include rich version or lifecycle fields.
DocuMind includes a conservative placeholder that checks only fields when they
exist, such as `version` or `document_status`. It does not invent metadata. If
future ingestion adds these fields, the same decision layer can flag deprecated,
expired, or conflicting active sources.

## Trace Fields

JSONL trace records now include:

- `fallback_reason`
- `fallback_status`
- `llm_called`
- `sensitive_input_detected`
- `out_of_scope_detected`
- `conflict_detected`

These fields make it clear whether the LLM was skipped because of low
confidence, insufficient evidence, sensitive input, out-of-scope requests, or
source conflicts.

## Limitations

This is a lightweight local-MVP implementation. It uses simple patterns and
metadata checks, so it can miss subtle sensitive data, nuanced policy decisions,
or semantic conflicts. Future upgrades can add dedicated redaction, moderation,
policy rules, richer document metadata, RAGAS faithfulness checks, or
LLM-as-judge review.
