# Query Rewrite

## Purpose

Query rewrite converts a natural user question into one concise retrieval query. It aims to preserve intent while removing conversational phrasing that can weaken embedding, BM25, or reranker matching. It does not answer the question.

```text
original query
→ optional rewrite
→ retrieval embedding / Hybrid retrieval / CrossEncoder
→ confidence
→ final answer generated from the original query
```

## Selected MVP policy

DocuMind uses a small conditional policy rather than rewriting every request. Rewrite is attempted when the query contains a vague-reference marker or is a question with at least nine words. Short clear queries are left unchanged.

The current `/api/chat` request contains no conversation history. The MVP can clarify a self-contained natural-language question, but it cannot reliably resolve a pronoun whose meaning exists only in an earlier turn. In that situation the prompt tells the model to leave the reference unchanged.

## Prompt rules

The rewrite model must:

- return one standalone, concise search query;
- preserve intent and constraints;
- not answer the question;
- not add unsupported facts, dates, numbers, names, or entities;
- return an already-clear query unchanged;
- return only one query without a label or explanation.

The original query remains the input to final answer generation. The rewritten query is used only for embedding, Hybrid/BM25 retrieval, and CrossEncoder reranking.

## Validation and fallback

DocuMind falls back to the normalized original query when:

- rewrite is disabled or the policy skips the query;
- the model call fails or times out;
- output is empty or excessively long;
- output looks like an answer;
- output introduces a number or capitalized entity absent from the original.

These checks reduce obvious drift but cannot prove semantic equivalence. Both `original_query` and `retrieval_query` are recorded in response and observability traces.

## Focused Corpus V2 evaluation

The evaluation used 13 existing grounded cases across semantic paraphrase, contextual, cross-document hard negative, and exact-term categories. It reused current Hybrid Top10 weights and the existing CrossEncoder; ground truth was unchanged.

| Metric | Original | Rewritten |
|---|---:|---:|
| Hit@1 | 38.46% | 53.85% |
| Hit@3 | 61.54% | 69.23% |
| Hit@5 | 61.54% | 69.23% |
| MRR | 0.4982 | 0.6282 |

- Improved: 3 cases.
- Unchanged: 9 cases, including one where both strategies missed Top10.
- Worsened: 1 case.
- Rewrite changed the retrieval query in 12 of 13 cases; one unsafe/invalid result fell back to the original.

Observed benefits were concentrated in semantic paraphrases. Examples include anonymous-complaint wording moving from rank 3 to rank 1 and a modern-AI-method question moving from not found to rank 1.

The explicit regression was `v2_case_022_gai_policy_vs_ethics`: correct evidence moved from rank 7 to outside Top10. This is an intent/term-emphasis regression and must not be hidden by aggregate improvement.

## Alternatives and future work

- **Always rewrite:** simpler policy but adds latency and drift risk to already-good queries.
- **Query expansion:** adds related terms to one query; useful for vocabulary mismatch but can amplify lexical noise.
- **Multi-query retrieval:** retrieves several rewrites and merges results; potentially higher recall with higher embedding, retrieval, reranking, and deduplication cost.
- **HyDE:** embeds a generated hypothetical answer; useful in some semantic tasks but carries stronger hallucination and cost risks.

Multi-query, HyDE, query routing, dynamic weights, conversational memory, and complex classifiers remain future work. The current focused benchmark is too small to claim universal retrieval improvement.
