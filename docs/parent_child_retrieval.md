# Parent-Child Retrieval

## Purpose and architecture

Parent-Child Retrieval separates retrieval precision from generation context size.

```text
PDF page
→ parent chunks
→ smaller overlapping child chunks
→ embed/store/retrieve/rerank children
→ select top children
→ resolve parent_id
→ deduplicate parents
→ send parent text to generation
```

Small child chunks can match one precise fact with less unrelated text. Larger parents preserve surrounding definitions, conditions, and nearby evidence needed by the LLM. Retrieval scores, BM25 scores, CrossEncoder scores, and confidence decisions remain child-based.

## Metadata relationship

Each stored child preserves normal source metadata plus:

- `parent_id`: deterministic identity built from document ID, version, page, and parent character range;
- `child_index`: child position within the parent;
- `parent_text`: persisted parent content used for reconstruction;
- `parent_start_char` and `parent_end_char`.

The current local MVP stores parent text in child metadata. This survives Chroma restart without a second collection, but duplicates parent text across sibling records. A larger production design could use a separate parent/document store keyed by `parent_id`.

## Current strategy

- Parent: page-local, non-overlapping windows of 1,600 characters.
- Child: 600 characters with 100-character overlap.
- Child records are the only embedded and indexed records.
- The top five selected children are converted to parents for generation.
- Sibling children with the same `parent_id` produce one context parent.
- `PARENT_CHILD_RETRIEVAL_ENABLED` defaults to false, preserving flat-chunk behavior and existing benchmarks.

Ordinary overlapping chunks do not have a structural parent relationship: every overlapping chunk remains an independent retrieval and context unit. Parent-Child explicitly maps several precise retrieval units back to one larger generation unit.

## Focused Corpus V2 result

The isolated parent-child collection contains 214 parents and 538 children across the same seven documents. The comparison used the six existing multi-evidence cases and six direct factual controls with unchanged Hybrid weights, CrossEncoder, and Top10 candidate depth. Query Rewrite was disabled in both branches to isolate chunk/context behavior.

### Multi-evidence cases

| Metric | Flat | Parent-Child |
|---|---:|---:|
| All evidence available in reranked Top10 | 5/6 | 3/6 |
| All evidence in final Top5 context | 5/6 | 2/6 |
| Average context characters | 3,807.5 | 4,568.2 |
| Average token proxy | 952 | 1,142 |

Parent dedup removed 11 duplicate sibling selections across the six cases, but the larger context did not compensate for child retrieval regressions. Cross-page cases still require several independently retrieved parents.

### Single-evidence controls

| Metric | Flat | Parent-Child |
|---|---:|---:|
| All evidence available in reranked Top10 | 6/6 | 4/6 |
| All evidence in final Top5 context | 6/6 | 5/6 |
| Average context characters | 3,739.3 | 5,757.5 |
| Average token proxy | 935 | 1,439 |

The normal-query regression was `v2_case_047_cold_start_cost`, where the required evidence did not reach parent context. Some queries produced almost 8,000 context characters because five selected children mapped to five distinct parents.

Final-answer generation was not run because Corpus V2 currently has evidence identity labels but no deterministic expected-answer labels. Adding an LLM judge would introduce a second model-based variable into a chunk/context experiment.

## Benefits and trade-offs

Potential benefits:

- precise child retrieval with richer generation context;
- sibling-parent deduplication;
- preserved document/page traceability;
- better nearby context when the needed facts share a parent.

Observed and structural risks:

- new child boundaries can reduce retrieval recall;
- larger parent context increases token use and distraction;
- duplicate parent text increases storage size in this MVP;
- page-local parents cannot join evidence from different pages;
- multiple distinct parents can still create oversized context;
- fixed character boundaries remain weak for tables and document structure.

The current result does not support enabling Parent-Child by default. The implementation remains behind a flag for further diagnosis; weights, confidence, reranker, Query Rewrite, APIs, and frontend are unchanged.
