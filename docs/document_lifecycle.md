# Document Metadata and Lifecycle

## Definition and system position

Document lifecycle management controls which persisted document versions may be
used for answers. It sits across the complete ingestion and retrieval path:

```text
PDF upload -> chunk metadata -> vector storage -> active-only retrieval -> answer
```

Persistence answers "does the data survive a restart?" Lifecycle management
answers "is this stored version currently allowed to influence an answer?"

## Why it is needed

A file name is not a sufficient enterprise document identity. Policies and
procedures change over time, while older versions may need to remain available
for audit. Without lifecycle metadata, duplicate uploads create unrelated IDs,
old and new policies can be retrieved together, and withdrawn content can reach
the language model.

## Current metadata

Every new chunk carries and persists:

- `document_id`: stable logical document identity;
- `version`: version label represented as a string;
- `status`: `ACTIVE` or `ARCHIVED`;
- `created_time`: ingestion time in UTC ISO 8601 form;
- `file_name`: original file name.

Existing retrieval metadata remains available: `source_file`, `page_number`,
`chunk_index`, and text. The Chroma record ID is
`document_id:version:chunk_index`, allowing versions of the same logical
document to coexist without overwriting each other.

Legacy records without lifecycle fields remain compatible: a missing status is
treated as `ACTIVE`, a missing version as `"1"`, and an unknown creation time as
`None`.

## Document states

| State | Stored | Normal retrieval | Intended use |
| --- | --- | --- | --- |
| `ACTIVE` | Yes | Yes | Current approved knowledge for this MVP |
| `ARCHIVED` | Yes | No | Historical or audit retention |

Archiving is therefore not deletion. `count()` includes archived chunks, while
normal `search()` results do not.

## State mutation

The MVP supports one explicit transition for an existing document version:

```text
ACTIVE -> ARCHIVED
```

The API operation is:

```text
POST /api/documents/{document_id}/versions/{version}/archive
```

It updates every chunk whose logical document ID and version match, while
preserving embeddings, documents, file hashes, page metadata, and all other
stored fields. Other versions are not changed. Repeating the operation is
idempotent: the version remains ARCHIVED and no records are created or deleted.

After a successful metadata update, the VectorStore rebuilds its active-only
BM25 corpus. This is required because BM25 is derived in-memory state; changing
Chroma metadata alone would leave the old active corpus in memory until restart.

The operation is implemented by both Chroma and the in-memory adapter behind
the same VectorStore interface. A missing document version returns HTTP 404.

Archive is used instead of delete because historical versions may be needed for
audit, incident review, or explaining past decisions. Archive removes retrieval
eligibility without destroying the underlying evidence.

## Version handling

The upload endpoint accepts optional multipart values for `document_id`,
`version`, and `status`. Defaults preserve existing clients: a new UUID,
version `"1"`, and status `ACTIVE` are used when values are omitted.

To represent two versions of one policy, clients should reuse the logical
document ID:

```text
document_id=credit-policy, version=1, status=ARCHIVED
document_id=credit-policy, version=2, status=ACTIVE
```

The MVP does not automatically archive the previous version. The caller is
responsible for assigning the intended states during ingestion.

## Retrieval filtering

Lifecycle filtering is implemented inside each VectorStore adapter so the RAG
business pipeline remains storage-independent. Archived records are excluded
both from the BM25 corpus and from vector candidates before hybrid scoring,
reranking, confidence evaluation, tracing, and LLM generation.

Filtering only after scoring would be incorrect: an archived record could set
the maximum BM25 score used for normalization and lower the normalized scores
of active records, even if it were removed from the final result.

## Important code paths

- `api/documents.py`: validates upload lifecycle fields and creates one document
  creation timestamp.
- `services/chunker.py`: copies document metadata to every chunk without
  changing the chunking algorithm.
- `services/chroma_vector_store.py`: persists metadata, keeps versions distinct,
  and builds/searches an active-only BM25 corpus.
- `services/vector_db.py`: applies the same contract to the fallback in-memory
  implementation.
- `api/chat.py`: exposes lifecycle metadata on returned sources.

## Alternatives and trade-offs

Filtering in `RAGService` was rejected because archived chunks would already
have affected retrieval and ranking. Separate active/archive collections were
not selected because state transitions would require moving records and
maintaining cross-collection consistency. A relational document registry is a
stronger future architecture but would be excessive for this local MVP.

## Common failure modes

- using `document_id:chunk_index` and overwriting an older version;
- treating missing legacy status as inactive and making existing data vanish;
- filtering vector results but leaving archived text in the BM25 corpus;
- uploading two versions as ACTIVE and receiving conflicting evidence;
- generating a new document ID for every version, losing their logical link;
- deleting archived records and losing the audit copy.

## Verification results

Verified on 2026-08-05 with Python 3.11.9 and the project-pinned Chroma
environment:

- Python compilation passed;
- backend unit tests passed: 20/20;
- score-semantics regression checks passed: 3/3;
- dependency integrity check reported no broken requirements;
- persistent lifecycle tests confirmed active retrieval, archived exclusion,
  version coexistence after store restart, and legacy metadata compatibility;
- lifecycle mutation tests confirmed every chunk in the selected version was
  archived, other versions remained ACTIVE, vectors/documents/file hashes were
  preserved, BM25 was rebuilt, repeat archive calls were idempotent, and the
  ARCHIVED state survived Chroma restart;
- the full evaluation retained its previous baseline: 12/15 cases passed,
  source hits 11/11, page hits 11/11, and fallback correctness 4/4.

The three existing evaluation failures remain classified as two confidence
failures and one ranking failure. No thresholds or expectations were changed.

## Project evolution

The next production step is a document registry with an append-only lifecycle
history. Later states can include
`PENDING_REVIEW`, `APPROVED`, `REJECTED`, and `EXPIRED`, together with ownership,
tenant permissions, approval workflow, human escalation, retention policy, and
legal hold. A database transaction or outbox pattern would keep registry state
and vector metadata synchronized.

## Transferable skills

The must-master ideas are stable identity, version-aware idempotency, metadata
propagation, filtering before ranking, and backward-compatible schema evolution.
They apply equally to enterprise RAG, agent memory/tool permissions, and vision
or document-intelligence processing pipelines.

## Interview explanation

Chinese:

> 我为 DocuMind 增加了最小化文档生命周期模型。文档版本使用稳定的
> document_id 和 version 标识，ACTIVE 与 ARCHIVED 数据都持久化，但归档数据
> 会在 VectorStore 边界内、BM25 和 reranking 之前被排除。这样既保留审计数据，
> 又避免过期政策进入回答，同时不改变上层 RAG 流程。

English:

> I introduced an MVP document lifecycle model using a stable document ID,
> version, status, and creation timestamp. Both active and archived versions
> remain persisted, but the VectorStore excludes archived chunks before BM25,
> reranking, confidence checks, and generation. This preserves audit data while
> keeping the RAG pipeline independent of storage-specific filtering.

## Truthful resume wording

> Added version-aware document metadata and active-only retrieval to a FastAPI
> hybrid RAG application, preserving archived Chroma records for audit while
> maintaining backward compatibility with legacy indexed data.

## Remaining limitations

- uploading without a stable `document_id` still creates a new logical document;
- no actor, archive reason, `archived_time`, or append-only transition history;
- only ACTIVE to ARCHIVED is supported; reactivation is intentionally absent;
- no owner, tenant, approval actor, reason, or append-only state history;
- no transaction coordinates document registry state with vector writes;
- locally embedded Chroma remains a single-host persistence design;
- legacy records have no trustworthy historical `created_time`.
