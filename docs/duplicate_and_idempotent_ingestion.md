# Duplicate Detection and Idempotent Ingestion

## Definition and system position

Idempotent ingestion means that submitting the same file repeatedly produces
the same stored result without repeated parsing, embedding, or duplicate chunk
records. Duplicate detection runs after reading the uploaded bytes but before
PDF parsing:

```text
PDF bytes -> SHA-256 -> identity lookup -> decision
                                      |-> duplicate: return existing identity
                                      |-> conflict: return HTTP 409
                                      `-> indexed: parse, chunk, embed, upsert
```

## Why duplicate documents are harmful

Duplicate chunks consume embedding API cost and storage, distort BM25 corpus
statistics, reduce top-k evidence diversity, and can make confidence appear
stronger merely because the same evidence occurs repeatedly. Silently
overwriting a different file under an existing version also breaks auditability.

## Identity model

- `file_hash`: SHA-256 of the original PDF bytes; identifies exact binary
  content.
- `document_id`: stable logical document identity, such as `credit-policy`.
- `version`: a version label within that logical document.
- `chunk_id`: deterministic storage identity using
  `document_id:version:chunk_index`.

`file_hash:chunk_index` would make chunks content-addressed, but a different
file uploaded under the same logical version could silently coexist. Keeping
the business-oriented chunk ID and checking the hash before upsert makes a
version conflict explicit.

## SHA-256 and metadata

The upload API computes:

```python
file_hash = hashlib.sha256(content).hexdigest()
```

The input is the uploaded binary content, not extracted text. The hash is
copied to every chunk because the current architecture has no central document
registry. Chroma and the in-memory adapter expose common lookups by file hash
and by document ID plus version; upper layers do not use Chroma-specific APIs.

## Decision policies

### Exact duplicate

If the hash already exists, the API returns HTTP 200 with
`ingestion_result="duplicate"` and the existing document identity. It does not
write a temporary PDF, parse, chunk, call the embedding model, or add records.

### New version

The same logical `document_id` with a different `version` and different hash is
allowed. Both versions remain stored, and the existing ACTIVE/ARCHIVED lifecycle
rules determine normal retrieval eligibility. Uploading a new version does not
automatically archive an old one.

### Version conflict

If `document_id` and `version` already exist but the incoming hash is different,
the API returns HTTP 409 with `ingestion_result="version_conflict"`. Existing
chunks remain unchanged and no embedding is generated.

Exact-hash lookup runs first. Therefore the same bytes requested under a new ID
or version are still classified as a duplicate of the existing document.

## Why upsert was insufficient

Upsert only acts when storage is reached. Previously, the application had
already parsed the PDF and regenerated every embedding. With a random document
ID, repeated uploads also generated different chunk IDs and created duplicate
records. With the same document ID and version, upsert avoided a count increase
but silently replaced content and could leave stale trailing chunks when the
new file had fewer chunks.

## Important code paths

- `api/documents.py`: hashes bytes and returns early for duplicate/conflict.
- `services/rag.py`: classifies ingestion without knowing the storage backend.
- `services/retrieval_service.py`: forwards common identity lookups.
- `services/chroma_vector_store.py`: uses metadata `where` lookups.
- `services/vector_db.py`: preserves the same metadata in memory.
- `services/chunker.py`: propagates `file_hash` to every chunk.

## Compatibility and failure modes

Legacy records without `file_hash` remain retrievable but cannot be found by
exact-hash lookup. If their document ID and version are supplied again, the
system treats the unknown old hash as a version conflict rather than overwriting
it. Without a supplied stable document ID, one legacy duplicate may still be
indexed.

The metadata check and vector upsert are not transactional. Two concurrent
identical requests can both pass the precheck and both generate embeddings,
although deterministic chunk IDs prevent duplicate final records. A future SQL
registry should enforce unique constraints on `file_hash` and
`(document_id, version)`.

SHA-256 detects exact bytes only. Re-saving a visually identical PDF can change
its hash. PDF canonicalization and semantic near-duplicate detection remain
future work and should normally produce review candidates rather than automatic
deletion.

## Project evolution

```text
Prototype: random IDs and unconditional embedding
Local MVP: SHA-256 precheck and explicit duplicate/conflict results
Deployment-ready: ingestion jobs, locks, retries, and status APIs
Enterprise: SQL document registry, unique constraints, event history,
            transaction/outbox, tenant ownership, and review workflows
```

## Verification results

Verified on 2026-08-04:

- Python compilation passed;
- backend tests passed: 16/16;
- uploading the same fixture twice kept the chunk count unchanged and did not
  increase the fake embedder call count;
- version conflict returned HTTP 409 before PDF parsing;
- Chroma metadata lookup continued to identify the duplicate after store
  restart;
- score-semantics checks passed: 3/3;
- dependency integrity reported no broken requirements;
- the full evaluation retained its previous 12/15 baseline, with source hits
  11/11, page hits 11/11, and fallback correctness 4/4.

The existing two confidence failures and one ranking failure remain. No
retrieval weights, confidence thresholds, or evaluation expectations changed.

## Transferable skills

The must-master concepts are content identity versus business identity,
pre-side-effect validation, deterministic keys, idempotency boundaries, and
backward-compatible metadata evolution. They apply to RAG ingestion, agent tool
execution, document OCR pipelines, payment requests, and event consumers.

## Interview explanation

Chinese:

> 我在文档解析和 embedding 前对原始 PDF bytes 计算 SHA-256，并通过统一的
> VectorStore metadata lookup 区分精确重复、新版本和版本冲突。精确重复会返回
> 已有文档身份且不重复生成 embedding；相同 document_id 和 version 的不同内容
> 返回 409，避免 upsert 静默覆盖。该设计保留了 Chroma 与内存实现的一致性。

English:

> I implemented SHA-256 duplicate detection before PDF parsing and embedding.
> Storage-agnostic metadata lookups classify exact duplicates, valid new
> versions, and conflicting content for an existing document version. Duplicate
> requests reuse the persisted identity, while version conflicts return HTTP
> 409 without overwriting indexed data.

## Truthful resume wording

> Added pre-embedding SHA-256 duplicate detection and version-conflict handling
> to a FastAPI/Chroma RAG ingestion pipeline, preventing repeated embedding cost
> and silent overwrites while preserving lifecycle and retrieval behaviour.
