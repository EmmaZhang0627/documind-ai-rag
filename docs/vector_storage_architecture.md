# Vector Storage Architecture

## Purpose and system position

Vector storage sits between embedding generation and retrieval:

```text
PDF -> chunks -> embeddings -> VectorStore
query -> query embedding -> VectorStore -> BM25 -> rerank -> confidence -> answer
```

The `VectorStore` contract keeps RAG business logic independent from storage
infrastructure. `RAGService` and `RetrievalService` operate on chunks and
candidates; the concrete adapter handles database initialization, persistence,
record mapping, and vector score conversion.

## Why process-local memory was insufficient

The prototype stored records in a Python list and rebuilt a BM25 index from a
second text list. This was useful for learning the retrieval pipeline, but:

- all indexed documents disappeared when the process stopped;
- separate workers had separate, potentially inconsistent lists;
- every vector query scanned every embedding in Python;
- duplicate IDs, persistence, recovery, and deletion required custom code;
- memory usage grew with every indexed chunk.

The in-memory adapter remains useful for isolated tests, but it is no longer the
default application backend.

## Current Chroma Persistent architecture

The default backend is `ChromaPersistentVectorStore`, constructed in the
dependency layer and injected through `RetrievalService`.

```text
RAGService
  -> RetrievalService
    -> VectorStore protocol
      -> ChromaPersistentVectorStore
        -> Chroma collection on local persistent disk
```

Each Chroma record contains:

- ID: `{document_id}:{chunk_index}`;
- document: the complete chunk text;
- embedding: the vector already produced by DocuMind's embedding service;
- metadata: `document_id`, `source_file`, `file_name`, `page_number`,
  `chunk_index`, and `text`.

`upsert` makes retries with the same document and chunk ID idempotent. It does
not deduplicate repeated uploads that receive different document IDs.

The collection uses cosine distance. Chroma distances are converted back to
cosine similarity with `1 - distance` before the existing weighted hybrid score
is calculated. BM25 remains unchanged: its in-process index is rebuilt from
documents persisted in Chroma at startup and after mutations.

To preserve the current retrieval algorithm, the adapter requests the complete
collection before applying the existing vector/BM25 weighted fusion. This is
compatible with the current evaluation but is not intended for a very large
collection.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `https://www.dmxapi.cn/v1` | OpenAI-compatible embedding/LLM endpoint |
| `VECTOR_STORE_BACKEND` | `chroma` | Selects `chroma` or `memory` |
| `CHROMA_PERSIST_DIRECTORY` | `backend/data/chroma` | Local database directory |
| `CHROMA_COLLECTION_NAME` | `documind_chunks` | Logical vector record collection |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding identity |

The collection records the embedding model and DocuMind storage schema version.
Startup fails with an actionable error when these do not match configuration.
A model or dimension change requires a new collection and document re-indexing;
old and new embedding spaces must not be mixed.

Chroma automatically persists records under the configured directory and loads
the collection when a new application process starts. In Docker, that directory
must be mounted to a persistent volume if data must survive container
replacement.

The project pins `chromadb==0.6.3` and uses Python 3.11 on Windows. Chroma 1.5.9
on Python 3.13 and Chroma 1.0.20 on Python 3.11 were rejected after their native
Rust `_upsert` operation caused a reproducible process-level access violation in
this environment. Chroma 0.6.3 also requires the current Microsoft Visual C++
x64 Redistributable for its ONNX dependency to load. An upgrade must repeat the
persistence regression tests before changing the pin.

## Evaluation and reset safety

Normal application startup never clears persisted data. The evaluation runner
uses `eval/.chroma` and the `documind_eval_chunks` collection, then resets only
that collection before ingesting the fixture. This prevents evaluation setup
from deleting development documents.

`clear()` is a test and administrative lifecycle operation, not part of normal
ingestion or restart recovery.

## Trade-offs and limitations

Chroma PersistentClient is a reliable local-MVP step, not a complete enterprise
database deployment:

- it depends on storage attached to one machine;
- it is not the intended shared database for multiple application hosts;
- container persistence requires a mounted volume;
- backup, replication, tenant authorization, and disaster recovery remain
  infrastructure responsibilities;
- repeated files with new document IDs are not content-deduplicated;
- no document deletion API or document registry exists yet;
- full-collection hybrid fusion will need redesign as the corpus grows;
- BM25 is rebuilt in memory after restart rather than persisted independently.

## Future migration to pgvector

The `VectorStore` boundary allows a future `PgVectorStore` adapter without
changing the RAG orchestration or API schema. pgvector becomes appropriate when
DocuMind needs shared multi-instance storage, relational document state,
transactions, tenant permissions, document versioning, mature backups, or
centralized operations.

A safe migration would:

1. define relational `documents` and `chunks` schemas;
2. preserve stable chunk IDs and metadata semantics;
3. backfill existing content and embeddings;
4. compare score and evaluation behavior;
5. switch dependency configuration after verification;
6. retain the old collection until rollback is no longer required.

## Verification result

Verified on Windows with Python 3.11.9 and `chromadb==0.6.3`:

- Python compilation passed;
- existing score-semantics regression passed 3/3;
- persistent store tests passed 5/5;
- FastAPI `/health` returned HTTP 200;
- dependency consistency check reported no broken requirements;
- a store wrote a record, stopped its Chroma system, reopened the same path,
  and retrieved the record without re-indexing;
- metadata, idempotent upsert, clear isolation, embedding-model identity, and
  embedding-dimension failure behavior were verified.

The complete 15-case evaluation passed 12 cases. Source, page, and evidence
checks passed 11/11, answer keyword checks passed 8/8, and fallback correctness
passed 4/4. The three remaining failures were classified rather than hidden:
two confidence failures (`case_011`, `case_014`) and one ranking failure
(`case_013`). No thresholds or expectations were changed.

Restart recovery was verified in a separate Python process. It opened the
existing evaluation collection with three persisted chunks, performed no
upload or ingestion, answered a query, and returned three sources from
`Study Plan - MSc Computer Science.pdf`.

The same sequence was also verified through the FastAPI boundary with an
isolated collection: the first process returned HTTP 200 from `/health` and
`/api/documents/parse-pdf`, then indexed three chunks. After that process
stopped, a second process returned HTTP 200 from `/health` and `/api/chat`,
loaded the same three chunks without another upload, produced status
`answered`, and returned the original fixture as its source.
