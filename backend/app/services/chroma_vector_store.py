from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from app.services.rag_types import Candidate, Chunk, StoredDocumentIdentity


logger = logging.getLogger(__name__)
COLLECTION_SCHEMA_VERSION = 1


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]

    return [float(score) / float(max_score) for score in scores]


def _is_active_metadata(metadata: dict[str, Any] | None) -> bool:
    stored = metadata or {}
    status = stored.get("status") or stored.get("document_status")
    return status is None or str(status).upper() == "ACTIVE"


class ChromaPersistentVectorStore:
    """Persistent storage adapter that preserves DocuMind candidate semantics."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model_name: str,
        embedding_score_weight: float = 0.7,
        bm25_score_weight: float = 0.3,
    ) -> None:
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_score_weight = embedding_score_weight
        self.bm25_score_weight = bm25_score_weight

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "documind_schema_version": COLLECTION_SCHEMA_VERSION,
                "embedding_model": embedding_model_name,
            },
            embedding_function=None,
        )
        self._validate_collection_identity()
        self._corpus_ids: list[str] = []
        self._corpus: list[str] = []
        self._bm25_model: BM25Okapi | None = None
        self._rebuild_bm25_index()

    def _validate_collection_identity(self) -> None:
        metadata = self.collection.metadata or {}
        stored_model = metadata.get("embedding_model")
        stored_schema_version = metadata.get("documind_schema_version")

        if stored_model != self.embedding_model_name:
            raise ValueError(
                "Chroma collection embedding model mismatch: "
                f"collection={stored_model!r}, configured={self.embedding_model_name!r}. "
                "Use a new CHROMA_COLLECTION_NAME and re-index the documents."
            )

        if stored_schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError(
                "Chroma collection schema mismatch: "
                f"collection={stored_schema_version!r}, "
                f"expected={COLLECTION_SCHEMA_VERSION!r}. "
                "Use a new collection or run an explicit migration."
            )

    def _rebuild_bm25_index(self) -> None:
        records = self.collection.get(include=["documents", "metadatas"])
        ids = records.get("ids") or []
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        active_records = [
            (record_id, document or "")
            for record_id, document, metadata in zip(ids, documents, metadatas)
            if _is_active_metadata(metadata)
        ]
        self._corpus_ids = [record_id for record_id, _ in active_records]
        self._corpus = [document for _, document in active_records]
        tokenized_corpus = [_tokenize(document) for document in self._corpus]
        self._bm25_model = (
            BM25Okapi(tokenized_corpus)
            if tokenized_corpus
            else None
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for chunk in chunks:
            embedding = chunk.get("embedding")
            if embedding is None:
                raise ValueError("Each chunk must contain an embedding before storage.")

            source_file = chunk["source_file"]
            content = chunk["content"]
            version = chunk.get("version", "1")
            status = str(chunk.get("status", "ACTIVE")).upper()
            if status not in {"ACTIVE", "ARCHIVED"}:
                raise ValueError("Document status must be ACTIVE or ARCHIVED.")
            metadata: dict[str, str | int | float | bool] = {
                "document_id": chunk["document_id"],
                "source_file": source_file,
                "file_name": chunk.get("file_name", source_file),
                "version": version,
                "status": status,
                "chunk_index": chunk["chunk_index"],
                "text": content,
            }
            created_time = chunk.get("created_time")
            if created_time is not None:
                metadata["created_time"] = created_time
            file_hash = chunk.get("file_hash")
            if file_hash is not None:
                metadata["file_hash"] = file_hash
            page_number = chunk.get("page_number")
            if page_number is not None:
                metadata["page_number"] = page_number

            ids.append(
                f"{chunk['document_id']}:{version}:{chunk['chunk_index']}"
            )
            embeddings.append(embedding)
            documents.append(content)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._rebuild_bm25_index()
        logger.info(
            "chroma_chunks_upserted count=%s collection=%s total=%s",
            len(chunks),
            self.collection_name,
            self.count(),
        )

    def _candidate_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        stored = metadata or {}
        return {
            "document_id": stored.get("document_id"),
            "source_file": stored.get("source_file") or stored.get("file_name"),
            "file_name": stored.get("file_name") or stored.get("source_file"),
            "version": str(stored.get("version", "1")),
            "status": str(
                stored.get("status") or stored.get("document_status") or "ACTIVE"
            ).upper(),
            "created_time": stored.get("created_time"),
            "file_hash": stored.get("file_hash"),
            "chunk_index": stored.get("chunk_index"),
            "page_number": stored.get("page_number"),
        }

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        record_count = self.count()
        if record_count == 0 or top_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=record_count,
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        if self._bm25_model is not None and query_text.strip():
            raw_bm25_scores = self._bm25_model.get_scores(_tokenize(query_text))
            normalized_bm25_scores = _normalize_scores(
                [float(score) for score in raw_bm25_scores]
            )
            bm25_by_document = {
                record_id: normalized_bm25_scores[index]
                for index, record_id in enumerate(self._corpus_ids)
            }
        else:
            bm25_by_document = {}

        result_ids = (results.get("ids") or [[]])[0]
        candidates: list[Candidate] = []
        for index, record_id in enumerate(result_ids):
            if not _is_active_metadata(metadatas[index]):
                continue
            document = documents[index] or ""
            distance = float(distances[index])
            embedding_score = 1.0 - distance
            bm25_score = bm25_by_document.get(record_id, 0.0)
            retrieval_score = (
                self.embedding_score_weight * embedding_score
                + self.bm25_score_weight * bm25_score
            )
            candidates.append({
                "document": document,
                "metadata": self._candidate_metadata(metadatas[index]),
                "embedding_score": embedding_score,
                "bm25_score": bm25_score,
                "retrieval_score": retrieval_score,
            })

        candidates.sort(
            key=lambda candidate: candidate["retrieval_score"],
            reverse=True,
        )
        return candidates[:top_k]

    def _stored_document_identity(
        self,
        metadata: dict[str, Any] | None,
    ) -> StoredDocumentIdentity | None:
        stored = metadata or {}
        document_id = stored.get("document_id")
        if not document_id:
            return None

        return {
            "document_id": str(document_id),
            "version": str(stored.get("version", "1")),
            "status": str(
                stored.get("status") or stored.get("document_status") or "ACTIVE"
            ).upper(),
            "file_hash": stored.get("file_hash"),
            "file_name": str(
                stored.get("file_name") or stored.get("source_file") or ""
            ),
            "created_time": stored.get("created_time"),
        }

    def _find_document(
        self,
        where: dict[str, Any],
    ) -> StoredDocumentIdentity | None:
        records = self.collection.get(
            where=where,
            limit=1,
            include=["metadatas"],
        )
        metadatas = records.get("metadatas") or []
        if not metadatas:
            return None
        return self._stored_document_identity(metadatas[0])

    def find_by_file_hash(
        self,
        file_hash: str,
    ) -> StoredDocumentIdentity | None:
        return self._find_document({"file_hash": file_hash})

    def find_by_document_version(
        self,
        document_id: str,
        version: str,
    ) -> StoredDocumentIdentity | None:
        return self._find_document({
            "$and": [
                {"document_id": document_id},
                {"version": version},
            ]
        })

    def archive_document_version(
        self,
        document_id: str,
        version: str,
    ) -> int:
        records = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        ids = records.get("ids") or []
        metadatas = records.get("metadatas") or []
        matching_ids: list[str] = []
        archived_metadatas: list[dict[str, str | int | float | bool]] = []

        for record_id, metadata in zip(ids, metadatas):
            stored = dict(metadata or {})
            if str(stored.get("version", "1")) != version:
                continue
            stored["status"] = "ARCHIVED"
            matching_ids.append(record_id)
            archived_metadatas.append(stored)

        if not matching_ids:
            return 0

        self.collection.update(
            ids=matching_ids,
            metadatas=archived_metadatas,
        )
        self._rebuild_bm25_index()
        logger.info(
            "chroma_document_version_archived document_id=%s version=%s chunks=%s",
            document_id,
            version,
            len(matching_ids),
        )
        return len(matching_ids)

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        records = self.collection.get(include=[])
        ids = records.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        self._rebuild_bm25_index()
        logger.info("chroma_collection_cleared collection=%s", self.collection_name)
