from app.services.rag_types import (
    Candidate,
    Chunk,
    StoredDocumentIdentity,
    VectorStore,
)
from app.services.vector_db import (
    add_chunks_to_db,
    archive_document_version as archive_in_memory_document_version,
    clear_vector_store,
    retrieve_candidates,
    vector_store,
)


class InMemoryVectorStore:
    def add(self, chunks: list[Chunk]) -> None:
        add_chunks_to_db(chunks)

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        return retrieve_candidates(query_embedding, query_text)[:top_k]

    def count(self) -> int:
        return len(vector_store)

    def _stored_document_identity(
        self,
        metadata: dict,
    ) -> StoredDocumentIdentity | None:
        document_id = metadata.get("document_id")
        if not document_id:
            return None
        return {
            "document_id": str(document_id),
            "version": str(metadata.get("version", "1")),
            "status": str(
                metadata.get("status")
                or metadata.get("document_status")
                or "ACTIVE"
            ).upper(),
            "file_hash": metadata.get("file_hash"),
            "file_name": str(
                metadata.get("file_name") or metadata.get("source_file") or ""
            ),
            "created_time": metadata.get("created_time"),
        }

    def find_by_file_hash(
        self,
        file_hash: str,
    ) -> StoredDocumentIdentity | None:
        for item in vector_store:
            if item["metadata"].get("file_hash") == file_hash:
                return self._stored_document_identity(item["metadata"])
        return None

    def find_by_document_version(
        self,
        document_id: str,
        version: str,
    ) -> StoredDocumentIdentity | None:
        for item in vector_store:
            metadata = item["metadata"]
            if (
                metadata.get("document_id") == document_id
                and str(metadata.get("version", "1")) == version
            ):
                return self._stored_document_identity(metadata)
        return None

    def archive_document_version(
        self,
        document_id: str,
        version: str,
    ) -> int:
        return archive_in_memory_document_version(document_id, version)

    def clear(self) -> None:
        clear_vector_store()


class RetrievalService:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def add(self, chunks: list[Chunk]) -> None:
        self.vector_store.add(chunks)

    def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        return self.vector_store.search(query_embedding, query_text, top_k)

    def count(self) -> int:
        return self.vector_store.count()

    def find_by_file_hash(
        self,
        file_hash: str,
    ) -> StoredDocumentIdentity | None:
        return self.vector_store.find_by_file_hash(file_hash)

    def find_by_document_version(
        self,
        document_id: str,
        version: str,
    ) -> StoredDocumentIdentity | None:
        return self.vector_store.find_by_document_version(document_id, version)

    def archive_document_version(
        self,
        document_id: str,
        version: str,
    ) -> int:
        return self.vector_store.archive_document_version(document_id, version)

    def clear(self) -> None:
        self.vector_store.clear()
