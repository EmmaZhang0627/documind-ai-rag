from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import chromadb

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chroma_vector_store import ChromaPersistentVectorStore
from app.services.rag import RAGService
from app.services.retrieval_service import RetrievalService
from chromadb.api.shared_system_client import SharedSystemClient


class ChromaPersistentVectorStoreTest(unittest.TestCase):
    def _build_store(self, directory: str) -> ChromaPersistentVectorStore:
        return ChromaPersistentVectorStore(
            persist_directory=directory,
            collection_name="test_chunks",
            embedding_model_name="test-embedding-model",
        )

    def _stop_store(self, store: ChromaPersistentVectorStore) -> None:
        store.client._system.stop()
        SharedSystemClient.clear_system_cache()

    def test_records_survive_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_store = self._build_store(directory)
            first_store.add([
                {
                    "document_id": "document-1",
                    "chunk_index": 0,
                    "source_file": "fixture.pdf",
                    "page_number": 2,
                    "content": "Persistent vector storage survives application restarts.",
                    "embedding": [1.0, 0.0, 0.0],
                }
            ])
            self._stop_store(first_store)

            second_store = self._build_store(directory)
            results = second_store.search(
                query_embedding=[1.0, 0.0, 0.0],
                query_text="persistent storage",
                top_k=1,
            )

            self.assertEqual(second_store.count(), 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["document_id"], "document-1")
            self.assertEqual(results[0]["metadata"]["source_file"], "fixture.pdf")
            self.assertEqual(results[0]["metadata"]["page_number"], 2)
            self.assertIn("survives application restarts", results[0]["document"])
            self._stop_store(second_store)

    def test_upsert_is_idempotent_for_same_chunk_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            chunk = {
                "document_id": "document-1",
                "chunk_index": 0,
                "source_file": "fixture.pdf",
                "page_number": None,
                "content": "Original content",
                "embedding": [1.0, 0.0, 0.0],
            }
            store.add([chunk])
            store.add([{**chunk, "content": "Updated content"}])

            results = store.search([1.0, 0.0, 0.0], "updated", top_k=1)
            self.assertEqual(store.count(), 1)
            self.assertEqual(results[0]["document"], "Updated content")
            self.assertIsNone(results[0]["metadata"]["page_number"])
            self._stop_store(store)

    def test_clear_only_removes_records_from_selected_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            store.add([
                {
                    "document_id": "document-1",
                    "chunk_index": 0,
                    "source_file": "fixture.pdf",
                    "page_number": 1,
                    "content": "Temporary evaluation content",
                    "embedding": [1.0, 0.0, 0.0],
                }
            ])

            store.clear()

            self.assertEqual(store.count(), 0)
            self.assertEqual(store.search([1.0, 0.0, 0.0], "content"), [])
            self._stop_store(store)

    def test_rejects_embedding_model_change_for_existing_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            self._stop_store(store)

            with self.assertRaisesRegex(ValueError, "embedding model mismatch"):
                ChromaPersistentVectorStore(
                    persist_directory=directory,
                    collection_name="test_chunks",
                    embedding_model_name="different-embedding-model",
                )

            cleanup_client = chromadb.PersistentClient(path=directory)
            cleanup_client._system.stop()
            SharedSystemClient.clear_system_cache()

    def test_rejects_embedding_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            store.add([
                {
                    "document_id": "document-1",
                    "chunk_index": 0,
                    "source_file": "fixture.pdf",
                    "page_number": 1,
                    "content": "Three dimensional embedding",
                    "embedding": [1.0, 0.0, 0.0],
                }
            ])

            with self.assertRaises(Exception):
                store.search([1.0, 0.0], "dimension mismatch", top_k=1)

            self._stop_store(store)

    def test_active_document_can_be_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            store.add([{
                "document_id": "policy",
                "version": "2",
                "status": "ACTIVE",
                "created_time": "2026-08-02T00:00:00+00:00",
                "chunk_index": 0,
                "source_file": "Credit_Policy_v2.pdf",
                "file_name": "Credit_Policy_v2.pdf",
                "page_number": 1,
                "content": "The active credit policy applies now.",
                "embedding": [1.0, 0.0, 0.0],
            }])

            results = store.search([1.0, 0.0, 0.0], "active policy", top_k=3)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["status"], "ACTIVE")
            self.assertEqual(results[0]["metadata"]["version"], "2")
            self._stop_store(store)

    def test_archived_document_is_stored_but_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            store.add([{
                "document_id": "policy",
                "version": "1",
                "status": "ARCHIVED",
                "created_time": "2026-08-01T00:00:00+00:00",
                "chunk_index": 0,
                "source_file": "Credit_Policy_v1.pdf",
                "page_number": 1,
                "content": "The archived credit policy must not be used.",
                "embedding": [1.0, 0.0, 0.0],
            }])

            self.assertEqual(store.count(), 1)
            self.assertEqual(
                store.search([1.0, 0.0, 0.0], "archived policy", top_k=3),
                [],
            )
            self._stop_store(store)

    def test_two_versions_can_coexist_and_only_active_is_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._build_store(directory)
            base_chunk = {
                "document_id": "credit-policy",
                "chunk_index": 0,
                "page_number": 1,
                "embedding": [1.0, 0.0, 0.0],
            }
            store.add([
                {
                    **base_chunk,
                    "version": "1",
                    "status": "ARCHIVED",
                    "file_hash": "hash-version-1",
                    "source_file": "Credit_Policy_v1.pdf",
                    "content": "Version one is retained for audit.",
                },
                {
                    **base_chunk,
                    "version": "2",
                    "status": "ACTIVE",
                    "file_hash": "hash-version-2",
                    "source_file": "Credit_Policy_v2.pdf",
                    "content": "Version two is the current policy.",
                },
            ])
            self.assertEqual(store.count(), 2)
            self.assertEqual(
                store._corpus_ids,
                ["credit-policy:2:0"],
            )
            self._stop_store(store)

            restarted_store = self._build_store(directory)
            results = restarted_store.search(
                [1.0, 0.0, 0.0], "credit policy", top_k=3
            )

            self.assertEqual(restarted_store.count(), 2)
            by_hash = restarted_store.find_by_file_hash("hash-version-2")
            by_version = restarted_store.find_by_document_version(
                "credit-policy", "1"
            )
            self.assertIsNotNone(by_hash)
            self.assertIsNotNone(by_version)
            self.assertEqual(by_hash["version"], "2")
            self.assertEqual(by_version["file_hash"], "hash-version-1")
            rag_service = RAGService.__new__(RAGService)
            rag_service.retriever = RetrievalService(restarted_store)
            duplicate_decision = rag_service.check_document_ingestion(
                "hash-version-2", None, "1"
            )
            self.assertEqual(duplicate_decision["result"], "duplicate")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["version"], "2")
            self.assertEqual(
                results[0]["metadata"]["source_file"],
                "Credit_Policy_v2.pdf",
            )
            self._stop_store(restarted_store)

    def test_legacy_record_without_status_remains_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_store = self._build_store(directory)
            first_store.collection.upsert(
                ids=["legacy-document:0"],
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["Legacy records remain active for compatibility."],
                metadatas=[{
                    "document_id": "legacy-document",
                    "source_file": "legacy.pdf",
                    "chunk_index": 0,
                }],
            )
            self._stop_store(first_store)

            second_store = self._build_store(directory)
            results = second_store.search(
                [1.0, 0.0, 0.0], "legacy compatibility", top_k=1
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["status"], "ACTIVE")
            self.assertEqual(results[0]["metadata"]["version"], "1")
            self.assertIsNone(results[0]["metadata"]["created_time"])
            self._stop_store(second_store)


if __name__ == "__main__":
    unittest.main()
