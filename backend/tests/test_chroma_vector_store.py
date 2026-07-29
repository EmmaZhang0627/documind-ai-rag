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


if __name__ == "__main__":
    unittest.main()
