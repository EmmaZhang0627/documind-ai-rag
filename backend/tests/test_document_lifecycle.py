from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import HTTPException


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chunker import split_pages_into_chunks
from app.api.documents import archive_document_version as archive_endpoint
from app.services.retrieval_service import InMemoryVectorStore


class DocumentLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryVectorStore()
        self.store.clear()

    def tearDown(self) -> None:
        self.store.clear()

    def test_chunker_propagates_document_metadata_to_every_chunk(self) -> None:
        chunks = split_pages_into_chunks(
            pages=[{"page_number": 1, "text": "A" * 900}],
            document_id="policy",
            source_file="Policy_v2.pdf",
            version="2",
            status="ACTIVE",
            created_time="2026-08-02T00:00:00+00:00",
            file_hash="fixture-hash",
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["document_id"], "policy")
            self.assertEqual(chunk["version"], "2")
            self.assertEqual(chunk["status"], "ACTIVE")
            self.assertEqual(chunk["file_name"], "Policy_v2.pdf")
            self.assertEqual(chunk["file_hash"], "fixture-hash")
            self.assertEqual(
                chunk["created_time"], "2026-08-02T00:00:00+00:00"
            )

    def test_in_memory_store_excludes_archived_document(self) -> None:
        base_chunk = {
            "document_id": "policy",
            "chunk_index": 0,
            "page_number": 1,
            "embedding": [1.0, 0.0],
        }
        self.store.add([
            {
                **base_chunk,
                "version": "1",
                "status": "ARCHIVED",
                "source_file": "Policy_v1.pdf",
                "content": "Old policy",
            },
            {
                **base_chunk,
                "version": "2",
                "status": "ACTIVE",
                "source_file": "Policy_v2.pdf",
                "content": "Current policy",
            },
        ])

        results = self.store.search([1.0, 0.0], "policy", top_k=3)

        self.assertEqual(self.store.count(), 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["version"], "2")

    def test_in_memory_archive_only_affects_selected_version(self) -> None:
        base_chunk = {
            "document_id": "policy",
            "chunk_index": 0,
            "page_number": 1,
            "embedding": [1.0, 0.0],
        }
        self.store.add([
            {
                **base_chunk,
                "version": "1",
                "status": "ACTIVE",
                "source_file": "Policy_v1.pdf",
                "content": "Old active policy",
            },
            {
                **base_chunk,
                "version": "2",
                "status": "ACTIVE",
                "source_file": "Policy_v2.pdf",
                "content": "Current active policy",
            },
        ])

        archived_count = self.store.archive_document_version("policy", "1")
        results = self.store.search([1.0, 0.0], "policy", top_k=3)

        self.assertEqual(archived_count, 1)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["version"], "2")
        self.assertEqual(
            self.store.find_by_document_version("policy", "1")["status"],
            "ARCHIVED",
        )
        self.assertEqual(
            self.store.archive_document_version("missing", "1"),
            0,
        )


class FakeArchiveRAGService:
    def __init__(self, archived_chunk_count: int) -> None:
        self.archived_chunk_count = archived_chunk_count

    def archive_document_version(self, document_id: str, version: str) -> int:
        return self.archived_chunk_count


class DocumentLifecycleApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_archive_endpoint_returns_archived_chunk_count(self) -> None:
        result = await archive_endpoint(
            document_id="policy",
            version="1",
            rag_service=FakeArchiveRAGService(3),
        )

        self.assertEqual(result["status"], "ARCHIVED")
        self.assertEqual(result["archived_chunk_count"], 3)

    async def test_archive_endpoint_returns_404_when_version_is_missing(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await archive_endpoint(
                document_id="missing",
                version="1",
                rag_service=FakeArchiveRAGService(0),
            )

        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
