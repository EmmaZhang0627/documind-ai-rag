from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chunker import split_pages_into_chunks
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
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["document_id"], "policy")
            self.assertEqual(chunk["version"], "2")
            self.assertEqual(chunk["status"], "ACTIVE")
            self.assertEqual(chunk["file_name"], "Policy_v2.pdf")
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


if __name__ == "__main__":
    unittest.main()
