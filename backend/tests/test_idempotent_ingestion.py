from __future__ import annotations

import hashlib
import json
import sys
import unittest
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api import documents as documents_api
from app.api.documents import parse_pdf
from app.services.rag import RAGService
from app.services.retrieval_service import InMemoryVectorStore, RetrievalService


class FakeUploadFile:
    def __init__(self, content: bytes, filename: str = "policy.pdf") -> None:
        self.content_type = "application/pdf"
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


class FakeRAGService:
    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision
        self.check_calls = 0
        self.ingest_calls = 0

    def check_document_ingestion(
        self,
        file_hash: str,
        document_id: str | None,
        version: str,
    ) -> dict[str, Any]:
        self.check_calls += 1
        return self.decision

    def ingest_document(self, chunks: list[dict[str, Any]]) -> None:
        self.ingest_calls += 1
        raise AssertionError("Duplicate/conflict must return before embedding.")


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0, 0.0]


class IdempotentIngestionTest(unittest.IsolatedAsyncioTestCase):
    async def test_uploading_same_pdf_twice_does_not_reembed_or_add_chunks(self) -> None:
        fixture_path = (
            BACKEND_ROOT.parent
            / "eval"
            / "fixtures"
            / "Study Plan - MSc Computer Science.pdf"
        )
        content = fixture_path.read_bytes()
        store = InMemoryVectorStore()
        store.clear()
        rag_service = RAGService.__new__(RAGService)
        rag_service.retriever = RetrievalService(store)
        rag_service.embedder = CountingEmbedder()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(documents_api, "UPLOAD_DIR", Path(directory)):
                first = await parse_pdf(
                    file=FakeUploadFile(content, fixture_path.name),
                    document_id="study-plan",
                    version="1",
                    status="ACTIVE",
                    rag_service=rag_service,
                )
                first_count = store.count()
                first_embedding_calls = rag_service.embedder.calls

                second = await parse_pdf(
                    file=FakeUploadFile(content, fixture_path.name),
                    document_id="study-plan",
                    version="1",
                    status="ACTIVE",
                    rag_service=rag_service,
                )

        self.assertEqual(first["ingestion_result"], "indexed")
        self.assertEqual(second["ingestion_result"], "duplicate")
        self.assertGreater(first_count, 0)
        self.assertEqual(store.count(), first_count)
        self.assertEqual(rag_service.embedder.calls, first_embedding_calls)
        store.clear()

    async def test_exact_duplicate_returns_before_parsing_or_embedding(self) -> None:
        content = b"not-even-a-valid-pdf"
        file_hash = hashlib.sha256(content).hexdigest()
        rag_service = FakeRAGService({
            "result": "duplicate",
            "existing_document": {
                "document_id": "policy",
                "version": "1",
                "status": "ACTIVE",
                "file_hash": file_hash,
                "file_name": "Policy_v1.pdf",
                "created_time": "2026-08-04T00:00:00+00:00",
            },
        })

        response = await parse_pdf(
            file=FakeUploadFile(content),
            document_id=None,
            version="1",
            status="ACTIVE",
            rag_service=rag_service,
        )

        self.assertEqual(response["ingestion_result"], "duplicate")
        self.assertEqual(response["document_id"], "policy")
        self.assertEqual(response["file_hash"], file_hash)
        self.assertEqual(rag_service.check_calls, 1)
        self.assertEqual(rag_service.ingest_calls, 0)

    async def test_version_conflict_returns_409_before_parsing(self) -> None:
        rag_service = FakeRAGService({
            "result": "version_conflict",
            "existing_document": {
                "document_id": "policy",
                "version": "1",
                "status": "ACTIVE",
                "file_hash": "existing-hash",
                "file_name": "Policy_v1.pdf",
                "created_time": "2026-08-04T00:00:00+00:00",
            },
        })

        response = await parse_pdf(
            file=FakeUploadFile(b"different-invalid-pdf"),
            document_id="policy",
            version="1",
            status="ACTIVE",
            rag_service=rag_service,
        )
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(body["ingestion_result"], "version_conflict")
        self.assertEqual(rag_service.ingest_calls, 0)


class IngestionDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryVectorStore()
        self.store.clear()
        self.retriever = RetrievalService(self.store)
        self.rag_service = RAGService.__new__(RAGService)
        self.rag_service.retriever = self.retriever

    def tearDown(self) -> None:
        self.store.clear()

    def test_same_document_version_with_different_hash_is_conflict(self) -> None:
        self.store.add([{
            "document_id": "policy",
            "version": "1",
            "status": "ACTIVE",
            "file_hash": "existing-hash",
            "source_file": "Policy_v1.pdf",
            "chunk_index": 0,
            "page_number": 1,
            "content": "Existing policy content",
            "embedding": [1.0, 0.0],
        }])

        decision = self.rag_service.check_document_ingestion(
            file_hash="different-hash",
            document_id="policy",
            version="1",
        )

        self.assertEqual(decision["result"], "version_conflict")
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(
            self.store.find_by_document_version("policy", "1")["file_hash"],
            "existing-hash",
        )

    def test_same_hash_is_duplicate_even_with_different_requested_identity(self) -> None:
        self.store.add([{
            "document_id": "policy",
            "version": "1",
            "status": "ACTIVE",
            "file_hash": "same-hash",
            "source_file": "Policy_v1.pdf",
            "chunk_index": 0,
            "page_number": 1,
            "content": "Existing policy content",
            "embedding": [1.0, 0.0],
        }])

        decision = self.rag_service.check_document_ingestion(
            file_hash="same-hash",
            document_id="different-document",
            version="9",
        )

        self.assertEqual(decision["result"], "duplicate")
        self.assertEqual(
            decision["existing_document"]["document_id"],
            "policy",
        )


if __name__ == "__main__":
    unittest.main()
