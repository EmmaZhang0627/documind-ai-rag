from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.metadata_permissions import AccessContext
from app.services.rag import RAGService
from app.services.retrieval_service import InMemoryVectorStore, RetrievalService


def chunk(
    document_id: str,
    *,
    tenant_id: str,
    department: str,
    access_level: str,
    status: str = "ACTIVE",
    content: str = "policy evidence",
) -> dict:
    return {
        "document_id": document_id,
        "chunk_index": 0,
        "source_file": f"{document_id}.pdf",
        "page_number": 1,
        "content": content,
        "embedding": [1.0, 0.0],
        "tenant_id": tenant_id,
        "department": department,
        "access_level": access_level,
        "status": status,
    }


class FakeEmbedder:
    def embed(self, _: str) -> list[float]:
        return [1.0, 0.0]


class RecordingReranker:
    def __init__(self) -> None:
        self.seen_document_ids: list[str] = []

    def rerank(self, _query, candidates):
        self.seen_document_ids = [
            item["metadata"]["document_id"] for item in candidates
        ]
        return candidates


class RecordingLLM:
    def __init__(self) -> None:
        self.context = ""

    def generate(self, _query, context):
        self.context = context
        return "allowed answer"


class MetadataPermissionFilteringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryVectorStore()
        self.store.clear()
        self.retriever = RetrievalService(self.store)
        self.retriever.add([
            chunk(
                "public-a",
                tenant_id="tenant-a",
                department="all",
                access_level="public",
                content="General handbook policy.",
            ),
            chunk(
                "finance-a",
                tenant_id="tenant-a",
                department="finance",
                access_level="confidential",
                content="Secret merger plan strongest semantic match.",
            ),
            chunk(
                "finance-internal-a",
                tenant_id="tenant-a",
                department="finance",
                access_level="internal",
                content="Internal finance operating procedure.",
            ),
            chunk(
                "engineering-a",
                tenant_id="tenant-a",
                department="engineering",
                access_level="internal",
            ),
            chunk(
                "public-b",
                tenant_id="tenant-b",
                department="all",
                access_level="public",
            ),
            chunk(
                "archived-a",
                tenant_id="tenant-a",
                department="finance",
                access_level="confidential",
                status="ARCHIVED",
            ),
        ])

    def tearDown(self) -> None:
        self.store.clear()

    def retrieve(self, context: AccessContext) -> list[str]:
        results = self.retriever.retrieve(
            [1.0, 0.0], "secret merger plan", top_k=10, access_context=context
        )
        return [item["metadata"]["document_id"] for item in results]

    def test_tenant_department_access_level_and_lifecycle_filters_compose(self) -> None:
        public_general = AccessContext("tenant-a", frozenset({"general"}), "public")
        finance_confidential = AccessContext(
            "tenant-a", frozenset({"finance"}), "confidential"
        )
        finance_internal = AccessContext(
            "tenant-a", frozenset({"finance"}), "internal"
        )
        engineering_internal = AccessContext(
            "tenant-a", frozenset({"engineering"}), "internal"
        )

        self.assertEqual(self.retrieve(public_general), ["public-a"])
        self.assertEqual(
            set(self.retrieve(finance_confidential)),
            {"public-a", "finance-a", "finance-internal-a"},
        )
        self.assertEqual(
            set(self.retrieve(finance_internal)),
            {"public-a", "finance-internal-a"},
        )
        self.assertEqual(
            set(self.retrieve(engineering_internal)),
            {"public-a", "engineering-a"},
        )
        self.assertNotIn("public-b", self.retrieve(finance_confidential))
        self.assertNotIn("archived-a", self.retrieve(finance_confidential))

    def test_missing_or_partial_permission_metadata_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.add([{
                "document_id": "partial",
                "chunk_index": 0,
                "source_file": "partial.pdf",
                "page_number": 1,
                "content": "partial permission metadata",
                "embedding": [1.0, 0.0],
                "tenant_id": "tenant-a",
            }])

    def test_restricted_chunk_never_reaches_reranker_or_llm_context(self) -> None:
        reranker = RecordingReranker()
        llm = RecordingLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            retriever=self.retriever,
            reranker=reranker,
            llm=llm,
            query_rewriter=None,
            confidence_threshold=0.0,
        )
        response = service.ask(
            "What is the secret merger plan?",
            access_context=AccessContext(
                "tenant-a", frozenset({"general"}), "public"
            ),
        )

        self.assertEqual(reranker.seen_document_ids, ["public-a"])
        self.assertNotIn("finance-a", reranker.seen_document_ids)
        self.assertNotIn("Secret merger", llm.context)
        self.assertEqual(response["sources"][0]["document_id"], "public-a")


if __name__ == "__main__":
    unittest.main()
