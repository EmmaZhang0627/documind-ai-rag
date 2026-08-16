from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chunker import split_pages_into_parent_child_chunks
from app.services.parent_child_retrieval import resolve_parent_context
from app.services.rag import RAGService
from app.services.chroma_vector_store import ChromaPersistentVectorStore
from chromadb.api.shared_system_client import SharedSystemClient


def child_candidate(chunk: dict, score: float = 0.8) -> dict:
    metadata = {
        key: value
        for key, value in chunk.items()
        if key not in {"content", "embedding", "start_char", "end_char"}
    }
    return {
        "document": chunk["content"],
        "metadata": metadata,
        "embedding_score": score,
        "bm25_score": score,
        "retrieval_score": score,
    }


class FakeEmbedder:
    def embed(self, text):
        return [1.0]


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve(self, query_embedding, query_text, top_k=10):
        return self.candidates[:top_k]


class FakeReranker:
    def rerank(self, query, candidates):
        for candidate in candidates:
            candidate["rerank_enabled"] = True
            candidate["rerank_score"] = 1.0
        return candidates


class CapturingLLM:
    def __init__(self):
        self.context = None

    def generate(self, query, context):
        self.context = context
        return "grounded answer"


class ParentChildRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = [{
            "page_number": 2,
            "text": "A" * 500 + " target fact " + "B" * 700,
        }]

    def chunks(self):
        return split_pages_into_parent_child_chunks(
            self.pages,
            document_id="document-1",
            source_file="policy.pdf",
            parent_size=1000,
            child_size=400,
            child_overlap=100,
            version="2",
        )

    def test_parent_id_is_deterministic(self) -> None:
        first = self.chunks()
        second = self.chunks()
        self.assertEqual(
            [chunk["parent_id"] for chunk in first],
            [chunk["parent_id"] for chunk in second],
        )
        self.assertEqual(first[0]["parent_id"], "document-1:2:page-2:parent-0-1000")

    def test_sibling_children_resolve_one_parent(self) -> None:
        chunks = self.chunks()
        siblings = [chunk for chunk in chunks if chunk["parent_id"] == chunks[0]["parent_id"]]
        resolved = resolve_parent_context([child_candidate(chunk) for chunk in siblings[:2]])

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["document"], chunks[0]["parent_text"])

    def test_parent_preserves_document_and_page_traceability(self) -> None:
        chunk = self.chunks()[0]
        resolved = resolve_parent_context([child_candidate(chunk)])[0]

        self.assertEqual(resolved["metadata"]["document_id"], "document-1")
        self.assertEqual(resolved["metadata"]["version"], "2")
        self.assertEqual(resolved["metadata"]["page_number"], 2)
        self.assertEqual(resolved["metadata"]["source_file"], "policy.pdf")

    def test_single_evidence_child_still_drives_retrieval_and_parent_context(self) -> None:
        matching = next(chunk for chunk in self.chunks() if "target fact" in chunk["content"])
        candidate = child_candidate(matching)
        llm = CapturingLLM()
        service = RAGService(
            embedder=FakeEmbedder(),
            retriever=FakeRetriever([candidate]),
            reranker=FakeReranker(),
            llm=llm,
            parent_child_retrieval_enabled=True,
        )

        response = service.ask("What is the target fact?")

        self.assertEqual(response["status"], "answered")
        self.assertIn("target fact", llm.context)
        self.assertEqual(llm.context, matching["parent_text"])
        self.assertEqual(response["sources"][0]["page_number"], 2)

    def test_chroma_round_trip_preserves_parent_metadata(self) -> None:
        chunk = self.chunks()[0]
        chunk["embedding"] = [1.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaPersistentVectorStore(
                persist_directory=directory,
                collection_name="parent_child_metadata",
                embedding_model_name="test-model",
            )
            store.add([chunk])
            result = store.search([1.0, 0.0], "target", top_k=1)[0]

            self.assertEqual(result["metadata"]["parent_id"], chunk["parent_id"])
            self.assertEqual(result["metadata"]["child_index"], 0)
            self.assertEqual(result["metadata"]["parent_text"], chunk["parent_text"])
            self.assertEqual(result["metadata"]["page_number"], 2)
            store.client._system.stop()
            SharedSystemClient.clear_system_cache()


if __name__ == "__main__":
    unittest.main()
