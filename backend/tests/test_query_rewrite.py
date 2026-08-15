from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.query_rewrite_service import QueryRewriteService
from app.services.rag import RAGService


class FakeEmbedder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [1.0]


class FakeRetriever:
    def __init__(self) -> None:
        self.query_texts: list[str] = []

    def retrieve(self, query_embedding, query_text, top_k=10):
        self.query_texts.append(query_text)
        return [{
            "document": "The policy permits a study break.",
            "metadata": {
                "document_id": "policy",
                "source_file": "policy.pdf",
                "chunk_index": 0,
                "page_number": 1,
            },
            "embedding_score": 0.8,
            "bm25_score": 0.7,
            "retrieval_score": 0.77,
        }]


class FakeReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(self, query, candidates):
        self.queries.append(query)
        for candidate in candidates:
            candidate["rerank_enabled"] = True
            candidate["rerank_score"] = 1.0
        return candidates


class FakeLLM:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def generate(self, query, context):
        self.queries.append(query)
        return "A study break is permitted."


class QueryRewriteServiceTest(unittest.TestCase):
    def test_short_clear_query_is_not_rewritten(self) -> None:
        calls: list[str] = []
        service = QueryRewriteService(lambda query: calls.append(query) or "changed")

        result = service.rewrite("What is collusion?")

        self.assertEqual(result, "What is collusion?")
        self.assertEqual(calls, [])

    def test_long_natural_question_is_rewritten(self) -> None:
        service = QueryRewriteService(
            lambda _: "circumstances preventing the usual two-year completion schedule?"
        )

        result = service.rewrite(
            "What circumstances can make the usual two-year completion schedule unachievable?"
        )

        self.assertIn("circumstances", result)

    def test_failure_empty_or_new_number_falls_back_to_original(self) -> None:
        original = "What circumstances can make the usual study schedule unachievable?"
        for output in (
            "",
            "The answer is a study break",
            "Maximum duration is 72 months",
            "According to University Policy the schedule can change",
        ):
            with self.subTest(output=output):
                service = QueryRewriteService(lambda _: output)
                self.assertEqual(service.rewrite(original), original)

    def test_rag_uses_rewrite_for_retrieval_but_original_for_generation(self) -> None:
        original = "What circumstances can make the usual study schedule unachievable?"
        rewritten = "circumstances preventing usual study schedule completion"
        embedder = FakeEmbedder()
        retriever = FakeRetriever()
        reranker = FakeReranker()
        llm = FakeLLM()
        service = RAGService(
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            query_rewriter=QueryRewriteService(lambda _: rewritten),
        )

        response = service.ask(original)

        self.assertEqual(embedder.inputs, [rewritten])
        self.assertEqual(retriever.query_texts, [rewritten])
        self.assertEqual(reranker.queries, [rewritten])
        self.assertEqual(llm.queries, [original])
        self.assertEqual(response["trace"]["original_query"], original)
        self.assertEqual(response["trace"]["retrieval_query"], rewritten)


if __name__ == "__main__":
    unittest.main()
