from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import app.services.rag as rag_module
import app.services.vector_db as vector_db
from app.services.rag import RAGService


def candidate(name: str, retrieval_score: float) -> dict:
    return {
        "document": name,
        "metadata": {
            "document_id": name,
            "source_file": "fixture.pdf",
            "chunk_index": 0,
            "page_number": 1,
        },
        "embedding_score": retrieval_score,
        "bm25_score": 0.0,
        "retrieval_score": retrieval_score,
    }


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = {"retrieval-first": -5.0, "rerank-first": 10.0}
        return [scores[document] for _, document in pairs]


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [1.0]


class FakeRetriever:
    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates

    def retrieve(self, query_embedding, query, top_k=10):
        return self.candidates[:top_k]


class FakeReranker:
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        candidates[0]["rerank_enabled"] = True
        candidates[0]["rerank_score"] = -5.0
        candidates[1]["rerank_enabled"] = True
        candidates[1]["rerank_score"] = 10.0
        return [candidates[1], candidates[0]]


class FailIfCalledLLM:
    def generate(self, query: str, context: str) -> str:
        raise AssertionError("LLM must not run when confidence is below threshold")


def verify_rerank_preserves_retrieval_score() -> None:
    original_reranker = vector_db.reranker
    original_enabled = vector_db.reranker_enabled
    try:
        vector_db.reranker = FakeCrossEncoder()
        vector_db.reranker_enabled = True
        candidates = [candidate("retrieval-first", 0.9), candidate("rerank-first", 0.7)]
        ranked = vector_db.rerank("question", candidates)

        assert ranked[0]["document"] == "rerank-first"
        assert ranked[0]["retrieval_score"] == 0.7
        assert ranked[0]["rerank_score"] == 10.0
        assert ranked[1]["retrieval_score"] == 0.9
    finally:
        vector_db.reranker = original_reranker
        vector_db.reranker_enabled = original_enabled


def verify_fallback_ordering() -> None:
    original_enabled = vector_db.reranker_enabled
    try:
        vector_db.reranker_enabled = False
        candidates = [candidate("lower", 0.2), candidate("higher", 0.8)]
        ranked = vector_db.rerank("question", candidates)

        assert [item["document"] for item in ranked] == ["higher", "lower"]
        assert all(item["rerank_enabled"] is False for item in ranked)
        assert all(item["rerank_score"] is None for item in ranked)
    finally:
        vector_db.reranker_enabled = original_enabled


def verify_confidence_uses_retrieval_score() -> None:
    candidates = [candidate("retrieval-first", 0.9), candidate("rerank-first", 0.7)]
    service = RAGService(
        embedder=FakeEmbedder(),
        retriever=FakeRetriever(candidates),
        reranker=FakeReranker(),
        llm=FailIfCalledLLM(),
        confidence_threshold=0.8,
    )
    original_write_trace = rag_module.write_trace
    try:
        rag_module.write_trace = lambda record: None
        response = service.ask("question")
    finally:
        rag_module.write_trace = original_write_trace

    assert response["status"] == "low_confidence"
    assert response["trace"]["decision"]["confidence_score"] == 0.7
    assert response["trace"]["decision"]["passed_gate"] is False


def main() -> int:
    checks = (
        verify_rerank_preserves_retrieval_score,
        verify_fallback_ordering,
        verify_confidence_uses_retrieval_score,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"Score semantics checks passed: {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
