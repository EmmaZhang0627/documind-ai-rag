import logging
from uuid import uuid4

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_types import Candidate, Chunk, RAGResponse, RAGTrace
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService
from app.services.vector_db import is_confident


logger = logging.getLogger(__name__)
ANSWER_TOP_K = 3
CONFIDENCE_THRESHOLD = 0.6
RETRIEVAL_TOP_K = 10


class RAGService:
    def __init__(
        self,
        embedder: EmbeddingService,
        retriever: RetrievalService,
        reranker: RerankService,
        llm: LLMService,
        retrieval_top_k_default: int = RETRIEVAL_TOP_K,
        answer_top_k_default: int = ANSWER_TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.retrieval_top_k_default = retrieval_top_k_default
        self.answer_top_k_default = answer_top_k_default
        self.confidence_threshold = confidence_threshold

    def ingest_document(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            chunk["embedding"] = self.embedder.embed(chunk["content"])

        self.retriever.add(chunks)

    def _build_trace(
        self,
        query: str,
        candidates: list[Candidate],
        ranked: list[Candidate],
        top_k: int,
    ) -> RAGTrace:
        before_rerank_score = candidates[0]["retrieval_score"] if candidates else 0.0
        top_results = ranked[:top_k]
        top1_score = (
            top_results[0].get("score", top_results[0]["retrieval_score"])
            if top_results
            else 0.0
        )
        rerank_enabled = (
            top_results[0].get("rerank_enabled", False)
            if top_results
            else False
        )

        return {
            "trace_id": str(uuid4()),
            "query": query,
            "retrieval": {
                "top1_score": before_rerank_score,
                "top_k_scores": [
                    item["retrieval_score"]
                    for item in candidates[:top_k]
                ],
            },
            "rerank": {
                "enabled": rerank_enabled,
                "improvement": top1_score - before_rerank_score,
            },
            "decision": {
                "passed_gate": None,
            },
        }

    def ask(self, query: str, top_k: int | None = None) -> RAGResponse:
        answer_top_k = max(1, top_k or self.answer_top_k_default)
        query_embedding = self.embedder.embed(query)
        candidates = self.retriever.retrieve(
            query_embedding,
            query,
            top_k=max(self.retrieval_top_k_default, answer_top_k),
        )
        ranked = self.reranker.rerank(query, candidates)
        top_chunks = ranked[:answer_top_k]
        top1_score = (
            top_chunks[0].get("score", top_chunks[0]["retrieval_score"])
            if top_chunks
            else 0.0
        )
        trace = self._build_trace(
            query,
            candidates,
            ranked,
            top_k=answer_top_k,
        )
        passed_gate = is_confident(
            top1_score,
            threshold=self.confidence_threshold,
        )
        trace["decision"]["passed_gate"] = passed_gate
        logger.info("rag_trace=%s", trace)

        if not passed_gate:
            return {
                "answer": "I cannot find relevant information in the documents.",
                "sources": [],
                "trace": trace,
            }

        context = "\n\n".join([chunk["document"] for chunk in top_chunks])
        answer = self.llm.generate(query, context)

        return {
            "answer": answer,
            "sources": [chunk["metadata"] for chunk in top_chunks],
            "trace": trace,
        }
