import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.observability.trace_logger import write_trace
from app.observability.trace_models import RAGTraceRecord, TraceCandidate
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_types import (
    Candidate,
    Chunk,
    ChunkMetadata,
    RAGResponse,
    RAGTrace,
)
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService
from app.services.vector_db import is_confident


logger = logging.getLogger(__name__)
ANSWER_TOP_K = 3
CONFIDENCE_THRESHOLD = 0.6
RETRIEVAL_TOP_K = 10
MAX_TRACE_SNIPPET_LENGTH = 240
MAX_SOURCE_SNIPPET_LENGTH = 400


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
        trace_id: str,
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
            "trace_id": trace_id,
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

    def _candidate_to_trace(self, candidate: Candidate) -> TraceCandidate:
        metadata = candidate["metadata"]
        final_score = candidate.get("score", candidate["retrieval_score"])
        trace_candidate: TraceCandidate = {
            "document_id": metadata.get("document_id"),
            "source_file": metadata.get("source_file"),
            "page_number": metadata.get("page_number"),
            "chunk_index": metadata.get("chunk_index"),
            "embedding_score": candidate.get("embedding_score"),
            "bm25_score": candidate.get("bm25_score"),
            "final_score": final_score,
        }

        if "rerank_score" in candidate:
            trace_candidate["rerank_score"] = candidate.get("rerank_score")

        document = candidate.get("document", "")
        if document:
            trace_candidate["snippet"] = (
                document.replace("\n", " ")[:MAX_TRACE_SNIPPET_LENGTH]
            )

        return trace_candidate

    def _candidate_to_source(self, candidate: Candidate) -> ChunkMetadata:
        metadata = candidate["metadata"].copy()
        document = candidate.get("document", "")
        if document:
            metadata["source_snippet"] = " ".join(document.split())[
                :MAX_SOURCE_SNIPPET_LENGTH
            ]

        return metadata

    def _build_observability_trace(
        self,
        trace_id: str,
        query: str,
        retrieval_top_k: int,
        candidates: list[Candidate],
        ranked: list[Candidate],
        answer_top_k: int,
        top1_score: float,
        confidence_decision: str,
        llm_called: bool,
        final_status: str,
        error_message: str | None = None,
    ) -> RAGTraceRecord:
        return {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "query_length": len(query),
            "retrieval_top_k": retrieval_top_k,
            "retrieved_candidate_count": len(candidates),
            "top_candidates": [
                self._candidate_to_trace(candidate)
                for candidate in ranked[:answer_top_k]
            ],
            "confidence_threshold": self.confidence_threshold,
            "top1_score": top1_score,
            "confidence_decision": confidence_decision,
            "llm_called": llm_called,
            "final_status": final_status,
            "error_message": error_message,
        }

    def _log_observability_trace(
        self,
        trace_id: str,
        query: str,
        retrieval_top_k: int,
        candidates: list[Candidate],
        ranked: list[Candidate],
        answer_top_k: int,
        top1_score: float,
        confidence_decision: str,
        llm_called: bool,
        final_status: str,
        error_message: str | None = None,
    ) -> None:
        write_trace(
            self._build_observability_trace(
                trace_id=trace_id,
                query=query,
                retrieval_top_k=retrieval_top_k,
                candidates=candidates,
                ranked=ranked,
                answer_top_k=answer_top_k,
                top1_score=top1_score,
                confidence_decision=confidence_decision,
                llm_called=llm_called,
                final_status=final_status,
                error_message=error_message,
            )
        )

    def ask(self, query: str, top_k: int | None = None) -> RAGResponse:
        trace_id = str(uuid4())
        answer_top_k = max(1, top_k or self.answer_top_k_default)
        retrieval_top_k = max(self.retrieval_top_k_default, answer_top_k)
        candidates: list[Candidate] = []
        ranked: list[Candidate] = []
        top_chunks: list[Candidate] = []
        top1_score = 0.0
        llm_called = False

        try:
            query_embedding = self.embedder.embed(query)
            candidates = self.retriever.retrieve(
                query_embedding,
                query,
                top_k=retrieval_top_k,
            )
            ranked = self.reranker.rerank(query, candidates)
            top_chunks = ranked[:answer_top_k]
            top1_score = (
                top_chunks[0].get("score", top_chunks[0]["retrieval_score"])
                if top_chunks
                else 0.0
            )
            trace = self._build_trace(
                trace_id,
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
                self._log_observability_trace(
                    trace_id=trace_id,
                    query=query,
                    retrieval_top_k=retrieval_top_k,
                    candidates=candidates,
                    ranked=ranked,
                    answer_top_k=answer_top_k,
                    top1_score=top1_score,
                    confidence_decision="low_confidence",
                    llm_called=False,
                    final_status="low_confidence",
                )
                return {
                    "trace_id": trace_id,
                    "answer": "I cannot find relevant information in the documents.",
                    "sources": [],
                    "trace": trace,
                    "status": "low_confidence",
                }

            context = "\n\n".join([chunk["document"] for chunk in top_chunks])
            llm_called = True
            answer = self.llm.generate(query, context)

            self._log_observability_trace(
                trace_id=trace_id,
                query=query,
                retrieval_top_k=retrieval_top_k,
                candidates=candidates,
                ranked=ranked,
                answer_top_k=answer_top_k,
                top1_score=top1_score,
                confidence_decision="confident",
                llm_called=True,
                final_status="answered",
            )
            return {
                "trace_id": trace_id,
                "answer": answer,
                "sources": [
                    self._candidate_to_source(chunk)
                    for chunk in top_chunks
                ],
                "trace": trace,
                "status": "answered",
            }
        except Exception as error:
            self._log_observability_trace(
                trace_id=trace_id,
                query=query,
                retrieval_top_k=retrieval_top_k,
                candidates=candidates,
                ranked=ranked,
                answer_top_k=answer_top_k,
                top1_score=top1_score,
                confidence_decision="error",
                llm_called=llm_called,
                final_status="error",
                error_message=str(error)[:300],
            )
            raise
