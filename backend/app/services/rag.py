import logging
import re
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
    RAGStatus,
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
OUT_OF_SCOPE_DECISION_PATTERNS = (
    "approve",
    "reject",
    "decide",
    "final decision",
    "customer risk rating",
    "compliance approval",
    "loan approval",
    "should we approve",
)
SENSITIVE_INPUT_PATTERNS = (
    re.compile(r"\b\d{17}[\dXx]\b"),
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b\d{12,19}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"\b(?:api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9_.-]{12,}\b",
        re.I,
    ),
)
ACTIVE_DOCUMENT_STATUSES = {"active", "current", "published", "approved"}
INACTIVE_DOCUMENT_STATUSES = {"deprecated", "expired", "inactive", "archived"}


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

    def _detect_sensitive_input(self, query: str) -> bool:
        return any(pattern.search(query) for pattern in SENSITIVE_INPUT_PATTERNS)

    def _detect_out_of_scope_decision_request(self, query: str) -> bool:
        normalized_query = query.lower()
        return any(
            pattern in normalized_query
            for pattern in OUT_OF_SCOPE_DECISION_PATTERNS
        )

    def _has_usable_evidence(self, candidates: list[Candidate]) -> bool:
        return any(candidate.get("document", "").strip() for candidate in candidates)

    def _detect_conflicting_sources(self, candidates: list[Candidate]) -> bool:
        versions: set[str] = set()
        active_source_count = 0

        for candidate in candidates:
            metadata = candidate["metadata"]
            status = str(metadata.get("document_status", "")).lower()
            if status in INACTIVE_DOCUMENT_STATUSES:
                return True

            version = metadata.get("version")
            if version is not None:
                versions.add(str(version))

            if status in ACTIVE_DOCUMENT_STATUSES or not status:
                active_source_count += 1

        if len(versions) > 1 and active_source_count > 1:
            return True

        # Current chunk metadata does not include version/effective_date/status.
        # Keep this conservative until richer source metadata exists.
        return False

    def _annotate_trace_decision(
        self,
        trace: RAGTrace,
        fallback_reason: str | None,
        fallback_status: RAGStatus | None,
        sensitive_input_detected: bool,
        out_of_scope_detected: bool,
        conflict_detected: bool,
    ) -> None:
        trace["decision"]["fallback_reason"] = fallback_reason
        trace["decision"]["fallback_status"] = fallback_status
        trace["decision"]["sensitive_input_detected"] = sensitive_input_detected
        trace["decision"]["out_of_scope_detected"] = out_of_scope_detected
        trace["decision"]["conflict_detected"] = conflict_detected

    def _build_fallback_response(
        self,
        trace_id: str,
        query: str,
        trace: RAGTrace,
        retrieval_top_k: int,
        candidates: list[Candidate],
        ranked: list[Candidate],
        answer_top_k: int,
        top1_score: float,
        status: RAGStatus,
        fallback_reason: str,
        answer: str,
        sources: list[ChunkMetadata] | None = None,
        sensitive_input_detected: bool = False,
        out_of_scope_detected: bool = False,
        conflict_detected: bool = False,
    ) -> RAGResponse:
        self._annotate_trace_decision(
            trace=trace,
            fallback_reason=fallback_reason,
            fallback_status=status,
            sensitive_input_detected=sensitive_input_detected,
            out_of_scope_detected=out_of_scope_detected,
            conflict_detected=conflict_detected,
        )
        self._log_observability_trace(
            trace_id=trace_id,
            query=query,
            retrieval_top_k=retrieval_top_k,
            candidates=candidates,
            ranked=ranked,
            answer_top_k=answer_top_k,
            top1_score=top1_score,
            confidence_decision=status,
            llm_called=False,
            final_status=status,
            fallback_reason=fallback_reason,
            fallback_status=status,
            sensitive_input_detected=sensitive_input_detected,
            out_of_scope_detected=out_of_scope_detected,
            conflict_detected=conflict_detected,
        )

        return {
            "trace_id": trace_id,
            "answer": answer,
            "sources": sources or [],
            "trace": trace,
            "status": status,
            "fallback_reason": fallback_reason,
        }

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
        fallback_reason: str | None = None,
        fallback_status: RAGStatus | None = None,
        sensitive_input_detected: bool = False,
        out_of_scope_detected: bool = False,
        conflict_detected: bool = False,
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
            "fallback_reason": fallback_reason,
            "fallback_status": fallback_status,
            "sensitive_input_detected": sensitive_input_detected,
            "out_of_scope_detected": out_of_scope_detected,
            "conflict_detected": conflict_detected,
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
        fallback_reason: str | None = None,
        fallback_status: RAGStatus | None = None,
        sensitive_input_detected: bool = False,
        out_of_scope_detected: bool = False,
        conflict_detected: bool = False,
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
                fallback_reason=fallback_reason,
                fallback_status=fallback_status,
                sensitive_input_detected=sensitive_input_detected,
                out_of_scope_detected=out_of_scope_detected,
                conflict_detected=conflict_detected,
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
        sensitive_input_detected = False
        out_of_scope_detected = False
        conflict_detected = False

        try:
            sensitive_input_detected = self._detect_sensitive_input(query)
            out_of_scope_detected = (
                self._detect_out_of_scope_decision_request(query)
            )

            if sensitive_input_detected:
                trace = self._build_trace(
                    trace_id,
                    query,
                    candidates,
                    ranked,
                    top_k=answer_top_k,
                )
                return self._build_fallback_response(
                    trace_id=trace_id,
                    query=query,
                    trace=trace,
                    retrieval_top_k=retrieval_top_k,
                    candidates=candidates,
                    ranked=ranked,
                    answer_top_k=answer_top_k,
                    top1_score=top1_score,
                    status="sensitive_input_detected",
                    fallback_reason="sensitive_input_detected",
                    answer=(
                        "Sensitive information was detected. Please remove or "
                        "redact personal identifiers, account numbers, API keys, "
                        "tokens, or secrets before retrying."
                    ),
                    sensitive_input_detected=True,
                    out_of_scope_detected=out_of_scope_detected,
                    conflict_detected=False,
                )

            if out_of_scope_detected:
                trace = self._build_trace(
                    trace_id,
                    query,
                    candidates,
                    ranked,
                    top_k=answer_top_k,
                )
                return self._build_fallback_response(
                    trace_id=trace_id,
                    query=query,
                    trace=trace,
                    retrieval_top_k=retrieval_top_k,
                    candidates=candidates,
                    ranked=ranked,
                    answer_top_k=answer_top_k,
                    top1_score=top1_score,
                    status="human_review_required",
                    fallback_reason="out_of_scope_decision_request",
                    answer=(
                        "I can retrieve relevant document guidance, but I cannot "
                        "make final business, risk, approval, loan, or compliance "
                        "decisions. Please route this to a qualified human reviewer."
                    ),
                    sensitive_input_detected=False,
                    out_of_scope_detected=True,
                    conflict_detected=False,
                )

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
            conflict_detected = self._detect_conflicting_sources(top_chunks)

            if conflict_detected:
                return self._build_fallback_response(
                    trace_id=trace_id,
                    query=query,
                    trace=trace,
                    retrieval_top_k=retrieval_top_k,
                    candidates=candidates,
                    ranked=ranked,
                    answer_top_k=answer_top_k,
                    top1_score=top1_score,
                    status="conflicting_sources",
                    fallback_reason="conflicting_sources",
                    answer=(
                        "The retrieved documents appear to contain conflicting or "
                        "inactive source metadata. Please review the source "
                        "documents before relying on an answer."
                    ),
                    sources=[
                        self._candidate_to_source(chunk)
                        for chunk in top_chunks
                    ],
                    sensitive_input_detected=sensitive_input_detected,
                    out_of_scope_detected=out_of_scope_detected,
                    conflict_detected=True,
                )

            if candidates and not self._has_usable_evidence(top_chunks):
                return self._build_fallback_response(
                    trace_id=trace_id,
                    query=query,
                    trace=trace,
                    retrieval_top_k=retrieval_top_k,
                    candidates=candidates,
                    ranked=ranked,
                    answer_top_k=answer_top_k,
                    top1_score=top1_score,
                    status="insufficient_evidence",
                    fallback_reason="insufficient_evidence",
                    answer=(
                        "Retrieved candidates were found, but they do not contain "
                        "usable evidence text for a grounded answer."
                    ),
                    sensitive_input_detected=sensitive_input_detected,
                    out_of_scope_detected=out_of_scope_detected,
                    conflict_detected=conflict_detected,
                )

            if not passed_gate:
                return self._build_fallback_response(
                    trace_id=trace_id,
                    query=query,
                    trace=trace,
                    retrieval_top_k=retrieval_top_k,
                    candidates=candidates,
                    ranked=ranked,
                    answer_top_k=answer_top_k,
                    top1_score=top1_score,
                    status="low_confidence",
                    fallback_reason="low_confidence",
                    answer=(
                        "There is not enough evidence in the indexed documents "
                        "to answer this question reliably."
                    ),
                    sensitive_input_detected=sensitive_input_detected,
                    out_of_scope_detected=out_of_scope_detected,
                    conflict_detected=conflict_detected,
                )

            context = "\n\n".join([chunk["document"] for chunk in top_chunks])
            llm_called = True
            answer = self.llm.generate(query, context)
            self._annotate_trace_decision(
                trace=trace,
                fallback_reason=None,
                fallback_status=None,
                sensitive_input_detected=sensitive_input_detected,
                out_of_scope_detected=out_of_scope_detected,
                conflict_detected=conflict_detected,
            )

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
                fallback_reason=None,
                fallback_status=None,
                sensitive_input_detected=sensitive_input_detected,
                out_of_scope_detected=out_of_scope_detected,
                conflict_detected=conflict_detected,
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
                "fallback_reason": None,
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
                fallback_reason="error",
                fallback_status="error",
                sensitive_input_detected=sensitive_input_detected,
                out_of_scope_detected=out_of_scope_detected,
                conflict_detected=conflict_detected,
                error_message=str(error)[:300],
            )
            raise
