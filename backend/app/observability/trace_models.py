from typing import Literal, NotRequired, TypedDict


ConfidenceDecision = Literal[
    "confident",
    "low_confidence",
    "insufficient_evidence",
    "human_review_required",
    "out_of_scope",
    "sensitive_input_detected",
    "conflicting_sources",
    "error",
]
FinalStatus = Literal[
    "answered",
    "low_confidence",
    "insufficient_evidence",
    "conflicting_sources",
    "out_of_scope",
    "sensitive_input_detected",
    "human_review_required",
    "error",
]


class TraceCandidate(TypedDict, total=False):
    document_id: str | None
    source_file: str | None
    page_number: int | None
    chunk_index: int | None
    embedding_score: float | None
    bm25_score: float | None
    final_score: float | None
    rerank_score: NotRequired[float | None]
    snippet: NotRequired[str]


class RAGTraceRecord(TypedDict):
    trace_id: str
    timestamp: str
    query: str
    query_length: int
    retrieval_top_k: int
    retrieved_candidate_count: int
    top_candidates: list[TraceCandidate]
    confidence_threshold: float
    top1_score: float
    confidence_decision: ConfidenceDecision
    llm_called: bool
    final_status: FinalStatus
    fallback_reason: str | None
    fallback_status: FinalStatus | None
    sensitive_input_detected: bool
    out_of_scope_detected: bool
    conflict_detected: bool
    error_message: str | None
