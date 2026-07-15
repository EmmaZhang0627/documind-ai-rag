from typing import Literal, NotRequired, TypedDict


ConfidenceDecision = Literal["confident", "low_confidence", "error"]
FinalStatus = Literal["answered", "low_confidence", "error"]


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
    error_message: str | None
