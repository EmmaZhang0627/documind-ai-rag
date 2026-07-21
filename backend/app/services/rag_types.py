from collections.abc import Callable
from typing import Literal, NotRequired, Protocol, TypedDict


RAGStatus = Literal[
    "answered",
    "low_confidence",
    "insufficient_evidence",
    "conflicting_sources",
    "out_of_scope",
    "sensitive_input_detected",
    "human_review_required",
    "error",
]


class ChunkMetadata(TypedDict):
    document_id: str
    source_file: str
    chunk_index: int
    page_number: int | None
    source_snippet: NotRequired[str]


class Chunk(TypedDict):
    document_id: str
    chunk_index: int
    source_file: str
    page_number: int | None
    content: str
    embedding: NotRequired[list[float]]


class Candidate(TypedDict):
    document: str
    metadata: ChunkMetadata
    embedding_score: float
    bm25_score: float
    retrieval_score: float
    rerank_enabled: NotRequired[bool]
    rerank_score: NotRequired[float]
    score: NotRequired[float]


class RetrievalTrace(TypedDict):
    top1_score: float
    top_k_scores: list[float]


class RerankTrace(TypedDict):
    enabled: bool
    improvement: float


class DecisionTrace(TypedDict):
    passed_gate: bool | None
    fallback_reason: NotRequired[str | None]
    fallback_status: NotRequired[RAGStatus | None]
    sensitive_input_detected: NotRequired[bool]
    out_of_scope_detected: NotRequired[bool]
    conflict_detected: NotRequired[bool]


class RAGTrace(TypedDict):
    trace_id: str
    query: str
    retrieval: RetrievalTrace
    rerank: RerankTrace
    decision: DecisionTrace


class RAGResponse(TypedDict):
    trace_id: str
    answer: str
    sources: list[ChunkMetadata]
    trace: RAGTrace
    status: RAGStatus
    fallback_reason: NotRequired[str | None]


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None:
        ...

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        ...


EmbeddingModel = Callable[[str], list[float]]
LLMModel = Callable[[str, str], str]
RerankerModel = Callable[[str, list[Candidate]], list[Candidate]]
