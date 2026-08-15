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

DocumentStatus = Literal["ACTIVE", "ARCHIVED"]
IngestionResult = Literal["indexed", "duplicate", "version_conflict"]


class StoredDocumentIdentity(TypedDict):
    document_id: str
    version: str
    status: DocumentStatus
    file_hash: str | None
    file_name: str
    created_time: str | None


class IngestionDecision(TypedDict):
    result: IngestionResult
    existing_document: NotRequired[StoredDocumentIdentity]


class ChunkMetadata(TypedDict):
    document_id: str
    source_file: str
    chunk_index: int
    page_number: int | None
    file_name: NotRequired[str]
    version: NotRequired[str]
    status: NotRequired[DocumentStatus]
    created_time: NotRequired[str | None]
    file_hash: NotRequired[str]
    source_snippet: NotRequired[str]


class Chunk(TypedDict):
    document_id: str
    chunk_index: int
    source_file: str
    page_number: int | None
    content: str
    file_name: NotRequired[str]
    version: NotRequired[str]
    status: NotRequired[DocumentStatus]
    created_time: NotRequired[str]
    file_hash: NotRequired[str]
    embedding: NotRequired[list[float]]


class Candidate(TypedDict):
    document: str
    metadata: ChunkMetadata
    embedding_score: float
    bm25_score: float
    retrieval_score: float
    rerank_enabled: NotRequired[bool]
    rerank_score: NotRequired[float | None]


class RetrievalTrace(TypedDict):
    top1_score: float
    top_k_scores: list[float]


class RerankTrace(TypedDict):
    enabled: bool
    improvement: float


class DecisionTrace(TypedDict):
    passed_gate: bool | None
    confidence_score: NotRequired[float]
    fallback_reason: NotRequired[str | None]
    fallback_status: NotRequired[RAGStatus | None]
    sensitive_input_detected: NotRequired[bool]
    out_of_scope_detected: NotRequired[bool]
    conflict_detected: NotRequired[bool]


class RAGTrace(TypedDict):
    trace_id: str
    query: str
    original_query: str
    retrieval_query: str
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

    def count(self) -> int:
        ...

    def find_by_file_hash(
        self,
        file_hash: str,
    ) -> StoredDocumentIdentity | None:
        ...

    def find_by_document_version(
        self,
        document_id: str,
        version: str,
    ) -> StoredDocumentIdentity | None:
        ...

    def archive_document_version(
        self,
        document_id: str,
        version: str,
    ) -> int:
        ...

    def clear(self) -> None:
        ...


EmbeddingModel = Callable[[str], list[float]]
LLMModel = Callable[[str, str], str]
RerankerModel = Callable[[str, list[Candidate]], list[Candidate]]
