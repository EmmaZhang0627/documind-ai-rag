import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    return int(value)


@dataclass(frozen=True)
class AppSettings:
    openai_api_key: str | None = None
    embedding_model_name: str = "text-embedding-3-small"
    chat_model_name: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 100.0

    retrieval_top_k_default: int = 10
    answer_top_k_default: int = 3
    confidence_threshold: float = 0.6

    embedding_score_weight: float = 0.7
    bm25_score_weight: float = 0.3

    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                cls.embedding_model_name,
            ),
            chat_model_name=os.getenv("CHAT_MODEL_NAME", cls.chat_model_name),
            openai_timeout_seconds=_get_float(
                "OPENAI_TIMEOUT_SECONDS",
                cls.openai_timeout_seconds,
            ),
            retrieval_top_k_default=_get_int(
                "RETRIEVAL_TOP_K_DEFAULT",
                cls.retrieval_top_k_default,
            ),
            answer_top_k_default=_get_int(
                "ANSWER_TOP_K_DEFAULT",
                cls.answer_top_k_default,
            ),
            confidence_threshold=_get_float(
                "CONFIDENCE_THRESHOLD",
                cls.confidence_threshold,
            ),
            embedding_score_weight=_get_float(
                "EMBEDDING_SCORE_WEIGHT",
                cls.embedding_score_weight,
            ),
            bm25_score_weight=_get_float(
                "BM25_SCORE_WEIGHT",
                cls.bm25_score_weight,
            ),
            reranker_enabled=_get_bool("RERANKER_ENABLED", cls.reranker_enabled),
            reranker_model_name=os.getenv(
                "RERANKER_MODEL_NAME",
                cls.reranker_model_name,
            ),
        )
