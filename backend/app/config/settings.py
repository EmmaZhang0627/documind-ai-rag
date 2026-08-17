import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_CHROMA_PERSIST_DIRECTORY = str(BACKEND_ROOT / "data" / "chroma")
DEFAULT_OPENAI_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_TESSDATA_DIRECTORY = str(BACKEND_ROOT / "app" / "assets" / "tessdata")

load_dotenv(PROJECT_ROOT / ".env.local", override=False)


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
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    embedding_model_name: str = "text-embedding-3-small"
    chat_model_name: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 100.0

    query_rewrite_enabled: bool = True
    parent_child_retrieval_enabled: bool = False
    parent_chunk_size: int = 1600
    child_chunk_size: int = 600
    child_chunk_overlap: int = 100

    ocr_fallback_enabled: bool = False
    table_aware_ingestion_enabled: bool = False
    ocr_minimum_text_characters: int = 200
    ocr_minimum_image_coverage: float = 0.5
    ocr_minimum_image_count: int = 3
    ocr_language: str = "eng"
    ocr_dpi: int = 150
    tessdata_directory: str = DEFAULT_TESSDATA_DIRECTORY

    retrieval_top_k_default: int = 10
    answer_top_k_default: int = 3
    confidence_threshold: float = 0.6

    embedding_score_weight: float = 0.7
    bm25_score_weight: float = 0.3

    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    vector_store_backend: str = "chroma"
    chroma_persist_directory: str = DEFAULT_CHROMA_PERSIST_DIRECTORY
    chroma_collection_name: str = "documind_chunks"

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            openai_api_key=(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("api_key")
            ),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL",
                cls.openai_base_url,
            ).rstrip("/"),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                cls.embedding_model_name,
            ),
            chat_model_name=os.getenv("CHAT_MODEL_NAME", cls.chat_model_name),
            openai_timeout_seconds=_get_float(
                "OPENAI_TIMEOUT_SECONDS",
                cls.openai_timeout_seconds,
            ),
            query_rewrite_enabled=_get_bool(
                "QUERY_REWRITE_ENABLED",
                cls.query_rewrite_enabled,
            ),
            parent_child_retrieval_enabled=_get_bool(
                "PARENT_CHILD_RETRIEVAL_ENABLED",
                cls.parent_child_retrieval_enabled,
            ),
            parent_chunk_size=_get_int(
                "PARENT_CHUNK_SIZE", cls.parent_chunk_size
            ),
            child_chunk_size=_get_int(
                "CHILD_CHUNK_SIZE", cls.child_chunk_size
            ),
            child_chunk_overlap=_get_int(
                "CHILD_CHUNK_OVERLAP", cls.child_chunk_overlap
            ),
            ocr_fallback_enabled=_get_bool(
                "OCR_FALLBACK_ENABLED", cls.ocr_fallback_enabled
            ),
            table_aware_ingestion_enabled=_get_bool(
                "TABLE_AWARE_INGESTION_ENABLED", cls.table_aware_ingestion_enabled
            ),
            ocr_minimum_text_characters=_get_int(
                "OCR_MINIMUM_TEXT_CHARACTERS", cls.ocr_minimum_text_characters
            ),
            ocr_minimum_image_coverage=_get_float(
                "OCR_MINIMUM_IMAGE_COVERAGE", cls.ocr_minimum_image_coverage
            ),
            ocr_minimum_image_count=_get_int(
                "OCR_MINIMUM_IMAGE_COUNT", cls.ocr_minimum_image_count
            ),
            ocr_language=os.getenv("OCR_LANGUAGE", cls.ocr_language),
            ocr_dpi=_get_int("OCR_DPI", cls.ocr_dpi),
            tessdata_directory=os.getenv(
                "TESSDATA_DIRECTORY", cls.tessdata_directory
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
            vector_store_backend=os.getenv(
                "VECTOR_STORE_BACKEND",
                cls.vector_store_backend,
            ).strip().lower(),
            chroma_persist_directory=os.getenv(
                "CHROMA_PERSIST_DIRECTORY",
                cls.chroma_persist_directory,
            ),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME",
                cls.chroma_collection_name,
            ),
        )
