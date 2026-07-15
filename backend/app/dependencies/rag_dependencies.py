from functools import lru_cache

from app.config.settings import AppSettings
from app.services.embedding import create_openai_embedding_model
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService, create_openai_llm
from app.services.rag import RAGService
from app.services.rerank_service import RerankService
from app.services.retrieval_service import InMemoryVectorStore, RetrievalService
from app.services.vector_db import configure_retrieval, configure_reranker, rerank


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.from_env()


@lru_cache
def _build_rag_service() -> RAGService:
    settings = get_settings()

    configure_retrieval(
        embedding_weight=settings.embedding_score_weight,
        bm25_weight=settings.bm25_score_weight,
    )
    configure_reranker(
        enabled=settings.reranker_enabled,
        model_name=settings.reranker_model_name,
    )

    embedding_service = EmbeddingService(create_openai_embedding_model(settings))
    retrieval_service = RetrievalService(InMemoryVectorStore())
    rerank_service = RerankService(rerank)
    llm_service = LLMService(create_openai_llm(settings))

    return RAGService(
        embedder=embedding_service,
        retriever=retrieval_service,
        reranker=rerank_service,
        llm=llm_service,
        retrieval_top_k_default=settings.retrieval_top_k_default,
        answer_top_k_default=settings.answer_top_k_default,
        confidence_threshold=settings.confidence_threshold,
    )


def get_rag_service() -> RAGService:
    return _build_rag_service()
