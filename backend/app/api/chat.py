import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.dependencies.rag_dependencies import get_rag_service
from app.services.errors import ServiceConfigurationError
from app.services.rag import RAGService

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceMetadata(BaseModel):
    document_id: str | None = None
    source_file: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    source_snippet: str | None = None


class ChatResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    sources: list[SourceMetadata]
    trace: dict[str, Any]
    status: str
    fallback_reason: str | None = None


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        result = rag_service.ask(request.question, top_k=request.top_k)
    except ServiceConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI request failed: {error.__class__.__name__}",
        ) from error
    except Exception as error:
        logger.exception("chat_request_failed")
        raise HTTPException(
            status_code=500,
            detail="Chat service failed to generate an answer.",
        ) from error

    return ChatResponse(
        trace_id=result["trace_id"],
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        trace=result["trace"],
        status=result["status"],
        fallback_reason=result.get("fallback_reason"),
    )
