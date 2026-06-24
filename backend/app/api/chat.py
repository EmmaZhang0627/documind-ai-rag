from pydantic import BaseModel
from fastapi import APIRouter

from app.services.rag import ask_question

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


@router.post("")
def chat(request: ChatRequest):
    result = ask_question(request.question)

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "status": "answered"
    }