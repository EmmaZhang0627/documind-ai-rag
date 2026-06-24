from app.api.documents import router as documents_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router

app = FastAPI(
    title="DocuMind API",
    description="Backend API for DocuMind RAG document Q&A system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "DocuMind API",
        "version": "0.1.0"
    }
