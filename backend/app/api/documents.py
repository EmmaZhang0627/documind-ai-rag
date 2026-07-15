from pathlib import Path
from uuid import uuid4
import logging

from app.services.chunker import split_pages_into_chunks
from app.dependencies.rag_dependencies import get_rag_service
from app.services.errors import ServiceConfigurationError
from app.services.rag import RAGService
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError
import fitz

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_extension = Path(file.filename).suffix
    document_id = str(uuid4())
    saved_file_name = f"{document_id}{file_extension}"
    saved_path = UPLOAD_DIR / saved_file_name

    content = await file.read()
    saved_path.write_bytes(content)

    return {
        "document_id": document_id,
        "original_file_name": file.filename,
        "saved_file_name": saved_file_name,
        "file_size": len(content),
        "status": "uploaded",
    }

@router.post("/parse-pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
):

    # 1. 只允许 PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    # 2. 读取文件
    content = await file.read()

    # 3. 临时保存（必须，否则 fitz 才能打开）
    temp_path = UPLOAD_DIR / f"{uuid4()}.pdf"
    temp_path.write_bytes(content)

    # 4. 用 fitz 打开
    doc = fitz.open(temp_path)

    # 5. 提取文本
    # text = ""
    # for page in doc:
    #     text += page.get_text()
    # document_id = str(uuid4())

    # chunks = split_text_into_chunks(
    #     text=text,
    #     document_id=document_id,
    #     source_file=file.filename,
    # )

    # 5. 按页提取文本，并保留页码
    pages = []
    full_text = ""

    for page_index, page in enumerate(doc):
        page_text = page.get_text()
        if page_text.strip():
            pages.append({
                "page_number": page_index + 1,
                "text": page_text,
            })

        full_text += page_text

    document_id = str(uuid4())

    # 6. 按页切块，使每个 chunk 保留 page_number
    chunks = split_pages_into_chunks(
        pages=pages,
        document_id=document_id,
        source_file=file.filename,
    )
    try:
        await run_in_threadpool(rag_service.ingest_document, chunks)
    except ServiceConfigurationError as error:
        doc.close()
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OpenAIError as error:
        doc.close()
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI embedding failed: {error.__class__.__name__}",
        ) from error
    except Exception as error:
        doc.close()
        logger.exception("document_ingestion_failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from error

    print("chunks added to vector store")
    chunks_preview = [
        {key: value for key, value in chunk.items() if key != "embedding"}
        for chunk in chunks[:3]
    ]
    doc.close()
    # 6. 返回 preview
    return {
        "document_id": document_id,
        "source_file": file.filename,
        "text_preview": full_text[:1500],
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunks_preview": chunks_preview,
        "status": "parsed_chunked_and_indexed",
    }



       
