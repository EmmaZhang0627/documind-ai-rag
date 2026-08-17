from datetime import datetime, timezone


def deterministic_parent_id(
    document_id: str,
    version: str,
    page_number: int,
    start_char: int,
    end_char: int,
) -> str:
    return f"{document_id}:{version}:page-{page_number}:parent-{start_char}-{end_char}"


def _document_metadata(
    source_file: str,
    version: str,
    status: str,
    created_time: str | None,
    file_hash: str | None,
    tenant_id: str | None = None,
    department: str | None = None,
    access_level: str | None = None,
) -> dict[str, str]:
    metadata = {
        "source_file": source_file,
        "file_name": source_file,
        "version": version,
        "status": status,
        "created_time": created_time or datetime.now(timezone.utc).isoformat(),
    }
    if file_hash is not None:
        metadata["file_hash"] = file_hash
    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id
    if department is not None:
        metadata["department"] = department
    if access_level is not None:
        metadata["access_level"] = access_level
    return metadata


def split_text_into_chunks(
    text: str,
    document_id: str,
    source_file: str,
    page_number: int,
    chunk_size: int = 800,
    overlap: int = 100,
    version: str = "1",
    status: str = "ACTIVE",
    created_time: str | None = None,
    file_hash: str | None = None,
    tenant_id: str | None = None,
    department: str | None = None,
    access_level: str | None = None,
):
    chunks = []
    document_metadata = _document_metadata(
        source_file, version, status, created_time, file_hash,
        tenant_id, department, access_level,
    )

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "document_id": document_id,
            "chunk_index": len(chunks),
            "content": chunk_text,
            **document_metadata,
            "page_number": page_number,
            "start_char": start,
            "end_char": min(end, text_length),
        })

        start = end - overlap

    return chunks


def split_pages_into_chunks(
    pages: list[dict],
    document_id: str,
    source_file: str,
    chunk_size: int = 800,
    overlap: int = 100,
    version: str = "1",
    status: str = "ACTIVE",
    created_time: str | None = None,
    file_hash: str | None = None,
    tenant_id: str | None = None,
    department: str | None = None,
    access_level: str | None = None,
):
    chunks = []
    document_metadata = _document_metadata(
        source_file, version, status, created_time, file_hash,
        tenant_id, department, access_level,
    )
    global_chunk_index = 0
    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        start = 0
        text_length = len(page_text)

        while start < text_length:
            end = start + chunk_size
            chunk_text = page_text[start:end]

            # 跳过只有空白字符的页面或 chunk
            if chunk_text.strip():
                chunks.append({
                    "document_id": document_id,
                    "chunk_index": global_chunk_index,
                    "content": chunk_text,
                    **document_metadata,
                    "page_number": page_number,
                    "start_char": start,
                    "end_char": min(end, text_length),
                })
            global_chunk_index += 1
            start = end - overlap

    return chunks


def split_pages_into_parent_child_chunks(
    pages: list[dict],
    document_id: str,
    source_file: str,
    parent_size: int = 1600,
    child_size: int = 600,
    child_overlap: int = 100,
    version: str = "1",
    status: str = "ACTIVE",
    created_time: str | None = None,
    file_hash: str | None = None,
    tenant_id: str | None = None,
    department: str | None = None,
    access_level: str | None = None,
):
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("Parent and child chunk sizes must be positive.")
    if child_size > parent_size:
        raise ValueError("Child chunk size must not exceed parent chunk size.")
    if child_overlap < 0 or child_overlap >= child_size:
        raise ValueError("Child overlap must be between zero and child size.")

    chunks = []
    document_metadata = _document_metadata(
        source_file, version, status, created_time, file_hash,
        tenant_id, department, access_level,
    )
    global_chunk_index = 0
    child_step = child_size - child_overlap

    for page in pages:
        page_number = int(page["page_number"])
        page_text = page["text"]
        for parent_start in range(0, len(page_text), parent_size):
            parent_end = min(parent_start + parent_size, len(page_text))
            parent_text = page_text[parent_start:parent_end]
            if not parent_text.strip():
                continue
            parent_id = deterministic_parent_id(
                document_id, version, page_number, parent_start, parent_end
            )
            child_index = 0
            for relative_start in range(0, len(parent_text), child_step):
                relative_end = min(relative_start + child_size, len(parent_text))
                child_text = parent_text[relative_start:relative_end]
                if child_text.strip():
                    chunks.append({
                        "document_id": document_id,
                        "chunk_index": global_chunk_index,
                        "content": child_text,
                        **document_metadata,
                        "page_number": page_number,
                        "start_char": parent_start + relative_start,
                        "end_char": parent_start + relative_end,
                        "parent_id": parent_id,
                        "child_index": child_index,
                        "parent_text": parent_text,
                        "parent_start_char": parent_start,
                        "parent_end_char": parent_end,
                    })
                    global_chunk_index += 1
                    child_index += 1
                if relative_end >= len(parent_text):
                    break

    return chunks
