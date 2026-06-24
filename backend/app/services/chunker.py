def split_text_into_chunks(
    text: str,
    document_id: str,
    source_file: str,
    page_number: int,
    chunk_size: int = 800,
    overlap: int = 100,
):
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "document_id": document_id,
            "chunk_index": len(chunks),
            "content": chunk_text,
            "source_file": source_file,
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
):
    chunks = []
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
                    "source_file": source_file,
                    "page_number": page_number,
                    "start_char": start,
                    "end_char": min(end, text_length),
                })
            global_chunk_index += 1
            start = end - overlap

    return chunks
