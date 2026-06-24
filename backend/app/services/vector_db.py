from math import sqrt

# 当前进程内存中的向量存储
# 服务重启后会清空，后续再单独做持久化。
vector_store = []


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def add_chunks_to_db(chunks):
    """
    将 chunk、embedding 和 metadata 保存到 Python 内存列表。
    不调用 Chroma，因此绕开 Windows native crash。
    """
    for chunk in chunks:
        vector_store.append({
            "id": f"{chunk['document_id']}_{chunk['chunk_index']}",
            "embedding": chunk["embedding"],
            "document": chunk["content"],
            "metadata": {
                "document_id": chunk["document_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk.get("page_number"),
            },
        })

    print(f"Added {len(chunks)} chunks. Total in memory: {len(vector_store)}")


def search(query_embedding, top_k=3):
    """
    返回结构刻意模拟 Chroma query()，
    这样 rag.py 不需要大改。
    """
    scored_results = []

    for item in vector_store:
        similarity = _cosine_similarity(
            query_embedding,
            item["embedding"],
        )

        scored_results.append({
            "document": item["document"],
            "metadata": item["metadata"],
            "similarity": similarity,
        })

    scored_results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    top_results = scored_results[:top_k]

    return {
        "documents": [[item["document"] for item in top_results]],
        "metadatas": [[item["metadata"] for item in top_results]],
        # Chroma 常见是 distance 越小越相近；
        # 这里转换成 1 - similarity，方便后续兼容理解。
        "distances": [[
            1 - item["similarity"]
            for item in top_results
        ]],
    }


def clear_vector_store():
    """
    后续做“重新索引”或测试时可调用。
    """
    vector_store.clear()
    print("In-memory vector store cleared.")