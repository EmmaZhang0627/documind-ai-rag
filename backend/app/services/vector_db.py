import logging
from math import sqrt
from uuid import uuid4

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

vector_store = []
corpus = []
corpus_ids = []
bm25_model = None
reranker = None
reranker_load_error = None
embedding_score_weight = 0.7
bm25_score_weight = 0.3
reranker_enabled = True
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def configure_retrieval(embedding_weight: float, bm25_weight: float):
    global embedding_score_weight, bm25_score_weight

    embedding_score_weight = embedding_weight
    bm25_score_weight = bm25_weight


def configure_reranker(enabled: bool, model_name: str):
    global reranker, reranker_enabled, reranker_load_error, reranker_model_name

    if model_name != reranker_model_name:
        reranker = None
        reranker_load_error = None

    reranker_enabled = enabled
    reranker_model_name = model_name


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _normalize_scores(scores):
    if len(scores) == 0:
        return []

    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]

    return [float(score) / float(max_score) for score in scores]


def _is_active_metadata(metadata):
    status = metadata.get("status") or metadata.get("document_status")
    return status is None or str(status).upper() == "ACTIVE"


def _get_reranker():
    global reranker, reranker_load_error

    if not reranker_enabled:
        return None

    if reranker is not None:
        return reranker

    if reranker_load_error is not None:
        return None

    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(reranker_model_name)
        return reranker
    except Exception as error:
        reranker_load_error = error
        logger.warning("CrossEncoder reranker unavailable: %s", error)
        return None


def _fallback_rerank(candidates: list):
    for item in candidates:
        item["rerank_enabled"] = False
        item["rerank_score"] = None

    return sorted(candidates, key=lambda item: item["retrieval_score"], reverse=True)


def rerank(query_text: str, candidates: list):
    if not candidates:
        return []

    model = _get_reranker()
    if model is None:
        return _fallback_rerank(candidates)

    pairs = [(query_text, item["document"]) for item in candidates]

    try:
        scores = model.predict(pairs)
    except Exception as error:
        logger.warning("CrossEncoder rerank failed: %s", error)
        return _fallback_rerank(candidates)

    for index, item in enumerate(candidates):
        item["rerank_enabled"] = True
        item["rerank_score"] = float(scores[index])

    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def _rebuild_bm25_index():
    global bm25_model, corpus, corpus_ids

    active_items = [
        item for item in vector_store if _is_active_metadata(item["metadata"])
    ]
    corpus_ids = [item["id"] for item in active_items]
    corpus = [item["document"] for item in active_items]
    tokenized_corpus = [_tokenize(document) for document in corpus]
    bm25_model = BM25Okapi(tokenized_corpus) if tokenized_corpus else None


def add_chunks_to_db(chunks):
    for chunk in chunks:
        version = chunk.get("version", "1")
        status = str(chunk.get("status", "ACTIVE")).upper()
        if status not in {"ACTIVE", "ARCHIVED"}:
            raise ValueError("Document status must be ACTIVE or ARCHIVED.")
        record_id = f"{chunk['document_id']}:{version}:{chunk['chunk_index']}"
        record = {
            "id": record_id,
            "embedding": chunk["embedding"],
            "document": chunk["content"],
            "metadata": {
                "document_id": chunk["document_id"],
                "source_file": chunk["source_file"],
                "file_name": chunk.get("file_name", chunk["source_file"]),
                "version": version,
                "status": status,
                "created_time": chunk.get("created_time"),
                "file_hash": chunk.get("file_hash"),
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk.get("page_number"),
                "parent_id": chunk.get("parent_id"),
                "child_index": chunk.get("child_index"),
                "parent_text": chunk.get("parent_text"),
                "parent_start_char": chunk.get("parent_start_char"),
                "parent_end_char": chunk.get("parent_end_char"),
            },
        }
        existing_index = next(
            (
                index
                for index, item in enumerate(vector_store)
                if item["id"] == record_id
            ),
            None,
        )
        if existing_index is None:
            vector_store.append(record)
        else:
            vector_store[existing_index] = record
    _rebuild_bm25_index()
    print(f"Added {len(chunks)} chunks. Total in memory: {len(vector_store)}")


def archive_document_version(document_id: str, version: str) -> int:
    matched_count = 0
    for item in vector_store:
        metadata = item["metadata"]
        if (
            metadata.get("document_id") == document_id
            and str(metadata.get("version", "1")) == version
        ):
            metadata["status"] = "ARCHIVED"
            matched_count += 1

    if matched_count:
        _rebuild_bm25_index()
    return matched_count


def retrieve_candidates(query_embedding, query_text: str = ""):
    bm25_scores = []
    if bm25_model is not None and query_text.strip():
        bm25_scores = _normalize_scores(bm25_model.get_scores(_tokenize(query_text)))
    else:
        bm25_scores = [0.0 for _ in corpus_ids]

    bm25_by_document = {
        record_id: float(bm25_scores[index])
        for index, record_id in enumerate(corpus_ids)
    }

    candidates = []

    for index, item in enumerate(vector_store):
        if not _is_active_metadata(item["metadata"]):
            continue
        embedding_score = _cosine_similarity(
            query_embedding,
            item["embedding"],
        )
        bm25_score = bm25_by_document.get(item["id"], 0.0)
        retrieval_score = (
            embedding_score_weight * embedding_score
            + bm25_score_weight * bm25_score
        )

        candidates.append({
            "document": item["document"],
            "metadata": item["metadata"],
            "embedding_score": embedding_score,
            "bm25_score": bm25_score,
            "retrieval_score": retrieval_score,
        })

    return sorted(
        candidates,
        key=lambda item: item["retrieval_score"],
        reverse=True,
    )


def search(query_embedding, query_text: str = "", top_k=3):
    trace_id = str(uuid4())
    before_rerank_results = retrieve_candidates(query_embedding, query_text)
    before_rerank_score = (
        before_rerank_results[0]["retrieval_score"]
        if before_rerank_results
        else 0.0
    )
    retrieval_top_results = before_rerank_results[:top_k]

    reranked_results = rerank(query_text, before_rerank_results)
    top_results = reranked_results[:top_k]
    confidence_score = top_results[0]["retrieval_score"] if top_results else 0.0
    rerank_improvement = confidence_score - before_rerank_score
    rerank_enabled = top_results[0].get("rerank_enabled", False) if top_results else False

    trace = {
        "trace_id": trace_id,
        "query": query_text,
        "retrieval": {
            "top1_score": before_rerank_score,
            "top_k_scores": [
                item["retrieval_score"]
                for item in retrieval_top_results
            ],
        },
        "rerank": {
            "enabled": rerank_enabled,
            "improvement": rerank_improvement,
        },
        "decision": {
            "passed_gate": None,
        },
    }

    logger.info("rag_trace_retrieval=%s", trace)

    return {
        "trace": trace,
        "documents": [[item["document"] for item in top_results]],
        "metadatas": [[item["metadata"] for item in top_results]],
        # Legacy search callers receive stable hybrid retrieval scores here.
        "scores": [[item["retrieval_score"] for item in top_results]],
        "retrieval_scores": [[item["retrieval_score"] for item in top_results]],
        "rerank_scores": [[item["rerank_score"] for item in top_results]],
        "embedding_scores": [[item["embedding_score"] for item in top_results]],
        "bm25_scores": [[item["bm25_score"] for item in top_results]],
        "top1_score": confidence_score,
        "confidence_score": confidence_score,
    }


def is_confident(score: float, threshold: float = 0.6) -> bool:
    return score >= threshold


def clear_vector_store():
    global bm25_model

    vector_store.clear()
    corpus.clear()
    corpus_ids.clear()
    bm25_model = None
    print("In-memory vector store cleared.")
