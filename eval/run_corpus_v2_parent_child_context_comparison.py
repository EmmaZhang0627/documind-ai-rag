from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
RESULTS_PATH = EVAL_DIR / "corpus_v2_parent_child_context_comparison_latest.json"
CASES_PATH = EVAL_DIR / "corpus_v2_eval_cases.json"
CATALOG_PATH = EVAL_DIR / "corpus_v2_evidence_catalog.json"
MANIFEST_PATH = EVAL_DIR / "corpus_v2_manifest.json"
FLAT_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_audit"
FLAT_COLLECTION = "documind_corpus_v2_ingestion_audit"
PARENT_CHILD_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_parent_child"
PARENT_CHILD_COLLECTION = "documind_corpus_v2_parent_child_eval"
MULTI_CASE_IDS = (
    "v2_case_007_study_duration_and_maximum",
    "v2_case_015_appeal_stage_timelines",
    "v2_case_023_integrity_categories_and_appeal",
    "v2_case_030_combined_identifier_rules",
    "v2_case_046_cancellation_definition_effect",
    "v2_case_054_startup_and_cold_start",
)
SINGLE_CASE_IDS = (
    "v2_case_001_study_duration",
    "v2_case_008_appeal_definition",
    "v2_case_024_constant_names",
    "v2_case_031_apollo_memory",
    "v2_case_039_cancellation_period",
    "v2_case_047_cold_start_cost",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_backend_path() -> None:
    backend = PROJECT_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


def build_service(persist_directory: Path, collection_name: str, parent_child: bool):
    ensure_backend_path()
    from app.config.settings import AppSettings
    from app.dependencies.rag_dependencies import build_rag_service

    settings = replace(
        AppSettings.from_env(),
        vector_store_backend="chroma",
        chroma_persist_directory=str(persist_directory),
        chroma_collection_name=collection_name,
        query_rewrite_enabled=False,
        parent_child_retrieval_enabled=parent_child,
    )
    return build_rag_service(settings), settings


def stop_service(service: Any) -> None:
    from chromadb.api.shared_system_client import SharedSystemClient

    try:
        service.retriever.vector_store.client._system.stop()
    except KeyError:
        pass
    SharedSystemClient.clear_system_cache()


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    import fitz

    document = fitz.open(pdf_path)
    try:
        return [
            {"page_number": index + 1, "text": page.get_text()}
            for index, page in enumerate(document)
            if page.get_text().strip()
        ]
    finally:
        document.close()


def ingest_parent_child_corpus(service: Any, settings: Any) -> dict[str, Any]:
    from app.services.chunker import split_pages_into_parent_child_chunks

    manifest = load_json(MANIFEST_PATH)
    included = [item for item in manifest["documents"] if item["include_in_text_retrieval"]]
    if service.retriever.count():
        records = service.retriever.vector_store.collection.get(include=["metadatas"])
        metadatas = records.get("metadatas") or []
        if not metadatas or not all(
            (metadata or {}).get("parent_id") and (metadata or {}).get("parent_text")
            for metadata in metadatas
        ):
            raise ValueError("Existing parent-child collection has invalid metadata.")
        child_counts = Counter(str(metadata["document_id"]) for metadata in metadatas)
        parent_ids_by_document: dict[str, set[str]] = {}
        for metadata in metadatas:
            parent_ids_by_document.setdefault(str(metadata["document_id"]), set()).add(
                str(metadata["parent_id"])
            )
        expected_ids = {entry["document_id"] for entry in included}
        if set(child_counts) != expected_ids:
            raise ValueError("Existing parent-child collection document IDs are incomplete.")
        return {
            "document_count": len(included),
            "parent_count": sum(len(values) for values in parent_ids_by_document.values()),
            "child_count": len(metadatas),
            "reused_persisted_collection": True,
            "per_document": [
                {
                    "document_id": entry["document_id"],
                    "parent_count": len(parent_ids_by_document[entry["document_id"]]),
                    "child_count": child_counts[entry["document_id"]],
                }
                for entry in included
            ],
        }
    service.retriever.clear()
    per_document = []
    parent_ids: set[str] = set()
    for index, entry in enumerate(included, start=1):
        pdf_path = (PROJECT_ROOT / entry["file_path"]).resolve()
        chunks = split_pages_into_parent_child_chunks(
            pages=extract_pages(pdf_path),
            document_id=entry["document_id"],
            source_file=entry["file_name"],
            parent_size=settings.parent_chunk_size,
            child_size=settings.child_chunk_size,
            child_overlap=settings.child_chunk_overlap,
            version=str(entry["version"]),
            status=str(entry["status"]).upper(),
            file_hash=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        )
        service.ingest_document(chunks)
        document_parents = {chunk["parent_id"] for chunk in chunks}
        parent_ids.update(document_parents)
        per_document.append({
            "document_id": entry["document_id"],
            "parent_count": len(document_parents),
            "child_count": len(chunks),
        })
        print(f"[ingest {index}/{len(included)}] {entry['document_id']} children={len(chunks)}", flush=True)
    return {
        "document_count": len(included),
        "parent_count": len(parent_ids),
        "child_count": service.retriever.count(),
        "reused_persisted_collection": False,
        "per_document": per_document,
    }


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def evidence_components(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components = []
    for item in items:
        components.append({
            "component": "canonical",
            "phrases": item["evidence_keywords_or_phrases"],
        })
        for index, supporting in enumerate(item["supporting_chunk_refs"], start=1):
            components.append({
                "component": f"supporting_{index}",
                "phrases": supporting.get("identifying_phrases") or [],
            })
    return components


def covers(text: str, components: list[dict[str, Any]]) -> bool:
    normalized = normalize(text)
    return all(
        component["phrases"]
        and all(normalize(phrase) in normalized for phrase in component["phrases"])
        for component in components
    )


def cumulative_evidence_rank(candidates: list[dict[str, Any]], components: list[dict[str, Any]]) -> int | None:
    text_parts = []
    for rank, candidate in enumerate(candidates, start=1):
        text_parts.append(candidate.get("document", ""))
        if covers(" ".join(text_parts), components):
            return rank
    return None


def context_summary(
    selected_children: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    context_text = "\n\n".join(candidate["document"] for candidate in context_candidates)
    return {
        "selected_child_count": len(selected_children),
        "context_unit_count": len(context_candidates),
        "duplicate_parent_reduction": len(selected_children) - len(context_candidates),
        "context_character_count": len(context_text),
        "context_token_proxy": round(len(context_text) / 4),
        "all_required_evidence_in_context": covers(context_text, components),
        "context_identities": [
            {
                "document_id": (candidate.get("metadata") or {}).get("document_id"),
                "page_number": (candidate.get("metadata") or {}).get("page_number"),
                "chunk_index": (candidate.get("metadata") or {}).get("chunk_index"),
                "parent_id": (candidate.get("metadata") or {}).get("parent_id"),
            }
            for candidate in context_candidates
        ],
    }


def retrieve(service: Any, query: str, retrieval_top_k: int) -> list[dict[str, Any]]:
    embedding = service.embedder.embed(query)
    candidates = service.retriever.retrieve(embedding, query, top_k=retrieval_top_k)
    reranked = service.reranker.rerank(query, deepcopy(candidates))
    if reranked and not all(candidate.get("rerank_enabled") for candidate in reranked):
        raise RuntimeError("CrossEncoder fallback occurred.")
    return reranked


def run() -> int:
    ensure_backend_path()
    from app.services.parent_child_retrieval import resolve_parent_context

    case_by_id = {case["case_id"]: case for case in load_json(CASES_PATH) if case["grounded"]}
    cases = [case_by_id[value] for value in (*MULTI_CASE_IDS, *SINGLE_CASE_IDS)]
    catalog = load_json(CATALOG_PATH)
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence_items"]}
    flat_service, flat_settings = build_service(FLAT_DIRECTORY, FLAT_COLLECTION, False)
    parent_service = None
    results = []
    try:
        if flat_service.retriever.count() != 425:
            raise ValueError("Flat Corpus V2 collection must contain 425 chunks.")
        parent_service, parent_settings = build_service(
            PARENT_CHILD_DIRECTORY, PARENT_CHILD_COLLECTION, True
        )
        ingestion = ingest_parent_child_corpus(parent_service, parent_settings)
        for index, case in enumerate(cases, start=1):
            components = evidence_components(
                [evidence_by_id[evidence_id] for evidence_id in case["evidence_ids"]]
            )
            query = case["question"]
            flat_ranked = retrieve(flat_service, query, flat_service.retrieval_top_k_default)
            parent_ranked = retrieve(parent_service, query, parent_service.retrieval_top_k_default)
            answer_top_k = int(case["top_k"])
            flat_selected = flat_ranked[:answer_top_k]
            parent_selected = parent_ranked[:answer_top_k]
            parent_context = resolve_parent_context(parent_selected)
            results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "question": query,
                "evidence_components": components,
                "flat": {
                    "reranked_evidence_rank": cumulative_evidence_rank(flat_ranked, components),
                    "required_evidence_available_top10": covers(
                        " ".join(item["document"] for item in flat_ranked), components
                    ),
                    "context": context_summary(flat_selected, flat_selected, components),
                },
                "parent_child": {
                    "reranked_child_evidence_rank": cumulative_evidence_rank(parent_ranked, components),
                    "required_evidence_available_top10": covers(
                        " ".join(item["document"] for item in parent_ranked), components
                    ),
                    "context": context_summary(parent_selected, parent_context, components),
                },
                "final_answer_correctness": "not_run_no_answer_labels",
            })
            print(f"[compare {index}/{len(cases)}] {case['case_id']}", flush=True)
    finally:
        if parent_service is not None:
            stop_service(parent_service)
        stop_service(flat_service)

    def aggregate(group: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
        context_chars = [item[strategy]["context"]["context_character_count"] for item in group]
        return {
            "case_count": len(group),
            "required_evidence_available_top10_count": sum(
                item[strategy]["required_evidence_available_top10"] for item in group
            ),
            "all_required_evidence_in_context_count": sum(
                item[strategy]["context"]["all_required_evidence_in_context"] for item in group
            ),
            "average_context_character_count": sum(context_chars) / len(context_chars),
            "average_context_token_proxy": sum(round(value / 4) for value in context_chars) / len(context_chars),
            "total_duplicate_parent_reduction": sum(
                item[strategy]["context"]["duplicate_parent_reduction"] for item in group
            ),
        }

    multi = [item for item in results if item["case_id"] in MULTI_CASE_IDS]
    single = [item for item in results if item["case_id"] in SINGLE_CASE_IDS]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "configuration": {
            "flat_collection": FLAT_COLLECTION,
            "parent_child_collection": PARENT_CHILD_COLLECTION,
            "retrieval_top_k": flat_service.retrieval_top_k_default,
            "answer_top_k": 5,
            "parent_chunk_size": parent_settings.parent_chunk_size,
            "child_chunk_size": parent_settings.child_chunk_size,
            "child_chunk_overlap": parent_settings.child_chunk_overlap,
            "embedding_score_weight": flat_settings.embedding_score_weight,
            "bm25_score_weight": flat_settings.bm25_score_weight,
            "query_rewrite_disabled_for_control": True,
        },
        "parent_child_ingestion": ingestion,
        "multi_evidence_metrics": {
            "flat": aggregate(multi, "flat"),
            "parent_child": aggregate(multi, "parent_child"),
        },
        "single_evidence_metrics": {
            "flat": aggregate(single, "flat"),
            "parent_child": aggregate(single, "parent_child"),
        },
        "per_case_results": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Results written to: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
