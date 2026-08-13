from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
CASES_PATH = EVAL_DIR / "corpus_v2_eval_cases.json"
CATALOG_PATH = EVAL_DIR / "corpus_v2_evidence_catalog.json"
MANIFEST_PATH = EVAL_DIR / "corpus_v2_manifest.json"
HARD_NEGATIVE_PATH = EVAL_DIR / "corpus_v2_hard_negative_map.json"
REVIEW_PATH = EVAL_DIR / "corpus_v2_eval_case_review.json"
PERSIST_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_audit"
COLLECTION_NAME = "documind_corpus_v2_ingestion_audit"
GROUNDED_TARGETS = {
    "direct_factual": 10,
    "semantic_paraphrase": 10,
    "keyword_exact_term": 8,
    "contextual": 9,
    "cross_document_hard_negative": 11,
    "multi_evidence": 6,
}
FALLBACK_TARGETS = {
    "document_unanswerable": 4,
    "out_of_scope": 3,
    "sensitive": 3,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value: str) -> str:
    value = value.casefold().replace("\u00ad", "")
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def phrases_present(text: str, phrases: list[str]) -> bool:
    normalized = normalize_text(text)
    return all(normalize_text(phrase) in normalized for phrase in phrases)


def load_chunks() -> dict[tuple[str, int], dict[str, Any]]:
    backend = PROJECT_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.config.settings import AppSettings
    from app.services.chroma_vector_store import ChromaPersistentVectorStore
    from chromadb.api.shared_system_client import SharedSystemClient

    settings = AppSettings.from_env()
    store = ChromaPersistentVectorStore(
        str(PERSIST_DIRECTORY), COLLECTION_NAME, settings.embedding_model_name,
        settings.embedding_score_weight, settings.bm25_score_weight,
    )
    try:
        records = store.collection.get(include=["documents", "metadatas"])
        return {
            (metadata["document_id"], metadata["chunk_index"]): {
                "text": text,
                "metadata": metadata,
            }
            for text, metadata in zip(records["documents"], records["metadatas"])
        }
    finally:
        store.client._system.stop()
        SharedSystemClient.clear_system_cache()


def validate() -> dict[str, Any]:
    cases = load_json(CASES_PATH)
    catalog = load_json(CATALOG_PATH)
    manifest = load_json(MANIFEST_PATH)
    hard_map = load_json(HARD_NEGATIVE_PATH)
    chunks = load_chunks()
    errors: list[str] = []

    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence_items"]}
    manifest_by_id = {item["document_id"]: item for item in manifest["documents"]}
    overlap_by_id = {item["overlap_id"]: item for item in hard_map["overlaps"]}
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("Case IDs must be unique.")

    grounded_cases = [case for case in cases if case.get("grounded") is True]
    fallback_cases = [case for case in cases if case.get("grounded") is False]
    grounded_categories = Counter(case["category"] for case in grounded_cases)
    fallback_categories = Counter(case["category"] for case in fallback_cases)
    if dict(grounded_categories) != GROUNDED_TARGETS:
        errors.append(f"Grounded category distribution mismatch: {dict(grounded_categories)}")
    if dict(fallback_categories) != FALLBACK_TARGETS:
        errors.append(f"Fallback category distribution mismatch: {dict(fallback_categories)}")

    review_cases: list[dict[str, Any]] = []
    for case in cases:
        case_id = case.get("case_id", "<missing>")
        evidence_ids = case.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            errors.append(f"{case_id}: evidence_ids must be a list")
            continue
        if not case.get("question") or not isinstance(case["question"], str):
            errors.append(f"{case_id}: question is required")

        if case["grounded"] is False:
            if evidence_ids or case.get("evidence_match_mode") is not None:
                errors.append(f"{case_id}: fallback cases must not carry ground truth")
            review_cases.append({
                "case_id": case_id,
                "grounded": False,
                "category": case["category"],
                "question": case["question"],
                "expected_status": case.get("expected_status"),
                "expected_fallback_reason": case.get("expected_fallback_reason"),
            })
            continue

        if not evidence_ids:
            errors.append(f"{case_id}: grounded case must reference evidence")
            continue
        missing_ids = [value for value in evidence_ids if value not in evidence_by_id]
        if missing_ids:
            errors.append(f"{case_id}: unknown evidence IDs {missing_ids}")
            continue
        items = [evidence_by_id[value] for value in evidence_ids]
        expected_documents = sorted({item["document_id"] for item in items})
        expected_sources = sorted({item["file_name"] for item in items})
        expected_pages = {item["page_number"] for item in items}
        for item in items:
            expected_pages.update(ref["page_number"] for ref in item["supporting_chunk_refs"])
        if case.get("expected_document_ids") != expected_documents:
            errors.append(f"{case_id}: expected_document_ids do not resolve from evidence")
        if case.get("expected_source_files") != expected_sources:
            errors.append(f"{case_id}: expected_source_files do not resolve from evidence")
        if case.get("expected_page_numbers") != sorted(expected_pages):
            errors.append(f"{case_id}: expected_page_numbers do not resolve from evidence")

        for item in items:
            manifest_item = manifest_by_id.get(item["document_id"])
            if not manifest_item or not manifest_item["include_in_text_retrieval"]:
                errors.append(f"{case_id}: evidence points to an excluded document")
            canonical = chunks.get((item["document_id"], item["chunk_index"]))
            if canonical is None:
                errors.append(f"{case_id}: canonical stored chunk is missing")
                continue
            if canonical["metadata"].get("page_number") != item["page_number"]:
                errors.append(f"{case_id}: canonical page metadata mismatch")
            if not phrases_present(canonical["text"], item["evidence_keywords_or_phrases"]):
                errors.append(f"{case_id}: canonical identifying evidence is missing")

        mode = case.get("evidence_match_mode")
        if mode == "single_chunk":
            if any(item["evidence_type"] != "single_chunk" for item in items):
                errors.append(f"{case_id}: single_chunk case references multi evidence")
            for item in items:
                if not item["acceptable_chunk_refs"]:
                    errors.append(f"{case_id}: no independently acceptable chunk")
                for ref in item["acceptable_chunk_refs"]:
                    candidate = chunks.get((ref["document_id"], ref["chunk_index"]))
                    if candidate is None or not phrases_present(candidate["text"], item["evidence_keywords_or_phrases"]):
                        errors.append(f"{case_id}: invalid acceptable chunk ref")
        elif mode == "cumulative_chunks":
            if not all(item["evidence_type"] == "multi_chunk_candidate" for item in items):
                errors.append(f"{case_id}: cumulative case lacks multi-chunk evidence")
            for item in items:
                if not item["supporting_chunk_refs"]:
                    errors.append(f"{case_id}: cumulative evidence lacks supporting chunks")
                for ref in item["supporting_chunk_refs"]:
                    candidate = chunks.get((ref["document_id"], ref["chunk_index"]))
                    if candidate is None or not phrases_present(candidate["text"], ref["identifying_phrases"]):
                        errors.append(f"{case_id}: invalid supporting chunk ref")
        else:
            errors.append(f"{case_id}: unsupported evidence_match_mode {mode!r}")

        overlap_id = case.get("hard_negative_overlap_id")
        competitor = None
        if case["category"] == "cross_document_hard_negative":
            overlap = overlap_by_id.get(overlap_id)
            if overlap is None:
                errors.append(f"{case_id}: hard-negative overlap is missing")
            else:
                if not set(evidence_ids).intersection(overlap["evidence_ids"]):
                    errors.append(f"{case_id}: target evidence is not in overlap map")
                competitor = {
                    "overlap_id": overlap_id,
                    "topic": overlap["topic"],
                    "competing_document_ids": sorted(
                        set(overlap["relevant_documents"]) - set(expected_documents)
                    ),
                    "why_they_could_compete": overlap["why_they_could_compete"],
                }
        elif overlap_id is not None:
            errors.append(f"{case_id}: non-hard-negative case has an overlap ID")

        review_cases.append({
            "case_id": case_id,
            "grounded": True,
            "category": case["category"],
            "question": case["question"],
            "source_documents": expected_documents,
            "source_files": expected_sources,
            "pages": sorted(expected_pages),
            "evidence": [
                {
                    "evidence_id": item["evidence_id"],
                    "short_claim": item["concise_factual_claim"],
                    "canonical_chunk": {
                        "page_number": item["page_number"],
                        "chunk_index": item["chunk_index"],
                    },
                    "acceptable_chunk_count": len(item["acceptable_chunk_refs"]),
                    "supporting_chunks": item["supporting_chunk_refs"],
                    "ambiguity_warning": item["ambiguity"],
                }
                for item in items
            ],
            "evidence_match_mode": mode,
            "hard_negative_competitor": competitor,
        })

    document_distribution = Counter(
        case["expected_document_ids"][0] for case in grounded_cases
    )
    review = {
        "review_version": "1",
        "validation": {"passed": not errors, "errors": errors},
        "case_design_decisions": {
            "ambiguity_warnings_included": [
                case["case_id"]
                for case in review_cases
                if case.get("grounded")
                and any(
                    evidence["ambiguity_warning"]
                    for evidence in case.get("evidence", [])
                )
            ],
            "rejected_proposals": [
                {
                    "proposal": "Ask for a universally applicable legal effect of the right to be forgotten.",
                    "reason": "ev-ethics-005 is a jurisdiction-specific example and must not be generalized.",
                },
                {
                    "proposal": "Ask for facts visible only in Serverless figures.",
                    "reason": "The current text benchmark cannot treat diagram-only relationships as recoverable evidence.",
                },
                {
                    "proposal": "Treat overlapping duplicate chunks as multi-evidence.",
                    "reason": "Duplicate complete chunks use OR semantics and do not require cumulative coverage.",
                },
            ],
        },
        "summary": {
            "total_case_count": len(cases),
            "grounded_case_count": len(grounded_cases),
            "fallback_case_count": len(fallback_cases),
            "grounded_category_distribution": dict(grounded_categories),
            "fallback_category_distribution": dict(fallback_categories),
            "per_document_case_distribution": dict(sorted(document_distribution.items())),
            "single_chunk_case_count": sum(case.get("evidence_match_mode") == "single_chunk" for case in grounded_cases),
            "cumulative_chunks_case_count": sum(case.get("evidence_match_mode") == "cumulative_chunks" for case in grounded_cases),
            "hard_negative_case_count": grounded_categories["cross_document_hard_negative"],
        },
        "cases": review_cases,
    }
    return review


def main() -> int:
    review = validate()
    with REVIEW_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(review, file, ensure_ascii=False, indent=2)
        file.write("\n")
    summary = review["summary"]
    print("Corpus V2 case validation complete")
    print(f"Grounded cases: {summary['grounded_case_count']}")
    print(f"Fallback cases: {summary['fallback_case_count']}")
    print(f"Validation passed: {review['validation']['passed']}")
    print(f"Review written to: {REVIEW_PATH}")
    if not review["validation"]["passed"]:
        for error in review["validation"]["errors"]:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
