from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
CASES_PATH = EVAL_DIR / "corpus_v2_eval_cases.json"
CATALOG_PATH = EVAL_DIR / "corpus_v2_evidence_catalog.json"
HARD_NEGATIVE_PATH = EVAL_DIR / "corpus_v2_hard_negative_map.json"
RESULTS_PATH = EVAL_DIR / "corpus_v2_retrieval_comparison_latest.json"
PERSIST_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_audit"
COLLECTION_NAME = "documind_corpus_v2_ingestion_audit"
STRATEGIES = ("vector_only", "hybrid", "hybrid_rerank")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_backend_path() -> None:
    backend = PROJECT_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


def build_service():
    ensure_backend_path()
    from dataclasses import replace
    from app.config.settings import AppSettings
    from app.dependencies.rag_dependencies import build_rag_service

    settings = replace(
        AppSettings.from_env(),
        vector_store_backend="chroma",
        chroma_persist_directory=str(PERSIST_DIRECTORY),
        chroma_collection_name=COLLECTION_NAME,
    )
    return build_rag_service(settings), settings


def stop_service(service: Any) -> None:
    from chromadb.api.shared_system_client import SharedSystemClient

    service.retriever.vector_store.client._system.stop()
    SharedSystemClient.clear_system_cache()


def identity(candidate: dict[str, Any]) -> tuple[str, int]:
    metadata = candidate.get("metadata") or {}
    return str(metadata.get("document_id")), int(metadata.get("chunk_index"))


def ref_identity(ref: dict[str, Any]) -> tuple[str, int]:
    return str(ref["document_id"]), int(ref["chunk_index"])


def evidence_components(item: dict[str, Any]) -> list[dict[str, Any]]:
    canonical = (item["document_id"], int(item["chunk_index"]))
    acceptable = {canonical}
    acceptable.update(ref_identity(ref) for ref in item["acceptable_chunk_refs"])
    components = [{
        "component": "canonical_or_acceptable",
        "acceptable_identities": acceptable,
    }]
    for index, ref in enumerate(item["supporting_chunk_refs"], start=1):
        components.append({
            "component": f"supporting_{index}",
            "acceptable_identities": {ref_identity(ref)},
        })
    return components


def evaluate_ranked(
    candidates: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    components = [
        component
        for item in items
        for component in evidence_components(item)
    ]
    first_ranks: list[int | None] = []
    matched_candidates: list[dict[str, Any] | None] = []
    for component in components:
        match = next(
            (
                (rank, candidate)
                for rank, candidate in enumerate(candidates, start=1)
                if identity(candidate) in component["acceptable_identities"]
            ),
            None,
        )
        first_ranks.append(match[0] if match else None)
        matched_candidates.append(match[1] if match else None)

    evidence_rank = (
        max(rank for rank in first_ranks if rank is not None)
        if first_ranks and all(rank is not None for rank in first_ranks)
        else None
    )
    first_evidence_candidate = next(
        (
            candidate
            for _, candidate in sorted(
                (
                    (rank, candidate)
                    for rank, candidate in zip(first_ranks, matched_candidates)
                    if rank is not None and candidate is not None
                ),
                key=lambda pair: pair[0],
            )
        ),
        None,
    )
    component_results = []
    for component, rank, candidate in zip(components, first_ranks, matched_candidates):
        component_results.append({
            "component": component["component"],
            "acceptable_identities": [
                {"document_id": document_id, "chunk_index": chunk_index}
                for document_id, chunk_index in sorted(component["acceptable_identities"])
            ],
            "first_rank": rank,
            "matched_identity": (
                {
                    "document_id": identity(candidate)[0],
                    "chunk_index": identity(candidate)[1],
                    "page_number": (candidate.get("metadata") or {}).get("page_number"),
                }
                if candidate else None
            ),
        })

    top = candidates[0] if candidates else None
    top_metadata = (top or {}).get("metadata") or {}
    return {
        "evidence_rank": evidence_rank,
        "hit_at_1": evidence_rank == 1,
        "hit_at_3": evidence_rank is not None and evidence_rank <= 3,
        "hit_at_5": evidence_rank is not None and evidence_rank <= 5,
        "reciprocal_rank": 1.0 / evidence_rank if evidence_rank else 0.0,
        "evidence_not_found": evidence_rank is None,
        "top1_source_document_id": top_metadata.get("document_id"),
        "top1_source_file": top_metadata.get("source_file"),
        "top1_identity": (
            {"document_id": identity(top)[0], "chunk_index": identity(top)[1]}
            if top else None
        ),
        "first_relevant_page": (
            (first_evidence_candidate.get("metadata") or {}).get("page_number")
            if first_evidence_candidate else None
        ),
        "components": component_results,
        "rerank_executed": any(candidate.get("rerank_enabled") for candidate in candidates),
        "top_candidates": [candidate_summary(candidate, rank) for rank, candidate in enumerate(candidates[:5], 1)],
    }


def candidate_summary(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    return {
        "rank": rank,
        "document_id": metadata.get("document_id"),
        "source_file": metadata.get("source_file"),
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index"),
        "embedding_score": candidate.get("embedding_score"),
        "bm25_score": candidate.get("bm25_score"),
        "retrieval_score": candidate.get("retrieval_score"),
        "rerank_score": candidate.get("rerank_score"),
        "rerank_enabled": candidate.get("rerank_enabled", False),
    }


def aggregate(results: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    values = [result["strategies"][strategy] for result in results]
    count = len(values)
    ranks = [value["evidence_rank"] for value in values if value["evidence_rank"] is not None]
    return {
        "case_count": count,
        "hit_at_1_count": sum(value["hit_at_1"] for value in values),
        "hit_at_1_rate": sum(value["hit_at_1"] for value in values) / count if count else 0.0,
        "hit_at_3_count": sum(value["hit_at_3"] for value in values),
        "hit_at_3_rate": sum(value["hit_at_3"] for value in values) / count if count else 0.0,
        "hit_at_5_count": sum(value["hit_at_5"] for value in values),
        "hit_at_5_rate": sum(value["hit_at_5"] for value in values) / count if count else 0.0,
        "mrr": sum(value["reciprocal_rank"] for value in values) / count if count else 0.0,
        "average_evidence_rank_when_found": sum(ranks) / len(ranks) if ranks else None,
        "evidence_not_found_count": count - len(ranks),
    }


def rank_effect(before: int | None, after: int | None) -> str:
    if before is None and after is None:
        return "evidence_not_found"
    if before is None:
        return "improved"
    if after is None:
        return "worsened"
    if after < before:
        return "improved"
    if after > before:
        return "worsened"
    return "unchanged"


def failure_labels(result: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    reranked = result["strategies"]["hybrid_rerank"]
    if not reranked["hit_at_5"]:
        labels.append("retrieval_failure")
    elif not reranked["hit_at_1"]:
        labels.append("ranking_failure")
    if result["category"] == "cross_document_hard_negative" and result["hard_negative"]["competitor_outranked_target"]["hybrid_rerank"]:
        labels.append("wrong_source_competition")
    if result["rerank_effect"] == "worsened":
        labels.append("reranker_regression")
    if result["evidence_match_mode"] == "cumulative_chunks":
        found_components = sum(component["first_rank"] is not None for component in reranked["components"])
        if found_components > 1:
            labels.append("multi_evidence_composition_limitation")
    return labels


def run() -> int:
    cases = [case for case in load_json(CASES_PATH) if case["grounded"]]
    fallback_count = len(load_json(CASES_PATH)) - len(cases)
    catalog = load_json(CATALOG_PATH)
    hard_map = load_json(HARD_NEGATIVE_PATH)
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence_items"]}
    overlap_by_id = {item["overlap_id"]: item for item in hard_map["overlaps"]}
    service, settings = build_service()
    retrieval_top_k = service.retrieval_top_k_default
    results: list[dict[str, Any]] = []
    try:
        if service.retriever.count() != 425:
            raise ValueError("Corpus V2 collection must contain exactly 425 chunks.")
        for index, case in enumerate(cases, start=1):
            query = case["question"]
            items = [evidence_by_id[value] for value in case["evidence_ids"]]
            query_embedding = service.embedder.embed(query)
            all_candidates = service.retriever.retrieve(
                query_embedding, query, top_k=service.retriever.count()
            )
            if len(all_candidates) != 425:
                raise ValueError(f"{case['case_id']}: candidate corpus is not 425 ACTIVE chunks")
            vector = sorted(deepcopy(all_candidates), key=lambda item: item["embedding_score"], reverse=True)[:retrieval_top_k]
            hybrid = sorted(deepcopy(all_candidates), key=lambda item: item["retrieval_score"], reverse=True)[:retrieval_top_k]
            reranked = service.reranker.rerank(query, deepcopy(hybrid))
            strategies = {
                "vector_only": evaluate_ranked(vector, items),
                "hybrid": evaluate_ranked(hybrid, items),
                "hybrid_rerank": evaluate_ranked(reranked, items),
            }
            expected_documents = set(case["expected_document_ids"])
            for value in strategies.values():
                value["top1_source_correct"] = value["top1_source_document_id"] in expected_documents
                allowed_pages = {
                    ref["page_number"]
                    for item in items
                    for ref in item["acceptable_chunk_refs"]
                } | {
                    ref["page_number"]
                    for item in items
                    for ref in item["supporting_chunk_refs"]
                } | {item["page_number"] for item in items}
                value["first_relevant_page_acceptable"] = (
                    value["first_relevant_page"] in allowed_pages
                    if value["first_relevant_page"] is not None else None
                )

            overlap = overlap_by_id.get(case.get("hard_negative_overlap_id"))
            competitors = sorted(set(overlap["relevant_documents"]) - expected_documents) if overlap else []
            hard_negative = {
                "overlap_id": case.get("hard_negative_overlap_id"),
                "competitor_document_ids": competitors,
                "competitor_outranked_target": {
                    strategy: strategies[strategy]["top1_source_document_id"] in competitors
                    and strategies[strategy]["evidence_rank"] != 1
                    for strategy in STRATEGIES
                },
            }
            effect = rank_effect(strategies["hybrid"]["evidence_rank"], strategies["hybrid_rerank"]["evidence_rank"])
            result = {
                "case_id": case["case_id"],
                "category": case["category"],
                "question": query,
                "evidence_ids": case["evidence_ids"],
                "evidence_match_mode": case["evidence_match_mode"],
                "expected_document_ids": case["expected_document_ids"],
                "expected_source_files": case["expected_source_files"],
                "expected_page_numbers": case["expected_page_numbers"],
                "strategies": strategies,
                "hard_negative": hard_negative,
                "rerank_effect": effect,
            }
            result["failure_attribution"] = failure_labels(result)
            if overlap:
                classifications: dict[str, str | None] = {}
                for strategy in STRATEGIES:
                    ranked = strategies[strategy]
                    if hard_negative["competitor_outranked_target"][strategy]:
                        classifications[strategy] = (
                            "lexical_competitor_promotion"
                            if strategy != "vector_only"
                            and not hard_negative["competitor_outranked_target"]["vector_only"]
                            else "semantic_competitor_promotion"
                        )
                    elif strategy == "hybrid_rerank" and effect == "worsened":
                        classifications[strategy] = "reranker_regression"
                    elif ranked["evidence_rank"] is None:
                        classifications[strategy] = "candidate_retrieval_weakness"
                    elif ranked["evidence_rank"] > 1:
                        classifications[strategy] = "other_or_uncertain"
                    else:
                        classifications[strategy] = None
                result["hard_negative"]["observed_failure_classification"] = classifications
            results.append(result)
            print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
    finally:
        stop_service(service)

    overall = {strategy: aggregate(results, strategy) for strategy in STRATEGIES}
    categories = {
        category: {strategy: aggregate([r for r in results if r["category"] == category], strategy) for strategy in STRATEGIES}
        for category in sorted({r["category"] for r in results})
    }
    documents = {
        document_id: {
            strategy: aggregate([r for r in results if document_id in r["expected_document_ids"]], strategy)
            for strategy in STRATEGIES
        }
        for document_id in sorted({d for r in results for d in r["expected_document_ids"]})
    }
    source_accuracy = {
        strategy: {
            "overall": sum(r["strategies"][strategy]["top1_source_correct"] for r in results) / len(results),
            "by_category": {
                category: sum(r["strategies"][strategy]["top1_source_correct"] for r in results if r["category"] == category) / sum(r["category"] == category for r in results)
                for category in categories
            },
            "hard_negative": sum(r["strategies"][strategy]["top1_source_correct"] for r in results if r["category"] == "cross_document_hard_negative") / 11,
        }
        for strategy in STRATEGIES
    }
    effects = Counter(result["rerank_effect"] for result in results)
    transitions = [{
        "case_id": result["case_id"],
        "hybrid_rank": result["strategies"]["hybrid"]["evidence_rank"],
        "reranked_rank": result["strategies"]["hybrid_rerank"]["evidence_rank"],
        "effect": result["rerank_effect"],
        "rerank_executed": result["strategies"]["hybrid_rerank"]["rerank_executed"],
    } for result in results]
    vector_hybrid = Counter(
        rank_effect(result["strategies"]["vector_only"]["evidence_rank"], result["strategies"]["hybrid"]["evidence_rank"])
        for result in results
    )
    vector_hybrid_by_category = {
        category: dict(Counter(
            rank_effect(result["strategies"]["vector_only"]["evidence_rank"], result["strategies"]["hybrid"]["evidence_rank"])
            for result in results if result["category"] == category
        ))
        for category in categories
    }
    page_analysis = {
        strategy: {
            "applicable_count": sum(result["strategies"][strategy]["first_relevant_page_acceptable"] is not None for result in results),
            "acceptable_page_count": sum(result["strategies"][strategy]["first_relevant_page_acceptable"] is True for result in results),
            "acceptable_page_rate": (
                sum(result["strategies"][strategy]["first_relevant_page_acceptable"] is True for result in results)
                / sum(result["strategies"][strategy]["first_relevant_page_acceptable"] is not None for result in results)
                if any(result["strategies"][strategy]["first_relevant_page_acceptable"] is not None for result in results)
                else 0.0
            ),
            "policy": "The page of the first identity-matched canonical, acceptable, or supporting chunk is accepted when catalogued for that evidence.",
        }
        for strategy in STRATEGIES
    }
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "configuration": {
            "grounded_case_count": len(results),
            "excluded_fallback_case_count": fallback_count,
            "stored_candidate_count": 425,
            "retrieval_top_k": retrieval_top_k,
            "embedding_score_weight": settings.embedding_score_weight,
            "bm25_score_weight": settings.bm25_score_weight,
            "reranker_enabled": settings.reranker_enabled,
            "reranker_model_name": settings.reranker_model_name,
            "collection_name": COLLECTION_NAME,
        },
        "overall_metrics": overall,
        "category_metrics": categories,
        "document_metrics": documents,
        "source_selection_analysis": source_accuracy,
        "page_analysis": page_analysis,
        "vector_to_hybrid_effect_counts": dict(vector_hybrid),
        "vector_to_hybrid_effect_by_category": vector_hybrid_by_category,
        "reranker_effect_analysis": {
            "counts": dict(effects),
            "crossencoder_executed_case_count": sum(t["rerank_executed"] for t in transitions),
            "transitions": transitions,
            "strongest_rescues": sorted([t for t in transitions if t["effect"] == "improved"], key=lambda t: ((t["hybrid_rank"] or 999) - (t["reranked_rank"] or 999)), reverse=True)[:5],
            "strongest_regressions": sorted([t for t in transitions if t["effect"] == "worsened"], key=lambda t: ((t["reranked_rank"] or 999) - (t["hybrid_rank"] or 999)), reverse=True)[:5],
        },
        "failure_attribution_counts": dict(Counter(label for result in results for label in result["failure_attribution"])),
        "hard_negative_analysis": [result for result in results if result["category"] == "cross_document_hard_negative"],
        "multi_evidence_analysis": [result for result in results if result["category"] == "multi_evidence"],
        "per_case_results": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("Corpus V2 retrieval comparison complete")
    for strategy in STRATEGIES:
        metrics = overall[strategy]
        print(f"{strategy}: Hit@1={metrics['hit_at_1_rate']:.4f} Hit@3={metrics['hit_at_3_rate']:.4f} Hit@5={metrics['hit_at_5_rate']:.4f} MRR={metrics['mrr']:.4f}")
    print(f"Results written to: {RESULTS_PATH}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(f"Corpus V2 retrieval comparison incomplete: {error.__class__.__name__}: {str(error)[:300]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
