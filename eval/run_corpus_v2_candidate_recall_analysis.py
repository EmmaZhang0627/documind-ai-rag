from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
CASES_PATH = EVAL_DIR / "corpus_v2_eval_cases.json"
CATALOG_PATH = EVAL_DIR / "corpus_v2_evidence_catalog.json"
RESULTS_PATH = EVAL_DIR / "corpus_v2_candidate_recall_latest.json"
PERSIST_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_audit"
COLLECTION_NAME = "documind_corpus_v2_ingestion_audit"
DEPTHS = (10, 20, 50)
STRATEGIES = ("vector_only", "hybrid")
FOCUS_CASE_IDS = {
    "v2_case_003_study_break_effect",
    "v2_case_019_collusion_exact",
    "v2_case_022_gai_policy_vs_ethics",
    "v2_case_044_programme_complaint_contact",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_backend_path() -> None:
    backend = PROJECT_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


def build_service():
    ensure_backend_path()
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


def candidate_identity(candidate: dict[str, Any]) -> tuple[str, int]:
    metadata = candidate.get("metadata") or {}
    return str(metadata["document_id"]), int(metadata["chunk_index"])


def ref_identity(ref: dict[str, Any]) -> tuple[str, int]:
    return str(ref["document_id"]), int(ref["chunk_index"])


def evidence_components(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for item in items:
        identities = {(item["document_id"], int(item["chunk_index"]))}
        identities.update(ref_identity(ref) for ref in item["acceptable_chunk_refs"])
        components.append({
            "evidence_id": item["evidence_id"],
            "component": "canonical_or_acceptable",
            "acceptable_identities": identities,
        })
        for index, ref in enumerate(item["supporting_chunk_refs"], start=1):
            components.append({
                "evidence_id": item["evidence_id"],
                "component": f"supporting_{index}",
                "acceptable_identities": {ref_identity(ref)},
            })
    return components


def candidate_details(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    return {
        "rank": rank,
        "document_id": metadata.get("document_id"),
        "chunk_index": metadata.get("chunk_index"),
        "page_number": metadata.get("page_number"),
        "embedding_score": candidate.get("embedding_score"),
        "bm25_score": candidate.get("bm25_score"),
        "hybrid_score": candidate.get("retrieval_score"),
    }


def evaluate_ranking(
    ranked: list[dict[str, Any]],
    components: list[dict[str, Any]],
    expected_documents: set[str],
) -> dict[str, Any]:
    component_results: list[dict[str, Any]] = []
    component_ranks: list[int | None] = []
    for component in components:
        match = next(
            (
                (rank, candidate)
                for rank, candidate in enumerate(ranked, start=1)
                if candidate_identity(candidate) in component["acceptable_identities"]
            ),
            None,
        )
        rank = match[0] if match else None
        component_ranks.append(rank)
        component_results.append({
            "evidence_id": component["evidence_id"],
            "component": component["component"],
            "first_rank": rank,
            "matched_candidate": candidate_details(match[1], rank) if match else None,
        })

    first_evidence_rank = (
        max(rank for rank in component_ranks if rank is not None)
        if component_ranks and all(rank is not None for rank in component_ranks)
        else None
    )
    recalls = {
        f"recall_at_{depth}": (
            all(rank is not None and rank <= depth for rank in component_ranks)
        )
        for depth in DEPTHS
    }
    competitor = None
    if first_evidence_rank != 1 and ranked:
        first_wrong = next(
            (
                (rank, candidate)
                for rank, candidate in enumerate(ranked, start=1)
                if first_evidence_rank is None or rank < first_evidence_rank
                if not any(
                    candidate_identity(candidate) in component["acceptable_identities"]
                    for component in components
                )
            ),
            None,
        )
        if first_wrong:
            rank, candidate = first_wrong
            details = candidate_details(candidate, rank)
            details["competition_type"] = (
                "same_document_competition"
                if details["document_id"] in expected_documents
                else "cross_document_competition"
            )
            details["target_evidence_rank"] = first_evidence_rank
            competitor = details

    return {
        "first_evidence_rank": first_evidence_rank,
        "maximum_required_component_rank": first_evidence_rank,
        **recalls,
        "components": component_results,
        "strongest_competitor_above_evidence": competitor,
    }


def depth_bucket(rank: int | None) -> str | None:
    if rank is not None and rank <= 10:
        return None
    if rank is not None and rank <= 20:
        return "rank_11_20"
    if rank is not None and rank <= 50:
        return "rank_21_50"
    return "beyond_50_or_absent"


def aggregate(results: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    values = [result["strategies"][strategy] for result in results]
    ranks = [value["first_evidence_rank"] for value in values if value["first_evidence_rank"] is not None]
    count = len(values)
    return {
        "case_count": count,
        **{
            f"recall_at_{depth}_count": sum(value[f"recall_at_{depth}"] for value in values)
            for depth in DEPTHS
        },
        **{
            f"recall_at_{depth}_rate": sum(value[f"recall_at_{depth}"] for value in values) / count if count else 0.0
            for depth in DEPTHS
        },
        "average_first_evidence_rank_when_found": sum(ranks) / len(ranks) if ranks else None,
        "median_first_evidence_rank_when_found": statistics.median(ranks) if ranks else None,
        "evidence_absent_beyond_top50_count": sum(not value["recall_at_50"] for value in values),
    }


def run() -> int:
    all_cases = load_json(CASES_PATH)
    cases = [case for case in all_cases if case["grounded"]]
    catalog = load_json(CATALOG_PATH)
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence_items"]}
    service, settings = build_service()
    results: list[dict[str, Any]] = []
    try:
        if service.retriever.count() != 425:
            raise ValueError("Corpus V2 collection must contain exactly 425 chunks.")
        for index, case in enumerate(cases, start=1):
            items = [evidence_by_id[evidence_id] for evidence_id in case["evidence_ids"]]
            components = evidence_components(items)
            query_embedding = service.embedder.embed(case["question"])
            candidates = service.retriever.retrieve(
                query_embedding, case["question"], top_k=service.retriever.count()
            )
            if len(candidates) != 425:
                raise ValueError(f"{case['case_id']}: expected 425 ACTIVE candidates")
            rankings = {
                "vector_only": sorted(
                    deepcopy(candidates), key=lambda item: item["embedding_score"], reverse=True
                ),
                "hybrid": sorted(
                    deepcopy(candidates), key=lambda item: item["retrieval_score"], reverse=True
                ),
            }
            strategy_results = {
                strategy: evaluate_ranking(
                    rankings[strategy], components, set(case["expected_document_ids"])
                )
                for strategy in STRATEGIES
            }
            for strategy in STRATEGIES:
                strategy_results[strategy]["failure_depth_bucket"] = depth_bucket(
                    strategy_results[strategy]["first_evidence_rank"]
                )
            results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "evidence_match_mode": case["evidence_match_mode"],
                "evidence_ids": case["evidence_ids"],
                "expected_document_ids": case["expected_document_ids"],
                "strategies": strategy_results,
            })
            print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
    finally:
        stop_service(service)

    overall = {strategy: aggregate(results, strategy) for strategy in STRATEGIES}
    categories = {
        category: {
            strategy: aggregate(
                [result for result in results if result["category"] == category], strategy
            )
            for strategy in STRATEGIES
        }
        for category in sorted({result["category"] for result in results})
    }
    failure_buckets = {
        strategy: dict(Counter(
            result["strategies"][strategy]["failure_depth_bucket"]
            for result in results
            if result["strategies"][strategy]["failure_depth_bucket"] is not None
        ))
        for strategy in STRATEGIES
    }
    competition = {
        strategy: {
            "counts": dict(Counter(
                competitor["competition_type"]
                for result in results
                if (competitor := result["strategies"][strategy]["strongest_competitor_above_evidence"]) is not None
            )),
            "cases": [
                {
                    "case_id": result["case_id"],
                    **result["strategies"][strategy]["strongest_competitor_above_evidence"],
                }
                for result in results
                if result["strategies"][strategy]["strongest_competitor_above_evidence"] is not None
            ],
        }
        for strategy in STRATEGIES
    }
    focused = [
        result for result in results
        if result["case_id"] in FOCUS_CASE_IDS
        or "corpus-v2-uolo-programme-terms" in result["expected_document_ids"]
        or result["evidence_match_mode"] == "cumulative_chunks"
    ]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "configuration": {
            "grounded_case_count": len(results),
            "excluded_fallback_case_count": len(all_cases) - len(results),
            "stored_candidate_count": 425,
            "reported_depths": list(DEPTHS),
            "production_retrieval_pool_unchanged": service.retrieval_top_k_default,
            "embedding_score_weight": settings.embedding_score_weight,
            "bm25_score_weight": settings.bm25_score_weight,
            "collection_name": COLLECTION_NAME,
            "reranker_executed": False,
        },
        "overall_metrics": overall,
        "category_metrics": categories,
        "failure_depth_buckets": failure_buckets,
        "reranker_recall_ceiling": {
            "candidate_source": "hybrid_top10",
            "available_case_count": overall["hybrid"]["recall_at_10_count"],
            "grounded_case_count": len(results),
            "coverage_rate": overall["hybrid"]["recall_at_10_rate"],
            "interpretation": "Maximum evidence availability, not guaranteed Hit@1 performance.",
        },
        "competition_decomposition": competition,
        "multi_evidence_analysis": [
            result for result in results if result["evidence_match_mode"] == "cumulative_chunks"
        ],
        "focused_diagnostics": focused,
        "per_case_results": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("Corpus V2 candidate recall analysis complete")
    for strategy in STRATEGIES:
        metrics = overall[strategy]
        print(
            f"{strategy}: Recall@10={metrics['recall_at_10_rate']:.4f} "
            f"Recall@20={metrics['recall_at_20_rate']:.4f} "
            f"Recall@50={metrics['recall_at_50_rate']:.4f}"
        )
    print(f"Results written to: {RESULTS_PATH}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(
            "Corpus V2 candidate recall analysis incomplete: "
            f"{error.__class__.__name__}: {str(error)[:300]}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
