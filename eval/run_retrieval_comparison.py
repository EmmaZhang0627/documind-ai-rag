from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_eval import (
    EVAL_DIR,
    discover_fixture_pdf,
    find_evidence_rank,
    is_grounded_case,
    load_cases,
    load_rag_service,
    match_keywords,
    prepare_eval_index,
    sanitize_error_message,
)


RESULTS_PATH = EVAL_DIR / "retrieval_comparison_latest.json"
STRATEGY_NAMES = ("vector_only", "hybrid", "hybrid_rerank")


def candidate_summary(
    candidate: dict[str, Any] | None,
    rank: int | None = None,
    expected_keywords: list[str] | None = None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None

    metadata = candidate.get("metadata") or {}
    keyword_result = match_keywords(
        candidate.get("document", ""),
        expected_keywords or [],
    )
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
        "matched_evidence_keywords": keyword_result["matched"],
        "contains_all_expected_evidence": keyword_result["passed"],
    }


def evaluate_strategy(
    ranked_candidates: list[dict[str, Any]],
    expected_keywords: list[str],
) -> dict[str, Any]:
    evidence_rank = find_evidence_rank(ranked_candidates, expected_keywords)
    reciprocal_rank = 1.0 / evidence_rank if evidence_rank is not None else 0.0

    return {
        "expected_evidence_keywords": expected_keywords,
        "evidence_rank": evidence_rank,
        "top_candidate": candidate_summary(
            ranked_candidates[0] if ranked_candidates else None,
            rank=1 if ranked_candidates else None,
            expected_keywords=expected_keywords,
        ),
        "evidence_in_top_1": evidence_rank == 1,
        "evidence_in_top_2": (
            evidence_rank is not None and evidence_rank <= 2
        ),
        "evidence_in_top_3": (
            evidence_rank is not None and evidence_rank <= 3
        ),
        "reciprocal_rank": reciprocal_rank,
        "evidence_not_found": evidence_rank is None,
        "ranked_candidates": [
            candidate_summary(
                candidate,
                rank=rank,
                expected_keywords=expected_keywords,
            )
            for rank, candidate in enumerate(ranked_candidates, start=1)
        ],
    }


def aggregate_strategy(
    per_case_results: list[dict[str, Any]],
    strategy_name: str,
) -> dict[str, Any]:
    strategy_results = [
        case_result["strategies"][strategy_name]
        for case_result in per_case_results
    ]
    case_count = len(strategy_results)
    top_1_count = sum(
        result["evidence_in_top_1"] for result in strategy_results
    )
    hit_2_count = sum(
        result["evidence_in_top_2"] for result in strategy_results
    )
    hit_3_count = sum(
        result["evidence_in_top_3"] for result in strategy_results
    )
    found_ranks = [
        result["evidence_rank"]
        for result in strategy_results
        if result["evidence_rank"] is not None
    ]
    reciprocal_rank_sum = sum(
        result["reciprocal_rank"] for result in strategy_results
    )

    return {
        "grounded_case_count": case_count,
        "top_1_evidence_hit_count": top_1_count,
        "top_1_evidence_hit_rate": (
            top_1_count / case_count if case_count else 0.0
        ),
        "hit_at_2_count": hit_2_count,
        "hit_at_2_rate": hit_2_count / case_count if case_count else 0.0,
        "hit_at_3_count": hit_3_count,
        "hit_at_3_rate": hit_3_count / case_count if case_count else 0.0,
        "mrr": reciprocal_rank_sum / case_count if case_count else 0.0,
        "average_evidence_rank_when_found": (
            sum(found_ranks) / len(found_ranks) if found_ranks else None
        ),
        "evidence_not_found_count": case_count - len(found_ranks),
    }


def classify_rerank_effect(
    hybrid_rank: int | None,
    reranked_rank: int | None,
) -> str:
    if hybrid_rank is None or reranked_rank is None:
        return "evidence_not_found"
    if reranked_rank < hybrid_rank:
        return "improved"
    if reranked_rank > hybrid_rank:
        return "worsened"
    return "unchanged"


def build_rerank_effect_analysis(
    per_case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    counts = {
        "improved": 0,
        "unchanged": 0,
        "worsened": 0,
        "evidence_not_found": 0,
    }

    for case_result in per_case_results:
        hybrid_rank = case_result["strategies"]["hybrid"]["evidence_rank"]
        reranked_rank = case_result["strategies"]["hybrid_rerank"][
            "evidence_rank"
        ]
        effect = classify_rerank_effect(hybrid_rank, reranked_rank)
        counts[effect] += 1
        cases.append({
            "case_id": case_result["case_id"],
            "category": case_result["category"],
            "hybrid_evidence_rank": hybrid_rank,
            "reranked_evidence_rank": reranked_rank,
            "effect": effect,
        })

    return {
        "counts": counts,
        "cases": cases,
    }


def print_summary(
    aggregate_metrics: dict[str, dict[str, Any]],
    rerank_effect_analysis: dict[str, Any],
    results_path: Path,
) -> None:
    print("Retrieval comparison complete")
    for strategy_name in STRATEGY_NAMES:
        metrics = aggregate_metrics[strategy_name]
        print(
            f"{strategy_name}: "
            f"Top-1={metrics['top_1_evidence_hit_count']}/"
            f"{metrics['grounded_case_count']} "
            f"Hit@2={metrics['hit_at_2_count']}/"
            f"{metrics['grounded_case_count']} "
            f"Hit@3={metrics['hit_at_3_count']}/"
            f"{metrics['grounded_case_count']} "
            f"MRR={metrics['mrr']:.4f} "
            f"AvgRank={metrics['average_evidence_rank_when_found']}"
        )
    counts = rerank_effect_analysis["counts"]
    print(
        "Rerank effects: "
        f"improved={counts['improved']} "
        f"unchanged={counts['unchanged']} "
        f"worsened={counts['worsened']} "
        f"not_found={counts['evidence_not_found']}"
    )
    print(f"Results written to: {results_path}")


def run_retrieval_comparison() -> int:
    cases = [case for case in load_cases() if is_grounded_case(case)]
    fixture_pdf = discover_fixture_pdf()
    rag_service = load_rag_service()
    fixture = prepare_eval_index(rag_service, fixture_pdf)

    from app.services import vector_db

    retrieval_top_k = rag_service.retrieval_top_k_default
    per_case_results: list[dict[str, Any]] = []

    for case in cases:
        query = case["question"]
        expected_keywords = case.get("expected_evidence_keywords") or []
        query_embedding = rag_service.embedder.embed(query)
        all_candidates = vector_db.retrieve_candidates(
            query_embedding,
            query,
        )

        vector_candidates = sorted(
            deepcopy(all_candidates),
            key=lambda candidate: candidate["embedding_score"],
            reverse=True,
        )[:retrieval_top_k]
        hybrid_candidates = sorted(
            deepcopy(all_candidates),
            key=lambda candidate: candidate["retrieval_score"],
            reverse=True,
        )[:retrieval_top_k]
        reranked_candidates = rag_service.reranker.rerank(
            query,
            deepcopy(hybrid_candidates),
        )

        per_case_results.append({
            "case_id": case.get("id"),
            "category": case.get("category"),
            "question": query,
            "expected_evidence_keywords": expected_keywords,
            "strategies": {
                "vector_only": evaluate_strategy(
                    vector_candidates,
                    expected_keywords,
                ),
                "hybrid": evaluate_strategy(
                    hybrid_candidates,
                    expected_keywords,
                ),
                "hybrid_rerank": evaluate_strategy(
                    reranked_candidates,
                    expected_keywords,
                ),
            },
        })

    aggregate_metrics = {
        strategy_name: aggregate_strategy(per_case_results, strategy_name)
        for strategy_name in STRATEGY_NAMES
    }
    rerank_effect_analysis = build_rerank_effect_analysis(per_case_results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
        "configuration": {
            "grounded_case_count": len(cases),
            "retrieval_top_k": retrieval_top_k,
            "embedding_score_weight": vector_db.embedding_score_weight,
            "bm25_score_weight": vector_db.bm25_score_weight,
            "reranker_enabled": vector_db.reranker_enabled,
            "reranker_model_name": vector_db.reranker_model_name,
            "strategy_definitions": {
                "vector_only": "Candidates ordered by embedding_score only.",
                "hybrid": (
                    "Candidates ordered by the configured weighted "
                    "embedding_score and normalized bm25_score."
                ),
                "hybrid_rerank": (
                    "Hybrid candidate pool reordered by the configured "
                    "CrossEncoder, with production fallback preserved."
                ),
            },
        },
        "aggregate_metrics": aggregate_metrics,
        "per_case_results": per_case_results,
        "rerank_effect_analysis": rerank_effect_analysis,
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")

    print_summary(aggregate_metrics, rerank_effect_analysis, RESULTS_PATH)
    return 0


def main() -> int:
    try:
        return run_retrieval_comparison()
    except Exception as error:
        print(
            "Retrieval comparison failed: "
            f"{error.__class__.__name__}: "
            f"{sanitize_error_message(str(error))[:300]}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
