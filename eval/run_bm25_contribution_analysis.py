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


RESULTS_PATH = EVAL_DIR / "bm25_contribution_latest.json"
EFFECT_NAMES = ("HELPFUL", "HARMFUL", "NEUTRAL", "EVIDENCE_NOT_FOUND")


def candidate_identity(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    return ":".join(
        str(value)
        for value in (
            metadata.get("document_id", "unknown"),
            metadata.get("version", "1"),
            metadata.get("chunk_index", "unknown"),
        )
    )


def summarize_candidate(
    candidate: dict[str, Any] | None,
    expected_keywords: list[str],
    rank: int | None = None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None

    metadata = candidate.get("metadata") or {}
    evidence_match = match_keywords(
        candidate.get("document", ""),
        expected_keywords,
    )
    return {
        "rank": rank,
        "candidate_identity": candidate_identity(candidate),
        "document_id": metadata.get("document_id"),
        "version": metadata.get("version"),
        "chunk_index": metadata.get("chunk_index"),
        "page_number": metadata.get("page_number"),
        "embedding_score": candidate.get("embedding_score"),
        "normalized_bm25_score": candidate.get("bm25_score"),
        "retrieval_score": candidate.get("retrieval_score"),
        "matched_evidence_keywords": evidence_match["matched"],
        "contains_all_expected_evidence": evidence_match["passed"],
    }


def evidence_rank(
    candidates: list[dict[str, Any]],
    expected_keywords: list[str],
    match_mode: str,
) -> int | None:
    return find_evidence_rank(
        candidates,
        expected_keywords,
        match_mode=match_mode,
    )


def classify_effect(
    vector_rank: int | None,
    hybrid_rank: int | None,
) -> str:
    if vector_rank is None and hybrid_rank is None:
        return "EVIDENCE_NOT_FOUND"
    if vector_rank is None:
        return "HELPFUL"
    if hybrid_rank is None:
        return "HARMFUL"
    if hybrid_rank < vector_rank:
        return "HELPFUL"
    if hybrid_rank > vector_rank:
        return "HARMFUL"
    return "NEUTRAL"


def score_margin(candidates: list[dict[str, Any]]) -> dict[str, float | None]:
    top_1_score = candidates[0]["retrieval_score"] if candidates else None
    top_2_score = candidates[1]["retrieval_score"] if len(candidates) > 1 else None
    margin = (
        top_1_score - top_2_score
        if top_1_score is not None and top_2_score is not None
        else None
    )
    return {
        "top1_score": top_1_score,
        "top2_score": top_2_score,
        "top1_minus_top2_margin": margin,
    }


def completion_candidate(
    candidates: list[dict[str, Any]],
    rank: int | None,
    expected_keywords: list[str],
) -> dict[str, Any] | None:
    if rank is None:
        return None
    return summarize_candidate(candidates[rank - 1], expected_keywords, rank)


def contributing_candidates(
    candidates: list[dict[str, Any]],
    expected_keywords: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        summary = summarize_candidate(candidate, expected_keywords, rank)
        if summary and summary["matched_evidence_keywords"]:
            summaries.append(summary)
    return summaries


def average_rank(case_results: list[dict[str, Any]], field: str) -> float | None:
    ranks = [result[field] for result in case_results if result[field] is not None]
    return sum(ranks) / len(ranks) if ranks else None


def rank_metrics(case_results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    ranks = [result[field] for result in case_results]
    case_count = len(ranks)
    found_ranks = [rank for rank in ranks if rank is not None]
    return {
        "case_count": case_count,
        "hit_at_1_count": sum(rank == 1 for rank in ranks),
        "hit_at_1_rate": (
            sum(rank == 1 for rank in ranks) / case_count if case_count else 0.0
        ),
        "hit_at_3_count": sum(
            rank is not None and rank <= 3 for rank in ranks
        ),
        "hit_at_3_rate": (
            sum(rank is not None and rank <= 3 for rank in ranks) / case_count
            if case_count
            else 0.0
        ),
        "mrr": (
            sum(1.0 / rank if rank is not None else 0.0 for rank in ranks)
            / case_count
            if case_count
            else 0.0
        ),
        "average_evidence_rank_when_found": (
            sum(found_ranks) / len(found_ranks) if found_ranks else None
        ),
        "evidence_not_found_count": case_count - len(found_ranks),
    }


def aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(case_results)
    counts = {
        effect: sum(result["bm25_effect"] == effect for result in case_results)
        for effect in EFFECT_NAMES
    }
    return {
        "case_count": case_count,
        "effect_counts": counts,
        "effect_percentages": {
            effect: counts[effect] / case_count if case_count else 0.0
            for effect in EFFECT_NAMES
        },
        "strategy_metrics": {
            "vector_only": rank_metrics(case_results, "vector_evidence_rank"),
            "bm25_only": rank_metrics(case_results, "bm25_evidence_rank"),
            "hybrid": rank_metrics(case_results, "hybrid_evidence_rank"),
        },
        "average_vector_evidence_rank_when_found": average_rank(
            case_results,
            "vector_evidence_rank",
        ),
        "average_bm25_evidence_rank_when_found": average_rank(
            case_results,
            "bm25_evidence_rank",
        ),
        "average_hybrid_evidence_rank_when_found": average_rank(
            case_results,
            "hybrid_evidence_rank",
        ),
        "vector_evidence_not_found_count": sum(
            result["vector_evidence_rank"] is None for result in case_results
        ),
        "bm25_evidence_not_found_count": sum(
            result["bm25_evidence_rank"] is None for result in case_results
        ),
        "hybrid_evidence_not_found_count": sum(
            result["hybrid_evidence_rank"] is None for result in case_results
        ),
    }


def build_category_analysis(
    case_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    categories = sorted({result["category"] for result in case_results})
    return {
        category: aggregate(
            [result for result in case_results if result["category"] == category]
        )
        for category in categories
    }


def build_focused_diagnostics(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {result["case_id"]: result for result in case_results}
    exact_term_rescues = [
        result
        for result in case_results
        if result["category"] == "keyword_exact_term"
        and result["bm25_effect"] == "HELPFUL"
    ]
    focus_ids = (
        "case_013_duration_assumptions",
        "case_017_capstone_length_paraphrase",
    )
    return {
        "hard_negative_cases": {
            case_id: by_id[case_id]
            for case_id in focus_ids
            if case_id in by_id
        },
        "exact_term_cases_rescued_by_bm25": exact_term_rescues,
    }


def print_summary(
    overall: dict[str, Any],
    category_analysis: dict[str, dict[str, Any]],
    results_path: Path,
) -> None:
    counts = overall["effect_counts"]
    percentages = overall["effect_percentages"]
    print("BM25 contribution analysis complete")
    for effect in EFFECT_NAMES:
        print(
            f"{effect}: {counts[effect]}/{overall['case_count']} "
            f"({percentages[effect]:.2%})"
        )
    print("By category")
    for category, metrics in category_analysis.items():
        category_counts = metrics["effect_counts"]
        print(
            f"  {category}: helpful={category_counts['HELPFUL']} "
            f"harmful={category_counts['HARMFUL']} "
            f"neutral={category_counts['NEUTRAL']} "
            f"avg_vector={metrics['average_vector_evidence_rank_when_found']} "
            f"avg_bm25={metrics['average_bm25_evidence_rank_when_found']} "
            f"avg_hybrid={metrics['average_hybrid_evidence_rank_when_found']}"
        )
    print(f"Results written to: {results_path}")


def run_bm25_contribution_analysis() -> int:
    cases = [case for case in load_cases() if is_grounded_case(case)]
    fixture_pdf = discover_fixture_pdf()
    rag_service = load_rag_service()
    fixture = prepare_eval_index(rag_service, fixture_pdf)

    from app.services import vector_db

    retrieval_top_k = rag_service.retrieval_top_k_default
    case_results: list[dict[str, Any]] = []

    for case in cases:
        query = case["question"]
        expected_keywords = case.get("expected_evidence_keywords") or []
        match_mode = case.get("evidence_match_mode", "single_chunk")
        query_embedding = rag_service.embedder.embed(query)
        all_candidates = rag_service.retriever.retrieve(
            query_embedding,
            query,
            top_k=rag_service.retriever.count(),
        )

        vector_candidates = sorted(
            deepcopy(all_candidates),
            key=lambda candidate: candidate["embedding_score"],
            reverse=True,
        )[:retrieval_top_k]
        bm25_candidates = sorted(
            deepcopy(all_candidates),
            key=lambda candidate: candidate["bm25_score"],
            reverse=True,
        )[:retrieval_top_k]
        hybrid_candidates = sorted(
            deepcopy(all_candidates),
            key=lambda candidate: candidate["retrieval_score"],
            reverse=True,
        )[:retrieval_top_k]

        vector_rank = evidence_rank(
            vector_candidates,
            expected_keywords,
            match_mode,
        )
        bm25_rank = evidence_rank(
            bm25_candidates,
            expected_keywords,
            match_mode,
        )
        hybrid_rank = evidence_rank(
            hybrid_candidates,
            expected_keywords,
            match_mode,
        )
        effect = classify_effect(vector_rank, hybrid_rank)

        vector_top_1 = vector_candidates[0] if vector_candidates else None
        bm25_top_1 = bm25_candidates[0] if bm25_candidates else None
        hybrid_top_1 = hybrid_candidates[0] if hybrid_candidates else None
        top_1_changed = (
            candidate_identity(vector_top_1) != candidate_identity(hybrid_top_1)
            if vector_top_1 is not None and hybrid_top_1 is not None
            else vector_top_1 is not hybrid_top_1
        )

        case_results.append({
            "case_id": case.get("id"),
            "category": case.get("category"),
            "question": query,
            "expected_evidence_keywords": expected_keywords,
            "evidence_match_mode": match_mode,
            "vector_evidence_rank": vector_rank,
            "bm25_evidence_rank": bm25_rank,
            "hybrid_evidence_rank": hybrid_rank,
            "bm25_effect": effect,
            "bm25_promoted_correct_candidate": effect == "HELPFUL",
            "bm25_promoted_incorrect_candidate": effect == "HARMFUL",
            "bm25_did_not_materially_affect_ranking": effect == "NEUTRAL",
            "vector_to_hybrid_top1_changed": top_1_changed,
            "candidate_diagnostics": {
                "vector_top_1": summarize_candidate(
                    vector_top_1,
                    expected_keywords,
                    1 if vector_top_1 is not None else None,
                ),
                "bm25_top_1": summarize_candidate(
                    bm25_top_1,
                    expected_keywords,
                    1 if bm25_top_1 is not None else None,
                ),
                "hybrid_top_1": summarize_candidate(
                    hybrid_top_1,
                    expected_keywords,
                    1 if hybrid_top_1 is not None else None,
                ),
                "vector_evidence_completion_candidate": completion_candidate(
                    vector_candidates,
                    vector_rank,
                    expected_keywords,
                ),
                "bm25_evidence_completion_candidate": completion_candidate(
                    bm25_candidates,
                    bm25_rank,
                    expected_keywords,
                ),
                "hybrid_evidence_completion_candidate": completion_candidate(
                    hybrid_candidates,
                    hybrid_rank,
                    expected_keywords,
                ),
                "hybrid_evidence_contributing_candidates": contributing_candidates(
                    hybrid_candidates,
                    expected_keywords,
                ),
            },
            "hybrid_score_margin": score_margin(hybrid_candidates),
        })

    overall = aggregate(case_results)
    category_analysis = build_category_analysis(case_results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
        "configuration": {
            "grounded_case_count": len(cases),
            "retrieval_top_k": retrieval_top_k,
            "embedding_score_weight": vector_db.embedding_score_weight,
            "bm25_score_weight": vector_db.bm25_score_weight,
            "bm25_normalization": "query-local max normalization",
            "effect_definition": {
                "HELPFUL": "Hybrid evidence rank is better than Vector-only.",
                "HARMFUL": "Hybrid evidence rank is worse than Vector-only.",
                "NEUTRAL": "Hybrid evidence rank is unchanged from Vector-only.",
                "EVIDENCE_NOT_FOUND": (
                    "Neither Vector-only nor Hybrid finds complete evidence."
                ),
            },
        },
        "aggregate_analysis": overall,
        "category_analysis": category_analysis,
        "per_case_analysis": case_results,
        "focused_diagnostics": build_focused_diagnostics(case_results),
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")

    print_summary(overall, category_analysis, RESULTS_PATH)
    return 0


def main() -> int:
    try:
        return run_bm25_contribution_analysis()
    except Exception as error:
        print(
            "BM25 contribution analysis failed: "
            f"{error.__class__.__name__}: "
            f"{sanitize_error_message(str(error))[:300]}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
