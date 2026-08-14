from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_corpus_v2_retrieval_comparison import (
    CATALOG_PATH,
    CASES_PATH,
    aggregate,
    build_service,
    evaluate_ranked,
    identity,
    load_json,
    rank_effect,
    stop_service,
)


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "corpus_v2_reranker_depth_comparison_latest.json"
DEPTHS = (10, 20)
STRATEGIES = ("hybrid_top10_rerank", "hybrid_top20_rerank")


def availability(
    candidates: list[dict[str, Any]], items: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluated = evaluate_ranked(candidates, items)
    return {
        "all_components_available": not evaluated["evidence_not_found"],
        "cumulative_evidence_rank": evaluated["evidence_rank"],
        "components": evaluated["components"],
    }


def strongest_competitor(
    ranked: list[dict[str, Any]],
    evaluated: dict[str, Any],
    expected_documents: set[str],
) -> dict[str, Any] | None:
    evidence_rank = evaluated["evidence_rank"]
    if evidence_rank in (None, 1):
        return None
    evidence_identities = {
        (acceptable["document_id"], acceptable["chunk_index"])
        for component in evaluated["components"]
        for acceptable in component["acceptable_identities"]
    }
    for rank, candidate in enumerate(ranked, start=1):
        if rank >= evidence_rank:
            break
        if identity(candidate) in evidence_identities:
            continue
        metadata = candidate.get("metadata") or {}
        document_id = str(metadata.get("document_id"))
        return {
            "rank": rank,
            "document_id": document_id,
            "chunk_index": metadata.get("chunk_index"),
            "page_number": metadata.get("page_number"),
            "competition_type": (
                "same_document_competition"
                if document_id in expected_documents
                else "cross_document_competition"
            ),
            "rerank_score": candidate.get("rerank_score"),
            "embedding_score": candidate.get("embedding_score"),
            "bm25_score": candidate.get("bm25_score"),
            "hybrid_score": candidate.get("retrieval_score"),
        }
    return None


def near_miss_outcome(top10_rank: int | None, top20_rank: int | None) -> str:
    if top20_rank is None:
        return "still_failed"
    if top20_rank <= 5:
        return "rescued"
    if top10_rank is None:
        return "partially_improved"
    if top20_rank < top10_rank:
        return "partially_improved"
    return "unchanged"


def run() -> int:
    all_cases = load_json(CASES_PATH)
    cases = [case for case in all_cases if case["grounded"]]
    catalog = load_json(CATALOG_PATH)
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence_items"]}
    service, settings = build_service()
    results: list[dict[str, Any]] = []
    rerank_seconds = {depth: 0.0 for depth in DEPTHS}
    executed_counts = {depth: 0 for depth in DEPTHS}
    warmup_seconds = 0.0

    try:
        candidate_count = service.retriever.count()
        if candidate_count != 425:
            raise ValueError("Corpus V2 collection must contain exactly 425 ACTIVE chunks.")

        prepared: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
        for index, case in enumerate(cases, start=1):
            items = [evidence_by_id[evidence_id] for evidence_id in case["evidence_ids"]]
            query_embedding = service.embedder.embed(case["question"])
            candidates = service.retriever.retrieve(
                query_embedding, case["question"], top_k=candidate_count
            )
            if len(candidates) != candidate_count:
                raise ValueError(f"{case['case_id']}: expected {candidate_count} candidates")
            hybrid = sorted(
                deepcopy(candidates), key=lambda item: item["retrieval_score"], reverse=True
            )
            prepared.append((case, items, hybrid))
            print(f"[prepare {index}/{len(cases)}] {case['case_id']}", flush=True)

        warm_case, _, warm_hybrid = prepared[0]
        started = time.perf_counter()
        warm_result = service.reranker.rerank(
            warm_case["question"], deepcopy(warm_hybrid[:10])
        )
        warmup_seconds = time.perf_counter() - started
        if not warm_result or not all(item.get("rerank_enabled") for item in warm_result):
            raise RuntimeError("CrossEncoder did not execute during warm-up; fallback results are invalid here.")

        for index, (case, items, hybrid) in enumerate(prepared, start=1):
            reranked_by_depth: dict[int, list[dict[str, Any]]] = {}
            order = DEPTHS if index % 2 else tuple(reversed(DEPTHS))
            for depth in order:
                started = time.perf_counter()
                reranked = service.reranker.rerank(
                    case["question"], deepcopy(hybrid[:depth])
                )
                rerank_seconds[depth] += time.perf_counter() - started
                if not all(item.get("rerank_enabled") for item in reranked):
                    raise RuntimeError(f"{case['case_id']}: CrossEncoder fallback at Top{depth}")
                executed_counts[depth] += 1
                reranked_by_depth[depth] = reranked

            strategies = {
                f"hybrid_top{depth}_rerank": evaluate_ranked(
                    reranked_by_depth[depth], items
                )
                for depth in DEPTHS
            }
            availabilities = {
                f"hybrid_top{depth}": availability(hybrid[:depth], items)
                for depth in DEPTHS
            }
            expected_documents = set(case["expected_document_ids"])
            competitors = {
                f"hybrid_top{depth}_rerank": strongest_competitor(
                    reranked_by_depth[depth],
                    strategies[f"hybrid_top{depth}_rerank"],
                    expected_documents,
                )
                for depth in DEPTHS
            }
            results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "evidence_match_mode": case["evidence_match_mode"],
                "evidence_ids": case["evidence_ids"],
                "expected_document_ids": case["expected_document_ids"],
                "hybrid_full_evidence_rank": availability(hybrid, items)["cumulative_evidence_rank"],
                "candidate_availability": availabilities,
                "strategies": strategies,
                "rank_effect_top10_to_top20": rank_effect(
                    strategies["hybrid_top10_rerank"]["evidence_rank"],
                    strategies["hybrid_top20_rerank"]["evidence_rank"],
                ),
                "strongest_competitors": competitors,
            })
            print(f"[rerank {index}/{len(cases)}] {case['case_id']}", flush=True)
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
    near_misses = []
    for result in results:
        full_rank = result["hybrid_full_evidence_rank"]
        if full_rank is None or not 11 <= full_rank <= 20:
            continue
        top10_rank = result["strategies"]["hybrid_top10_rerank"]["evidence_rank"]
        top20_rank = result["strategies"]["hybrid_top20_rerank"]["evidence_rank"]
        near_misses.append({
            "case_id": result["case_id"],
            "category": result["category"],
            "original_hybrid_rank": full_rank,
            "top10_rerank_rank": top10_rank,
            "top20_rerank_rank": top20_rank,
            "outcome": near_miss_outcome(top10_rank, top20_rank),
            "top20_competitor": result["strongest_competitors"]["hybrid_top20_rerank"],
        })

    top10_success_regressions = []
    for result in results:
        top10 = result["strategies"]["hybrid_top10_rerank"]
        top20 = result["strategies"]["hybrid_top20_rerank"]
        if top10["hit_at_5"] and (
            top20["evidence_rank"] is None or top20["evidence_rank"] > top10["evidence_rank"]
        ):
            top10_success_regressions.append({
                "case_id": result["case_id"],
                "category": result["category"],
                "top10_rank": top10["evidence_rank"],
                "top20_rank": top20["evidence_rank"],
                "hit_at_5_lost": not top20["hit_at_5"],
                "top20_competitor": result["strongest_competitors"]["hybrid_top20_rerank"],
            })

    latency = {
        f"top{depth}": {
            "candidate_pairs": len(results) * depth,
            "total_reranker_seconds": rerank_seconds[depth],
            "average_seconds_per_query": rerank_seconds[depth] / len(results),
            "average_seconds_per_pair": rerank_seconds[depth] / (len(results) * depth),
            "executed_query_count": executed_counts[depth],
        }
        for depth in DEPTHS
    }
    latency["top20_vs_top10_latency_ratio"] = (
        rerank_seconds[20] / rerank_seconds[10] if rerank_seconds[10] else None
    )
    latency["warmup_seconds_excluded"] = warmup_seconds
    latency["measurement_note"] = (
        "Wall-clock measurements are local observations. Pair count is the deterministic cost proxy. "
        "Depth execution order alternates by case after an excluded warm-up."
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "configuration": {
            "grounded_case_count": len(results),
            "excluded_fallback_case_count": len(all_cases) - len(results),
            "stored_candidate_count": 425,
            "production_retrieval_pool_unchanged": service.retrieval_top_k_default,
            "experimental_depths": list(DEPTHS),
            "embedding_score_weight": settings.embedding_score_weight,
            "bm25_score_weight": settings.bm25_score_weight,
            "reranker_model_name": settings.reranker_model_name,
            "crossencoder_required": True,
        },
        "overall_metrics": overall,
        "category_metrics": categories,
        "near_miss_recovery": {
            "case_count": len(near_misses),
            "outcome_counts": dict(Counter(item["outcome"] for item in near_misses)),
            "cases": near_misses,
        },
        "rank_effect_analysis": {
            "counts": dict(Counter(result["rank_effect_top10_to_top20"] for result in results)),
            "top10_hit5_success_regressions": top10_success_regressions,
            "strong_regressions": [
                item for item in top10_success_regressions if item["hit_at_5_lost"]
            ],
        },
        "multi_evidence_analysis": [
            result for result in results if result["category"] == "multi_evidence"
        ],
        "hard_negative_analysis": [
            result
            for result in results
            if result["category"] == "cross_document_hard_negative"
            or result["rank_effect_top10_to_top20"] == "worsened"
        ],
        "latency_and_cost_proxy": latency,
        "per_case_results": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("Corpus V2 reranker depth comparison complete")
    for strategy in STRATEGIES:
        metrics = overall[strategy]
        print(
            f"{strategy}: Hit@1={metrics['hit_at_1_rate']:.4f} "
            f"Hit@3={metrics['hit_at_3_rate']:.4f} "
            f"Hit@5={metrics['hit_at_5_rate']:.4f} MRR={metrics['mrr']:.4f}"
        )
    print(f"Results written to: {RESULTS_PATH}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(
            "Corpus V2 reranker depth comparison incomplete: "
            f"{error.__class__.__name__}: {str(error)[:400]}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
