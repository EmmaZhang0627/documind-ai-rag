from __future__ import annotations

import json
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
    load_json,
    rank_effect,
    stop_service,
)


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "corpus_v2_query_rewrite_comparison_latest.json"
FOCUSED_CASE_IDS = (
    "v2_case_003_study_break_effect",
    "v2_case_005_elective_group_context",
    "v2_case_010_anonymous_complaint_paraphrase",
    "v2_case_012_judgement_scope",
    "v2_case_014_admissions_judgement_vs_integrity",
    "v2_case_018_commissioned_work_paraphrase",
    "v2_case_019_collusion_exact",
    "v2_case_022_gai_policy_vs_ethics",
    "v2_case_028_class_name_context",
    "v2_case_032_modern_ai_method",
    "v2_case_034_reontologizing_exact",
    "v2_case_037_ai_method_vs_student_use",
    "v2_case_044_programme_complaint_contact",
)
STRATEGIES = ("original_query", "rewritten_query")


def comparison_effect(before: int | None, after: int | None) -> str:
    effect = rank_effect(before, after)
    return "unchanged" if effect == "evidence_not_found" else effect


def retrieve_and_rerank(service: Any, query: str, top_k: int) -> tuple[list, list]:
    embedding = service.embedder.embed(query)
    hybrid = service.retriever.retrieve(embedding, query, top_k=top_k)
    reranked = service.reranker.rerank(query, deepcopy(hybrid))
    if reranked and not all(item.get("rerank_enabled") for item in reranked):
        raise RuntimeError("CrossEncoder fallback occurred; comparison would be invalid.")
    return hybrid, reranked


def run() -> int:
    all_cases = load_json(CASES_PATH)
    by_id = {case["case_id"]: case for case in all_cases if case["grounded"]}
    cases = [by_id[case_id] for case_id in FOCUSED_CASE_IDS]
    catalog = load_json(CATALOG_PATH)
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence_items"]}
    service, settings = build_service()
    results: list[dict[str, Any]] = []
    top_k = service.retrieval_top_k_default

    try:
        if service.retriever.count() != 425:
            raise ValueError("Corpus V2 collection must contain exactly 425 chunks.")
        if service.query_rewriter is None:
            raise RuntimeError("Query rewrite service is not configured.")

        for index, case in enumerate(cases, start=1):
            original_query = case["question"]
            retrieval_query = service.query_rewriter.rewrite(original_query)
            items = [evidence_by_id[value] for value in case["evidence_ids"]]

            original_hybrid, original_reranked = retrieve_and_rerank(
                service, original_query, top_k
            )
            rewritten_hybrid, rewritten_reranked = retrieve_and_rerank(
                service, retrieval_query, top_k
            )
            original_result = evaluate_ranked(original_reranked, items)
            rewritten_result = evaluate_ranked(rewritten_reranked, items)
            results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "original_query": original_query,
                "retrieval_query": retrieval_query,
                "rewrite_applied": retrieval_query != " ".join(original_query.split()),
                "evidence_match_mode": case["evidence_match_mode"],
                "evidence_ids": case["evidence_ids"],
                "strategies": {
                    "original_query": original_result,
                    "rewritten_query": rewritten_result,
                },
                "hybrid_evidence_ranks": {
                    "original_query": evaluate_ranked(original_hybrid, items)["evidence_rank"],
                    "rewritten_query": evaluate_ranked(rewritten_hybrid, items)["evidence_rank"],
                },
                "effect": comparison_effect(
                    original_result["evidence_rank"], rewritten_result["evidence_rank"]
                ),
            })
            print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
    finally:
        stop_service(service)

    overall = {strategy: aggregate(results, strategy) for strategy in STRATEGIES}
    categories = {
        category: {
            strategy: aggregate(
                [result for result in results if result["category"] == category],
                strategy,
            )
            for strategy in STRATEGIES
        }
        for category in sorted({result["category"] for result in results})
    }
    effects = Counter(result["effect"] for result in results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "configuration": {
            "focused_case_count": len(results),
            "stored_candidate_count": 425,
            "retrieval_top_k": top_k,
            "embedding_score_weight": settings.embedding_score_weight,
            "bm25_score_weight": settings.bm25_score_weight,
            "reranker_model_name": settings.reranker_model_name,
            "query_rewrite_policy": "vague-reference-or-question-with-at-least-9-words",
        },
        "overall_metrics": overall,
        "category_metrics": categories,
        "effect_counts": dict(effects),
        "rewrite_applied_count": sum(result["rewrite_applied"] for result in results),
        "improved_cases": [result for result in results if result["effect"] == "improved"],
        "worsened_cases": [result for result in results if result["effect"] == "worsened"],
        "unchanged_cases": [result for result in results if result["effect"] == "unchanged"],
        "per_case_results": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("Corpus V2 query rewrite comparison complete")
    for strategy in STRATEGIES:
        metrics = overall[strategy]
        print(
            f"{strategy}: Hit@1={metrics['hit_at_1_rate']:.4f} "
            f"Hit@3={metrics['hit_at_3_rate']:.4f} "
            f"Hit@5={metrics['hit_at_5_rate']:.4f} MRR={metrics['mrr']:.4f}"
        )
    print(f"Effects: {dict(effects)}")
    print(f"Results written to: {RESULTS_PATH}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(
            "Corpus V2 query rewrite comparison incomplete: "
            f"{error.__class__.__name__}: {str(error)[:400]}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
