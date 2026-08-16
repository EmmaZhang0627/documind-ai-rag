from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_corpus_v2_retrieval_comparison import (
    CATALOG_PATH,
    CASES_PATH,
    build_service,
    evaluate_ranked,
    load_json,
    stop_service,
)


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "corpus_v2_confidence_calibration_latest.json"
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
EXPECTED_CHUNK_COUNT = 425


def score_summary(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0] if ordered else None,
        "median": statistics.median(ordered) if ordered else None,
        "maximum": ordered[-1] if ordered else None,
    }


def candidate_identity(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    metadata = candidate.get("metadata") or {}
    return {
        "document_id": metadata.get("document_id"),
        "source_file": metadata.get("source_file"),
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index"),
    }


def threshold_metrics(
    threshold: float,
    grounded: list[dict[str, Any]],
    unanswerable: list[dict[str, Any]],
    guardrail: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted_grounded = [case for case in grounded if case["confidence_score"] >= threshold]
    refused_grounded = [case for case in grounded if case["confidence_score"] < threshold]
    false_accepts = [case for case in unanswerable if case["confidence_score"] >= threshold]
    confidence_refusals = [case for case in unanswerable if case["confidence_score"] < threshold]
    evidence_backed_accepts = [
        case for case in accepted_grounded if case["complete_evidence_in_context"]
    ]
    accepted_without_evidence = [
        case for case in accepted_grounded if not case["complete_evidence_in_context"]
    ]
    evidence_backed_false_refusals = [
        case for case in refused_grounded if case["complete_evidence_in_context"]
    ]
    return {
        "threshold": threshold,
        "true_accept": len(accepted_grounded),
        "false_refusal": len(refused_grounded),
        "grounded_acceptance_rate": len(accepted_grounded) / len(grounded),
        "true_refusal": len(confidence_refusals) + len(guardrail),
        "false_accept": len(false_accepts),
        "confidence_true_refusal": len(confidence_refusals),
        "pre_retrieval_guardrail_true_refusal": len(guardrail),
        "evidence_backed_accept": len(evidence_backed_accepts),
        "accepted_without_complete_evidence": len(accepted_without_evidence),
        "evidence_backed_false_refusal": len(evidence_backed_false_refusals),
        "false_refusal_case_ids": [case["case_id"] for case in refused_grounded],
        "evidence_backed_false_refusal_case_ids": [
            case["case_id"] for case in evidence_backed_false_refusals
        ],
        "false_accept_case_ids": [case["case_id"] for case in false_accepts],
        "accepted_without_complete_evidence_case_ids": [
            case["case_id"] for case in accepted_without_evidence
        ],
    }


def run() -> int:
    cases = load_json(CASES_PATH)
    catalog = load_json(CATALOG_PATH)
    evidence_by_id = {
        item["evidence_id"]: item for item in catalog["evidence_items"]
    }
    grounded_cases = [case for case in cases if case["grounded"]]
    unanswerable_cases = [
        case for case in cases if case["category"] == "document_unanswerable"
    ]
    guarded_cases = [
        case for case in cases if case["category"] in {"out_of_scope", "sensitive"}
    ]

    service, settings = build_service()
    grounded_results: list[dict[str, Any]] = []
    unanswerable_results: list[dict[str, Any]] = []
    guardrail_results: list[dict[str, Any]] = []
    try:
        if service.retriever.count() != EXPECTED_CHUNK_COUNT:
            raise ValueError(
                f"Corpus V2 collection must contain exactly {EXPECTED_CHUNK_COUNT} chunks."
            )

        for case in grounded_cases:
            _, ranked, retrieval_query = service._retrieve_and_rank(
                case["question"], service.retrieval_top_k_default
            )
            top = ranked[0] if ranked else None
            answer_top_k = int(case.get("top_k") or service.answer_top_k_default)
            evidence = [evidence_by_id[value] for value in case["evidence_ids"]]
            full_evaluation = evaluate_ranked(ranked, evidence)
            context_evaluation = evaluate_ranked(ranked[:answer_top_k], evidence)
            grounded_results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "retrieval_query": retrieval_query,
                "confidence_score": top["retrieval_score"] if top else 0.0,
                "selected_rerank_score": top.get("rerank_score") if top else None,
                "selected_candidate": candidate_identity(top),
                "answer_top_k": answer_top_k,
                "evidence_rank_in_reranked_pool": full_evaluation["evidence_rank"],
                "complete_evidence_in_context": not context_evaluation["evidence_not_found"],
                "context_evidence_rank": context_evaluation["evidence_rank"],
            })

        for case in unanswerable_cases:
            _, ranked, retrieval_query = service._retrieve_and_rank(
                case["question"], service.retrieval_top_k_default
            )
            top = ranked[0] if ranked else None
            unanswerable_results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "retrieval_query": retrieval_query,
                "confidence_score": top["retrieval_score"] if top else 0.0,
                "selected_rerank_score": top.get("rerank_score") if top else None,
                "selected_candidate": candidate_identity(top),
            })

        for case in guarded_cases:
            sensitive = service._detect_sensitive_input(case["question"])
            out_of_scope = service._detect_out_of_scope_decision_request(case["question"])
            detected_reason = (
                "sensitive_input_detected" if sensitive
                else "out_of_scope_decision_request" if out_of_scope
                else None
            )
            if detected_reason != case["expected_fallback_reason"]:
                raise AssertionError(
                    f"{case['case_id']}: expected {case['expected_fallback_reason']}, "
                    f"detected {detected_reason}"
                )
            guardrail_results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "detected_reason": detected_reason,
                "rejected_before_retrieval": True,
                "confidence_score": None,
            })
    finally:
        stop_service(service)

    sweep = [
        threshold_metrics(value, grounded_results, unanswerable_results, guardrail_results)
        for value in THRESHOLDS
    ]
    grounded_scores = [case["confidence_score"] for case in grounded_results]
    evidence_backed_scores = [
        case["confidence_score"]
        for case in grounded_results
        if case["complete_evidence_in_context"]
    ]
    unanswerable_scores = [case["confidence_score"] for case in unanswerable_results]
    overlap = (
        max(min(grounded_scores), min(unanswerable_scores))
        <= min(max(grounded_scores), max(unanswerable_scores))
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Corpus V2 evaluation-only confidence threshold calibration",
        "production_modified": False,
        "confidence_formula": {
            "selected_score": "reranked_top1.retrieval_score",
            "retrieval_score": "0.7 * embedding_score + 0.3 * normalized_bm25_score",
            "decision": "confidence_score >= threshold",
            "crossencoder_role": "ordering only; rerank_score is not thresholded",
        },
        "configuration": {
            "configured_production_threshold": settings.confidence_threshold,
            "tested_thresholds": list(THRESHOLDS),
            "retrieval_top_k": service.retrieval_top_k_default,
            "parent_child_retrieval_enabled": service.parent_child_retrieval_enabled,
            "query_rewrite_enabled": service.query_rewriter is not None,
            "stored_chunk_count": EXPECTED_CHUNK_COUNT,
        },
        "case_counts": {
            "grounded": len(grounded_results),
            "confidence_evaluable_unanswerable": len(unanswerable_results),
            "pre_retrieval_guardrail": len(guardrail_results),
        },
        "threshold_sweep": sweep,
        "score_distributions": {
            "grounded_all": score_summary(grounded_scores),
            "grounded_with_complete_context_evidence": score_summary(evidence_backed_scores),
            "document_unanswerable": score_summary(unanswerable_scores),
            "grounded_unanswerable_ranges_overlap": overlap,
        },
        "grounded_cases": grounded_results,
        "unanswerable_cases": unanswerable_results,
        "pre_retrieval_guardrail_cases": guardrail_results,
        "interpretation_notes": [
            "Label-based true_accept means a grounded case passed the scalar gate; it does not prove the selected context contains all required evidence.",
            "Only document_unanswerable fallback cases calibrate the confidence gate. Sensitive and out-of-scope cases are rejected before retrieval.",
            "This development benchmark contains seven PDFs and four confidence-evaluable negative cases; it cannot establish a production-optimal threshold.",
        ],
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "results_path": str(RESULTS_PATH),
        "threshold_sweep": sweep,
        "score_distributions": output["score_distributions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
