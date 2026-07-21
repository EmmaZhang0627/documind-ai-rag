from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
CASES_PATH = EVAL_DIR / "documind_eval_cases.json"
RESULTS_PATH = EVAL_DIR / "eval_results_latest.json"
NOT_APPLICABLE = "not_applicable"
CITATION_CORRECTNESS_THRESHOLD = 1.0
SOURCE_EVIDENCE_TEXT_KEYS = (
    "source_snippet",
    "snippet",
    "chunk_text",
    "content",
    "text",
    "document",
)

ANSWERED_STATUSES = {"answered", "success"}
LOW_CONFIDENCE_STATUSES = {
    "low_confidence",
    "refused",
    "insufficient_evidence",
}
EXPECTED_BEHAVIOR_STATUSES = {
    "answer_with_sources": ANSWERED_STATUSES,
    "low_confidence_refusal": LOW_CONFIDENCE_STATUSES,
    "insufficient_evidence": {"insufficient_evidence"},
    "conflicting_sources": {"conflicting_sources"},
    "out_of_scope": {"out_of_scope", "human_review_required"},
    "human_review_required": {"human_review_required", "out_of_scope"},
    "sensitive_input_detected": {"sensitive_input_detected"},
}


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON list.")

    return cases


def load_rag_service():
    backend_path = PROJECT_ROOT / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from app.dependencies.rag_dependencies import get_rag_service

    return get_rag_service()


def normalize_text(value: str) -> str:
    return value.lower()


def get_source_value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)

    return getattr(source, key, None)


def has_expected_source_fields(
    expected_source_file: str | None,
    expected_page_number: int | None,
) -> bool:
    return expected_source_file is not None or expected_page_number is not None


def source_matches_expectation(
    source: Any,
    expected_source_file: str | None,
    expected_page_number: int | None,
) -> bool:
    if expected_source_file is not None:
        if get_source_value(source, "source_file") != expected_source_file:
            return False

    if expected_page_number is not None:
        if get_source_value(source, "page_number") != expected_page_number:
            return False

    return True


def compute_retrieval_hit(
    sources: list[Any],
    expected_source_file: str | None,
    expected_page_number: int | None,
) -> bool | str:
    if not has_expected_source_fields(expected_source_file, expected_page_number):
        return NOT_APPLICABLE

    return any(
        source_matches_expectation(
            source,
            expected_source_file=expected_source_file,
            expected_page_number=expected_page_number,
        )
        for source in sources
    )


def compute_source_accuracy(
    sources: list[Any],
    expected_source_file: str | None,
    expected_page_number: int | None,
) -> bool | str:
    if not has_expected_source_fields(expected_source_file, expected_page_number):
        return NOT_APPLICABLE

    if not sources:
        return False

    return all(
        source_matches_expectation(
            source,
            expected_source_file=expected_source_file,
            expected_page_number=expected_page_number,
        )
        for source in sources
    )


def compute_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> dict[str, Any]:
    if not expected_keywords:
        return {
            "status": NOT_APPLICABLE,
            "matched_keywords": [],
            "missing_keywords": [],
            "keyword_coverage_ratio": None,
        }

    normalized_answer = normalize_text(answer)
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_answer
    ]
    missing_keywords = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) not in normalized_answer
    ]

    return {
        "status": "applicable",
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_coverage_ratio": len(matched_keywords) / len(expected_keywords),
    }


def compute_refusal_accuracy(
    actual_status: str | None,
    expected_behavior: str | None,
    expected_status: str | None = None,
) -> bool:
    if expected_status is not None:
        return actual_status == expected_status

    expected_statuses = EXPECTED_BEHAVIOR_STATUSES.get(expected_behavior or "")
    if expected_statuses is not None:
        return actual_status in expected_statuses

    return False


def compute_citation_presence(
    sources: list[Any],
    expected_behavior: str | None,
) -> bool | str:
    if expected_behavior == "answer_with_sources":
        return bool(sources)

    if expected_behavior != "answer_with_sources":
        return NOT_APPLICABLE

    return False


def get_source_evidence_text(source: Any) -> str:
    evidence_parts: list[str] = []
    for key in SOURCE_EVIDENCE_TEXT_KEYS:
        value = get_source_value(source, key)
        if isinstance(value, str) and value:
            evidence_parts.append(value)

    return " ".join(evidence_parts)


def compute_citation_correctness(
    sources: list[Any],
    expected_behavior: str | None,
    expected_evidence_keywords: list[str],
) -> dict[str, Any]:
    if expected_behavior != "answer_with_sources":
        return {
            "expected_evidence_keywords": expected_evidence_keywords,
            "matched_evidence_keywords": [],
            "missing_evidence_keywords": [],
            "evidence_coverage_ratio": None,
            "citation_correctness_passed": NOT_APPLICABLE,
        }

    if not expected_evidence_keywords:
        return {
            "expected_evidence_keywords": [],
            "matched_evidence_keywords": [],
            "missing_evidence_keywords": [],
            "evidence_coverage_ratio": None,
            "citation_correctness_passed": NOT_APPLICABLE,
        }

    combined_evidence = normalize_text(
        " ".join(get_source_evidence_text(source) for source in sources)
    )
    matched_keywords = [
        keyword
        for keyword in expected_evidence_keywords
        if normalize_text(keyword) in combined_evidence
    ]
    missing_keywords = [
        keyword
        for keyword in expected_evidence_keywords
        if normalize_text(keyword) not in combined_evidence
    ]
    coverage_ratio = len(matched_keywords) / len(expected_evidence_keywords)

    return {
        "expected_evidence_keywords": expected_evidence_keywords,
        "matched_evidence_keywords": matched_keywords,
        "missing_evidence_keywords": missing_keywords,
        "evidence_coverage_ratio": coverage_ratio,
        "citation_correctness_passed": (
            coverage_ratio >= CITATION_CORRECTNESS_THRESHOLD
        ),
    }


def compute_metrics(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected_keywords = case.get("expected_keywords") or []
    expected_evidence_keywords = case.get("expected_evidence_keywords") or []
    expected_source_file = case.get("expected_source_file")
    expected_page_number = case.get("expected_page_number")
    expected_behavior = case.get("expected_behavior")
    expected_status = case.get("expected_status")

    answer = response.get("answer") or ""
    sources = response.get("sources") or []
    actual_status = response.get("status")

    return {
        "retrieval_hit": compute_retrieval_hit(
            sources,
            expected_source_file=expected_source_file,
            expected_page_number=expected_page_number,
        ),
        "source_accuracy": compute_source_accuracy(
            sources,
            expected_source_file=expected_source_file,
            expected_page_number=expected_page_number,
        ),
        "keyword_coverage": compute_keyword_coverage(
            answer,
            expected_keywords=expected_keywords,
        ),
        "refusal_accuracy": compute_refusal_accuracy(
            actual_status,
            expected_behavior=expected_behavior,
            expected_status=expected_status,
        ),
        "citation_presence": compute_citation_presence(
            sources,
            expected_behavior=expected_behavior,
        ),
        "citation_correctness": compute_citation_correctness(
            sources,
            expected_behavior=expected_behavior,
            expected_evidence_keywords=expected_evidence_keywords,
        ),
    }


def build_failed_checks(
    case: dict[str, Any],
    response: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    expected_behavior = case.get("expected_behavior")
    expected_fallback_reason = case.get("expected_fallback_reason")
    actual_status = response.get("status")
    actual_fallback_reason = response.get("fallback_reason")
    failed_checks: list[str] = []

    if expected_behavior == "answer_with_sources":
        if metrics["refusal_accuracy"] is False:
            failed_checks.append(
                f"expected_answered_status_got:{actual_status}"
            )

        if metrics["citation_presence"] is False:
            failed_checks.append("expected_sources_not_empty")

        keyword_coverage = metrics["keyword_coverage"]
        if keyword_coverage["status"] != NOT_APPLICABLE:
            missing_keywords = keyword_coverage["missing_keywords"]
            if missing_keywords:
                failed_checks.append(
                    "missing_expected_keywords:" + ",".join(missing_keywords)
                )

        if metrics["retrieval_hit"] is False:
            failed_checks.append("expected_retrieval_hit_missing")

        if metrics["source_accuracy"] is False:
            failed_checks.append("expected_source_accuracy_mismatch")

        citation_correctness = metrics["citation_correctness"]
        if citation_correctness["citation_correctness_passed"] is False:
            failed_checks.append("citation_correctness_failed")
    elif expected_behavior == "low_confidence_refusal":
        if metrics["refusal_accuracy"] is False:
            failed_checks.append(
                f"expected_low_confidence_status_got:{actual_status}"
            )

        trace = response.get("trace") or {}
        decision = trace.get("decision") if isinstance(trace, dict) else {}
        if isinstance(decision, dict) and decision.get("passed_gate") is True:
            failed_checks.append("confidence_gate_unexpectedly_passed")
    elif expected_behavior in EXPECTED_BEHAVIOR_STATUSES:
        if metrics["refusal_accuracy"] is False:
            failed_checks.append(
                f"expected_fallback_status_got:{actual_status}"
            )
    else:
        failed_checks.append(f"unknown_expected_behavior:{expected_behavior}")

    if expected_fallback_reason is not None:
        if actual_fallback_reason != expected_fallback_reason:
            failed_checks.append(
                f"expected_fallback_reason_got:{actual_fallback_reason}"
            )

    return failed_checks


def metric_rate(results: list[dict[str, Any]], metric_name: str) -> float | None:
    applicable_values = [
        result["metrics"][metric_name]
        for result in results
        if result["metrics"][metric_name] != NOT_APPLICABLE
    ]

    if not applicable_values:
        return None

    return sum(1 for value in applicable_values if value is True) / len(
        applicable_values
    )


def average_keyword_coverage(results: list[dict[str, Any]]) -> float | None:
    ratios = [
        result["metrics"]["keyword_coverage"]["keyword_coverage_ratio"]
        for result in results
        if result["metrics"]["keyword_coverage"]["status"] != NOT_APPLICABLE
    ]

    if not ratios:
        return None

    return sum(ratios) / len(ratios)


def citation_correctness_rate(results: list[dict[str, Any]]) -> float | None:
    applicable_values = [
        result["metrics"]["citation_correctness"][
            "citation_correctness_passed"
        ]
        for result in results
        if result["metrics"]["citation_correctness"][
            "citation_correctness_passed"
        ]
        != NOT_APPLICABLE
    ]

    if not applicable_values:
        return None

    return sum(1 for value in applicable_values if value is True) / len(
        applicable_values
    )


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed_cases = sum(1 for result in results if result["passed"])
    total_cases = len(results)

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": passed_cases / total_cases if total_cases else 0.0,
        "retrieval_hit_rate": metric_rate(results, "retrieval_hit"),
        "source_accuracy_rate": metric_rate(results, "source_accuracy"),
        "average_keyword_coverage": average_keyword_coverage(results),
        "refusal_accuracy_rate": metric_rate(results, "refusal_accuracy"),
        "citation_presence_rate": metric_rate(results, "citation_presence"),
        "citation_correctness_rate": citation_correctness_rate(results),
    }


def evaluate_result(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected_behavior = case.get("expected_behavior")
    actual_status = response.get("status")
    sources = response.get("sources") or []
    metrics = compute_metrics(case, response)
    failed_checks = build_failed_checks(case, response, metrics)

    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "expected_behavior": expected_behavior,
        "actual_status": actual_status,
        "fallback_reason": response.get("fallback_reason"),
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "trace_id": response.get("trace_id"),
        "metrics": metrics,
        "top_sources": sources[:3],
    }


def build_error_result(case: dict[str, Any], error: Exception) -> dict[str, Any]:
    response = {
        "status": "error",
        "answer": "",
        "sources": [],
        "trace_id": None,
        "fallback_reason": "error",
    }
    metrics = compute_metrics(case, response)

    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "expected_behavior": case.get("expected_behavior"),
        "actual_status": "error",
        "fallback_reason": "error",
        "passed": False,
        "failed_checks": [
            f"rag_service_error:{error.__class__.__name__}:{str(error)[:200]}"
        ],
        "trace_id": None,
        "metrics": metrics,
        "top_sources": [],
    }


def write_results(results: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def run_eval() -> int:
    cases = load_cases()
    results: list[dict[str, Any]] = []

    try:
        rag_service = load_rag_service()
        service_error: Exception | None = None
    except Exception as error:
        rag_service = None
        service_error = error

    for case in cases:
        if service_error is not None or rag_service is None:
            results.append(build_error_result(case, service_error))
            continue

        try:
            response = rag_service.ask(case["question"])
            results.append(evaluate_result(case, response))
        except Exception as error:
            results.append(build_error_result(case, error))

    summary = build_summary(results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }
    write_results(output)

    print(
        f"Eval complete: {summary['passed_cases']}/{summary['total_cases']} passed. "
        f"Results written to {RESULTS_PATH}"
    )

    return 0 if summary["passed_cases"] == summary["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(run_eval())
