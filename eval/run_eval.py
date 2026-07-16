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

ANSWERED_STATUSES = {"answered", "success"}
LOW_CONFIDENCE_STATUSES = {
    "low_confidence",
    "refused",
    "insufficient_evidence",
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


def source_matches(
    sources: list[Any],
    expected_source_file: str | None,
    expected_page_number: int | None,
) -> list[str]:
    failed_checks: list[str] = []

    if expected_source_file is not None:
        if not any(
            get_source_value(source, "source_file") == expected_source_file
            for source in sources
        ):
            failed_checks.append(
                f"expected_source_file_not_found:{expected_source_file}"
            )

    if expected_page_number is not None:
        if not any(
            get_source_value(source, "page_number") == expected_page_number
            for source in sources
        ):
            failed_checks.append(
                f"expected_page_number_not_found:{expected_page_number}"
            )

    return failed_checks


def evaluate_result(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected_behavior = case.get("expected_behavior")
    expected_keywords = case.get("expected_keywords") or []
    expected_source_file = case.get("expected_source_file")
    expected_page_number = case.get("expected_page_number")

    actual_status = response.get("status")
    answer = response.get("answer") or ""
    sources = response.get("sources") or []
    failed_checks: list[str] = []

    if expected_behavior == "answer_with_sources":
        if actual_status not in ANSWERED_STATUSES:
            failed_checks.append(
                f"expected_answered_status_got:{actual_status}"
            )
        if not sources:
            failed_checks.append("expected_sources_not_empty")

        normalized_answer = normalize_text(answer)
        missing_keywords = [
            keyword
            for keyword in expected_keywords
            if normalize_text(keyword) not in normalized_answer
        ]
        if missing_keywords:
            failed_checks.append(
                "missing_expected_keywords:" + ",".join(missing_keywords)
            )

        failed_checks.extend(
            source_matches(
                sources,
                expected_source_file=expected_source_file,
                expected_page_number=expected_page_number,
            )
        )
    elif expected_behavior == "low_confidence_refusal":
        if actual_status not in LOW_CONFIDENCE_STATUSES:
            failed_checks.append(
                f"expected_low_confidence_status_got:{actual_status}"
            )

        trace = response.get("trace") or {}
        decision = trace.get("decision") if isinstance(trace, dict) else {}
        if isinstance(decision, dict) and decision.get("passed_gate") is True:
            failed_checks.append("confidence_gate_unexpectedly_passed")
    else:
        failed_checks.append(f"unknown_expected_behavior:{expected_behavior}")

    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "expected_behavior": expected_behavior,
        "actual_status": actual_status,
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "trace_id": response.get("trace_id"),
        "top_sources": sources[:3],
    }


def build_error_result(case: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "expected_behavior": case.get("expected_behavior"),
        "actual_status": "error",
        "passed": False,
        "failed_checks": [
            f"rag_service_error:{error.__class__.__name__}:{str(error)[:200]}"
        ],
        "trace_id": None,
        "top_sources": [],
    }


def write_results(results: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    with path.open("w", encoding="utf-8") as file:
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

    passed_count = sum(1 for result in results if result["passed"])
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "results": results,
    }
    write_results(output)

    print(
        f"Eval complete: {passed_count}/{len(results)} passed. "
        f"Results written to {RESULTS_PATH}"
    )

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_eval())
