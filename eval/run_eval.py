from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
FIXTURES_DIR = EVAL_DIR / "fixtures"
CASES_PATH = EVAL_DIR / "documind_eval_cases.json"
RESULTS_PATH = EVAL_DIR / "eval_results_latest.json"
NOT_APPLICABLE = "not_applicable"
SOURCE_EVIDENCE_TEXT_KEYS = (
    "source_snippet",
    "snippet",
    "chunk_text",
    "content",
    "text",
    "document",
)
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_*.-]+")


def sanitize_error_message(message: str) -> str:
    return SECRET_PATTERN.sub("sk-***redacted***", message)


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON list.")

    return cases


def discover_fixture_pdf(fixtures_dir: Path = FIXTURES_DIR) -> Path:
    if not fixtures_dir.exists():
        raise FileNotFoundError(
            f"Fixture directory is required but does not exist: {fixtures_dir}"
        )

    pdf_paths = sorted(fixtures_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"At least one fixture PDF is required under: {fixtures_dir}"
        )

    if len(pdf_paths) > 1:
        names = ", ".join(path.name for path in pdf_paths)
        raise ValueError(
            "Expected exactly one fixture PDF for repeatable evaluation; "
            f"found {len(pdf_paths)}: {names}"
        )

    return pdf_paths[0]


def ensure_backend_import_path() -> None:
    backend_path = PROJECT_ROOT / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))


def load_rag_service():
    ensure_backend_import_path()

    from app.dependencies.rag_dependencies import get_rag_service

    return get_rag_service()


def clear_eval_process_vector_store() -> None:
    ensure_backend_import_path()

    from app.services.vector_db import clear_vector_store

    clear_vector_store()


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        pages: list[dict[str, Any]] = []
        for page_index, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                pages.append({
                    "page_number": page_index + 1,
                    "text": page_text,
                })

        return pages
    finally:
        doc.close()


def ingest_fixture_pdf(rag_service: Any, fixture_pdf: Path) -> dict[str, Any]:
    ensure_backend_import_path()

    from app.services.chunker import split_pages_into_chunks

    pages = extract_pdf_pages(fixture_pdf)
    document_id = str(uuid4())
    chunks = split_pages_into_chunks(
        pages=pages,
        document_id=document_id,
        source_file=fixture_pdf.name,
    )
    rag_service.ingest_document(chunks)

    return {
        "document_id": document_id,
        "fixture_pdf": fixture_pdf.name,
        "fixture_path": str(fixture_pdf),
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }


def prepare_eval_index(rag_service: Any, fixture_pdf: Path) -> dict[str, Any]:
    clear_eval_process_vector_store()
    return ingest_fixture_pdf(rag_service, fixture_pdf)


def normalize_text(value: str) -> str:
    return value.casefold()


def get_source_value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)

    return getattr(source, key, None)


def get_expected_page_numbers(case: dict[str, Any]) -> list[int]:
    value = case.get("expected_page_numbers")
    if value is None and case.get("expected_page_number") is not None:
        value = [case["expected_page_number"]]

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"Case {case.get('id')} expected_page_numbers must be a list."
        )

    return [int(page_number) for page_number in value]


def get_retrieved_page_numbers(sources: list[Any]) -> list[int]:
    page_numbers: set[int] = set()
    for source in sources:
        page_number = get_source_value(source, "page_number")
        if page_number is not None:
            page_numbers.add(int(page_number))

    return sorted(page_numbers)


def get_source_files(sources: list[Any]) -> list[str]:
    source_files: list[str] = []
    for source in sources:
        source_file = get_source_value(source, "source_file")
        if isinstance(source_file, str):
            source_files.append(source_file)

    return source_files


def get_source_evidence_text(source: Any) -> str:
    evidence_parts: list[str] = []
    for key in SOURCE_EVIDENCE_TEXT_KEYS:
        value = get_source_value(source, key)
        if isinstance(value, str) and value:
            evidence_parts.append(value)

    return " ".join(evidence_parts)


def match_keywords(text: str, expected_keywords: list[str]) -> dict[str, Any]:
    if not expected_keywords:
        return {
            "applicable": False,
            "passed": NOT_APPLICABLE,
            "matched": [],
            "missing": [],
        }

    normalized_text = normalize_text(text)
    matched = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_text
    ]
    missing = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) not in normalized_text
    ]

    return {
        "applicable": True,
        "passed": not missing,
        "matched": matched,
        "missing": missing,
    }


def build_check_result(
    applicable: bool,
    passed: bool | str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "applicable": applicable,
        "passed": passed,
        "details": details or {},
    }


def compute_checks(
    case: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    sources = response.get("sources") or []
    answer = response.get("answer") or ""
    actual_status = response.get("status")
    actual_fallback_reason = response.get("fallback_reason")
    expected_status = case.get("expected_status")
    expected_source_file = case.get("expected_source_file")
    expected_page_numbers = get_expected_page_numbers(case)
    expected_evidence_keywords = case.get("expected_evidence_keywords") or []
    expected_answer_keywords = case.get("expected_answer_keywords") or []
    expected_fallback_reason = case.get("expected_fallback_reason")

    source_files = get_source_files(sources)
    retrieved_page_numbers = get_retrieved_page_numbers(sources)
    source_evidence_text = " ".join(
        get_source_evidence_text(source) for source in sources
    )
    evidence_keyword_result = match_keywords(
        source_evidence_text,
        expected_evidence_keywords,
    )
    answer_keyword_result = match_keywords(answer, expected_answer_keywords)
    fallback_applicable = (
        expected_fallback_reason is not None
        or (
            expected_status is not None
            and expected_status != "answered"
        )
    )

    return {
        "status": build_check_result(
            applicable=expected_status is not None,
            passed=actual_status == expected_status
            if expected_status is not None
            else NOT_APPLICABLE,
            details={
                "expected_status": expected_status,
                "actual_status": actual_status,
            },
        ),
        "fallback": build_check_result(
            applicable=fallback_applicable,
            passed=(
                actual_status == expected_status
                and (
                    expected_fallback_reason is None
                    or actual_fallback_reason == expected_fallback_reason
                )
            )
            if fallback_applicable
            else NOT_APPLICABLE,
            details={
                "expected_fallback_reason": expected_fallback_reason,
                "actual_fallback_reason": actual_fallback_reason,
            },
        ),
        "source_match": build_check_result(
            applicable=expected_source_file is not None,
            passed=expected_source_file in source_files
            if expected_source_file is not None
            else NOT_APPLICABLE,
            details={
                "expected_source_file": expected_source_file,
                "returned_source_files": source_files,
            },
        ),
        "page_match": build_check_result(
            applicable=bool(expected_page_numbers),
            passed=all(
                page_number in retrieved_page_numbers
                for page_number in expected_page_numbers
            )
            if expected_page_numbers
            else NOT_APPLICABLE,
            details={
                "expected_page_numbers": expected_page_numbers,
                "retrieved_page_numbers": retrieved_page_numbers,
            },
        ),
        "evidence_keywords": build_check_result(
            applicable=evidence_keyword_result["applicable"],
            passed=evidence_keyword_result["passed"],
            details={
                "expected_keywords": expected_evidence_keywords,
                "matched_keywords": evidence_keyword_result["matched"],
                "missing_keywords": evidence_keyword_result["missing"],
            },
        ),
        "answer_keywords": build_check_result(
            applicable=answer_keyword_result["applicable"],
            passed=answer_keyword_result["passed"],
            details={
                "expected_keywords": expected_answer_keywords,
                "matched_keywords": answer_keyword_result["matched"],
                "missing_keywords": answer_keyword_result["missing"],
            },
        ),
    }


def build_failed_checks(checks: dict[str, Any]) -> list[str]:
    failed_checks: list[str] = []
    for check_name, check_result in checks.items():
        if check_result["applicable"] and check_result["passed"] is not True:
            failed_checks.append(check_name)

    return failed_checks


def evaluate_result(
    case: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    sources = response.get("sources") or []
    checks = compute_checks(case, response)
    failed_checks = build_failed_checks(checks)

    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "expected_status": case.get("expected_status"),
        "actual_status": response.get("status"),
        "fallback_reason": response.get("fallback_reason"),
        "returned_sources": sources,
        "retrieved_page_numbers": get_retrieved_page_numbers(sources),
        "checks": checks,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "trace_id": response.get("trace_id"),
        "retrieval_trace": response.get("trace"),
        "answer": response.get("answer"),
    }


def build_error_result(case: dict[str, Any], error: Exception) -> dict[str, Any]:
    response = {
        "status": "error",
        "answer": "",
        "sources": [],
        "trace_id": None,
        "fallback_reason": "error",
        "trace": {},
    }
    checks = compute_checks(case, response)
    failed_checks = build_failed_checks(checks)
    failed_checks.append(f"rag_service_error:{error.__class__.__name__}")

    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "expected_status": case.get("expected_status"),
        "actual_status": "error",
        "fallback_reason": "error",
        "returned_sources": [],
        "retrieved_page_numbers": [],
        "checks": checks,
        "passed": False,
        "failed_checks": failed_checks,
        "trace_id": None,
        "retrieval_trace": {},
        "answer": "",
        "error_message": sanitize_error_message(str(error))[:300],
    }


def count_passed_check(results: list[dict[str, Any]], check_name: str) -> int:
    return sum(
        1
        for result in results
        if result["checks"][check_name]["applicable"]
        and result["checks"][check_name]["passed"] is True
    )


def count_applicable_check(results: list[dict[str, Any]], check_name: str) -> int:
    return sum(
        1
        for result in results
        if result["checks"][check_name]["applicable"]
    )


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["passed"])

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "source_hit_count": count_passed_check(results, "source_match"),
        "source_applicable_count": count_applicable_check(results, "source_match"),
        "page_hit_count": count_passed_check(results, "page_match"),
        "page_applicable_count": count_applicable_check(results, "page_match"),
        "evidence_keyword_match_count": count_passed_check(
            results,
            "evidence_keywords",
        ),
        "evidence_keyword_applicable_count": count_applicable_check(
            results,
            "evidence_keywords",
        ),
        "answer_keyword_match_count": count_passed_check(
            results,
            "answer_keywords",
        ),
        "answer_keyword_applicable_count": count_applicable_check(
            results,
            "answer_keywords",
        ),
        "fallback_correctness_count": count_passed_check(results, "fallback"),
        "fallback_applicable_count": count_applicable_check(results, "fallback"),
    }


def write_results(results: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
    print("Eval complete")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed cases: {summary['passed_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    print(
        "Source hits: "
        f"{summary['source_hit_count']}/"
        f"{summary['source_applicable_count']}"
    )
    print(
        "Page hits: "
        f"{summary['page_hit_count']}/"
        f"{summary['page_applicable_count']}"
    )
    print(
        "Fallback correctness: "
        f"{summary['fallback_correctness_count']}/"
        f"{summary['fallback_applicable_count']}"
    )
    print(f"Results written to: {RESULTS_PATH}")


def build_setup_error_output(
    cases: list[dict[str, Any]],
    error: Exception,
    fixture_pdf: Path | None,
) -> dict[str, Any]:
    results = [build_error_result(case, error) for case in cases]
    summary = build_summary(results)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "fixture_pdf": fixture_pdf.name if fixture_pdf is not None else None,
            "fixture_path": str(fixture_pdf) if fixture_pdf is not None else None,
        },
        "setup_error": {
            "type": error.__class__.__name__,
            "message": sanitize_error_message(str(error))[:500],
        },
        "summary": summary,
        "results": results,
    }


def run_eval() -> int:
    cases = load_cases()
    fixture_pdf: Path | None = None

    try:
        fixture_pdf = discover_fixture_pdf()
        rag_service = load_rag_service()
        indexing = prepare_eval_index(rag_service, fixture_pdf)
    except Exception as error:
        output = build_setup_error_output(cases, error, fixture_pdf)
        write_results(output)
        print_summary(output["summary"])
        print(
            "Setup failed: "
            f"{error.__class__.__name__}: "
            f"{sanitize_error_message(str(error))[:200]}"
        )
        return 1

    results: list[dict[str, Any]] = []

    for case in cases:
        try:
            response = rag_service.ask(
                case["question"],
                top_k=case.get("top_k"),
            )
            results.append(evaluate_result(case, response))
        except Exception as error:
            results.append(build_error_result(case, error))

    summary = build_summary(results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": indexing,
        "summary": summary,
        "results": results,
    }
    write_results(output)
    print_summary(summary)

    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_eval())
