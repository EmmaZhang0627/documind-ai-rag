from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
RESULTS_PATH = EVAL_DIR / "permission_filtering_benchmark_latest.json"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.metadata_permissions import AccessContext
from app.services.retrieval_service import InMemoryVectorStore, RetrievalService


def make_chunk(
    document_id: str,
    embedding: list[float],
    *,
    tenant_id: str,
    department: str,
    access_level: str,
    content: str,
    status: str = "ACTIVE",
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_index": 0,
        "source_file": f"{document_id}.pdf",
        "page_number": 1,
        "content": content,
        "embedding": embedding,
        "tenant_id": tenant_id,
        "department": department,
        "access_level": access_level,
        "status": status,
    }


def run() -> int:
    store = InMemoryVectorStore()
    store.clear()
    retriever = RetrievalService(store)
    chunks = [
        make_chunk(
            "tenant-a-handbook", [0.82, 0.18], tenant_id="tenant-a",
            department="all", access_level="public",
            content="Public employee travel policy and general guidance.",
        ),
        make_chunk(
            "tenant-a-finance", [1.0, 0.0], tenant_id="tenant-a",
            department="finance", access_level="confidential",
            content="Confidential acquisition budget codeword ORCHID.",
        ),
        make_chunk(
            "tenant-a-engineering", [0.9, 0.1], tenant_id="tenant-a",
            department="engineering", access_level="internal",
            content="Internal engineering deployment procedure.",
        ),
        make_chunk(
            "tenant-b-finance", [0.99, 0.01], tenant_id="tenant-b",
            department="finance", access_level="confidential",
            content="Tenant B confidential acquisition budget codeword ORCHID.",
        ),
        make_chunk(
            "tenant-a-archived", [0.98, 0.02], tenant_id="tenant-a",
            department="finance", access_level="confidential", status="ARCHIVED",
            content="Archived acquisition budget codeword ORCHID.",
        ),
    ]
    retriever.add(chunks)
    contexts = {
        "a_public": AccessContext("tenant-a", frozenset({"general"}), "public"),
        "a_finance": AccessContext(
            "tenant-a", frozenset({"finance"}), "confidential"
        ),
        "a_engineering": AccessContext(
            "tenant-a", frozenset({"engineering"}), "internal"
        ),
        "b_finance": AccessContext(
            "tenant-b", frozenset({"finance"}), "confidential"
        ),
    }
    scenarios = [
        ("allowed_public_document", "a_public", "tenant-a-handbook", {"tenant-a-handbook"}),
        ("unauthorized_confidential_document", "a_public", "tenant-a-handbook", {"tenant-a-handbook"}),
        ("finance_context_changes_candidates", "a_finance", "tenant-a-finance", {"tenant-a-handbook", "tenant-a-finance"}),
        ("strongest_restricted_match_excluded", "a_engineering", "tenant-a-engineering", {"tenant-a-handbook", "tenant-a-engineering"}),
        ("archived_and_unauthorized_compose", "a_finance", "tenant-a-finance", {"tenant-a-handbook", "tenant-a-finance"}),
        ("tenant_b_isolation", "b_finance", "tenant-b-finance", {"tenant-b-finance"}),
    ]
    active_count = sum(chunk["status"] == "ACTIVE" for chunk in chunks)
    results = []
    leakage_count = 0
    source_correct_count = 0
    try:
        for scenario_id, context_name, expected_top1, allowed_ids in scenarios:
            ranked = retriever.retrieve(
                [1.0, 0.0], "confidential acquisition budget codeword ORCHID",
                top_k=10, access_context=contexts[context_name],
            )
            candidate_ids = [item["metadata"]["document_id"] for item in ranked]
            leaked_ids = sorted(set(candidate_ids) - allowed_ids)
            source_correct = bool(candidate_ids) and candidate_ids[0] == expected_top1
            leakage_count += len(leaked_ids)
            source_correct_count += int(source_correct)
            results.append({
                "scenario_id": scenario_id,
                "access_context": context_name,
                "candidates_before_permission_filter": active_count,
                "candidates_after_permission_filter": len(candidate_ids),
                "candidate_document_ids": candidate_ids,
                "expected_top1_document_id": expected_top1,
                "top1_source_correct": source_correct,
                "unauthorized_document_ids": leaked_ids,
            })
    finally:
        store.clear()

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "evaluation-only metadata permission filtering MVP",
        "scenario_count": len(results),
        "unauthorized_evidence_leakage_count": leakage_count,
        "correct_accessible_source_count": source_correct_count,
        "source_accuracy": source_correct_count / len(results),
        "passed": leakage_count == 0 and source_correct_count == len(results),
        "scenarios": results,
        "notes": [
            "The benchmark uses deterministic synthetic chunks and the production in-memory Hybrid retrieval path.",
            "Authentication and identity verification are intentionally outside this MVP.",
        ],
    }
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
