from typing import Any

from fastapi import APIRouter, Query

from app.observability.trace_logger import read_latest_traces


router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("/latest")
def latest_traces(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    traces = read_latest_traces(limit=limit)
    return {
        "count": len(traces),
        "traces": traces,
    }
