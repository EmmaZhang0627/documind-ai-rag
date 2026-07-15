import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from app.observability.trace_models import RAGTraceRecord


logger = logging.getLogger(__name__)
TRACE_LOG_PATH = Path("logs") / "rag_traces.jsonl"


def write_trace(record: RAGTraceRecord, log_path: Path = TRACE_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        logger.info(
            "rag_trace_lifecycle trace_id=%s query_length=%s top1_score=%s "
            "llm_called=%s status=%s",
            record["trace_id"],
            record["query_length"],
            record["top1_score"],
            record["llm_called"],
            record["final_status"],
        )
    except Exception as error:
        logger.warning("rag_trace_write_failed: %s", error)


def read_latest_traces(
    limit: int = 20,
    log_path: Path = TRACE_LOG_PATH,
) -> list[dict[str, Any]]:
    try:
        if limit <= 0 or not log_path.exists():
            return []

        latest_lines: deque[str] = deque(maxlen=limit)
        with log_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    latest_lines.append(line)

        records: list[dict[str, Any]] = []
        for line in latest_lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("rag_trace_read_skipped_invalid_jsonl")

        return records
    except Exception as error:
        logger.warning("rag_trace_read_failed: %s", error)
        return []
