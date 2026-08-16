from copy import deepcopy

from app.services.rag_types import Candidate


def resolve_parent_context(candidates: list[Candidate]) -> list[Candidate]:
    resolved: list[Candidate] = []
    seen_parent_ids: set[str] = set()

    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        parent_id = metadata.get("parent_id")
        parent_text = metadata.get("parent_text")
        if not parent_id or not isinstance(parent_text, str) or not parent_text.strip():
            resolved.append(deepcopy(candidate))
            continue
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        parent_candidate = deepcopy(candidate)
        parent_candidate["document"] = parent_text
        resolved.append(parent_candidate)

    return resolved
