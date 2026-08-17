from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACCESS_LEVELS = ("public", "internal", "confidential")
SHARED_DEPARTMENT = "all"
PERMISSION_METADATA_KEYS = ("tenant_id", "department", "access_level")


def _normalized(value: Any) -> str:
    return str(value or "").strip().casefold()


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str
    departments: frozenset[str]
    access_level: str = "public"

    def __post_init__(self) -> None:
        tenant_id = _normalized(self.tenant_id)
        departments = frozenset(
            value for value in (_normalized(item) for item in self.departments) if value
        )
        access_level = _normalized(self.access_level)
        if not tenant_id:
            raise ValueError("Access context tenant_id must not be empty.")
        if not departments:
            raise ValueError("Access context must include at least one department.")
        if access_level not in ACCESS_LEVELS:
            raise ValueError(f"Unknown access level: {self.access_level!r}.")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "departments", departments)
        object.__setattr__(self, "access_level", access_level)

    @property
    def allowed_access_levels(self) -> tuple[str, ...]:
        maximum = ACCESS_LEVELS.index(self.access_level)
        return ACCESS_LEVELS[: maximum + 1]


def validate_permission_metadata(metadata: dict[str, Any]) -> None:
    present = [key for key in PERMISSION_METADATA_KEYS if metadata.get(key) is not None]
    if not present:
        return
    if len(present) != len(PERMISSION_METADATA_KEYS):
        raise ValueError(
            "Permission metadata must provide tenant_id, department, and access_level together."
        )
    if not _normalized(metadata["tenant_id"]):
        raise ValueError("Permission metadata tenant_id must not be empty.")
    if not _normalized(metadata["department"]):
        raise ValueError("Permission metadata department must not be empty.")
    if _normalized(metadata["access_level"]) not in ACCESS_LEVELS:
        raise ValueError(f"Unknown access level: {metadata['access_level']!r}.")


def normalized_permission_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    validate_permission_metadata(metadata)
    if not any(metadata.get(key) is not None for key in PERMISSION_METADATA_KEYS):
        return {}
    return {key: _normalized(metadata[key]) for key in PERMISSION_METADATA_KEYS}


def is_metadata_accessible(
    metadata: dict[str, Any] | None,
    access_context: AccessContext | None,
) -> bool:
    if access_context is None:
        return True
    stored = metadata or {}
    validate_permission_metadata(stored)
    if any(stored.get(key) is None for key in PERMISSION_METADATA_KEYS):
        return False
    document_department = _normalized(stored["department"])
    return (
        _normalized(stored["tenant_id"]) == access_context.tenant_id
        and (
            document_department == SHARED_DEPARTMENT
            or document_department in access_context.departments
        )
        and _normalized(stored["access_level"]) in access_context.allowed_access_levels
    )


def chroma_permission_filter(access_context: AccessContext) -> dict[str, Any]:
    return {
        "$and": [
            {"status": {"$eq": "ACTIVE"}},
            {"tenant_id": {"$eq": access_context.tenant_id}},
            {
                "department": {
                    "$in": sorted(access_context.departments | {SHARED_DEPARTMENT})
                }
            },
            {"access_level": {"$in": list(access_context.allowed_access_levels)}},
        ]
    }
