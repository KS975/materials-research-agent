from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas.user_context import UserContext


class EngineWorkflowToolError(RuntimeError):
    """Carry a structured engine-tool failure without leaking an exception."""

    def __init__(self, result: dict[str, Any]):
        error = dict(result.get("error") or {})
        super().__init__(str(error.get("message") or "engine tool failed"))
        self.result = result


@dataclass(frozen=True, slots=True)
class ArtifactScope:
    ctx: UserContext
    company_id: str
    project_id: int
    project_root: Path
    model_registry_path: Path
    session_root: Path
    conversation_id: str


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    path: Path
    source_hash: str
    records: list[dict[str, Any]]
    catalog: dict[str, Any]
    numeric_feature_fields: list[str]
    sample_count: int
    warnings: list[str]
