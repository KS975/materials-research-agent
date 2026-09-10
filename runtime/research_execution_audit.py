from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


_LOCK = Lock()
_AUDIT_PATH = Path(".runtime/research_execution_audit.jsonl")


def record_research_execution(
    *,
    user_id: str,
    company_id: str,
    scenario_id: int,
    workflow_id: str,
    strategy: str,
    status: str,
    evidence_count: int,
    vector_hit_count: int,
    synthesis_status: str,
    missing_dimensions: list[str],
) -> None:
    """Append a compact execution audit record without query text or secrets."""
    record = {
        "user_id": str(user_id),
        "company_id": str(company_id),
        "scenario_id": int(scenario_id),
        "workflow_id": str(workflow_id),
        "strategy": str(strategy),
        "status": str(status),
        "evidence_count": int(evidence_count),
        "vector_hit_count": int(vector_hit_count),
        "synthesis_status": str(synthesis_status),
        "missing_dimensions": list(missing_dimensions),
    }
    _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with _AUDIT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_research_executions(
    *,
    user_id: str,
    company_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    normalized_limit = max(1, min(int(limit), 200))
    if not _AUDIT_PATH.exists():
        return []
    with _LOCK:
        lines = _AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    output: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("user_id")) != str(user_id)
            or str(item.get("company_id")) != str(company_id)
        ):
            continue
        output.append(item)
        if len(output) >= normalized_limit:
            break
    return output
