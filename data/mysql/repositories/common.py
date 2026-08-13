from __future__ import annotations

from typing import Sequence


def project_scope_clause(project_ids: Sequence[int], column: str = "project_id") -> tuple[str, list[int]]:
    ids = [int(x) for x in project_ids]
    if not ids:
        # Fail closed; never interpret empty permission scope as "all projects".
        return "1 = 0", []
    placeholders = ", ".join(["%s"] * len(ids))
    return f"{column} IN ({placeholders})", ids


def bounded_limit(value: int, *, default: int = 20, maximum: int = 100) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(number, maximum))
