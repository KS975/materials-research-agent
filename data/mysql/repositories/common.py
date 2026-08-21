from __future__ import annotations

from typing import Sequence


def project_scope_clause(
    project_ids: Sequence[int],
    column: str = "project_id",
    *,
    allow_all: bool = False,
) -> tuple[str, list[int]]:
    """Build the project part of a company-scoped read-only query.

    ``allow_all=True`` means all projects *inside the already-required company
    scope*. It must only be supplied from an explicitly authorized
    ``UserContext.all_projects`` flag.

    An empty ordinary project list remains fail-closed.
    """
    if allow_all:
        return "1 = 1", []

    ids = [int(x) for x in project_ids]
    if not ids:
        # Fail closed; never interpret an empty ordinary permission scope as
        # "all projects".
        return "1 = 0", []
    placeholders = ", ".join(["%s"] * len(ids))
    return f"{column} IN ({placeholders})", ids


def bounded_limit(value: int, *, default: int = 20, maximum: int = 100) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(number, maximum))
