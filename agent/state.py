from __future__ import annotations

from typing import Any, TypedDict

from schemas.user_context import UserContext


class AgentState(TypedDict, total=False):
    message: str
    user_context: UserContext
    intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: Any
    evidence: list[dict[str, Any]]
    warnings: list[str]
    answer: str
