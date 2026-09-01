from __future__ import annotations

from typing import Any, TypedDict

from schemas.chat_ui import ChatUIRequest, ChatUIResponse
from schemas.user_context import UserContext


class ChatUIWorkflowState(TypedDict, total=False):
    """In-memory state for the production Chat UI LangGraph workflow.

    Database clients and service instances remain request-scoped runtime
    dependencies. V4 checkpoints only a JSON-safe projection through the
    dedicated Agent runtime store; the container itself is never serialized.
    """

    body: ChatUIRequest
    user_context: UserContext
    container: Any
    workflow_version: str
    workflow_id: str
    workflow_status: str
    resuming: bool
    resume_count: int
    resume_cached: bool
    pause_after: str | None
    primary_family: str
    deterministic_kind: str
    semantic_family: str
    history: list[dict[str, str]]
    attachment_meta: list[dict[str, Any]]
    database_explorer_enabled: bool
    intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    router_name: str
    reasoning_summary: str
    routing_meta: dict[str, Any]
    needs_clarification: bool
    clarification_question: str
    response: ChatUIResponse
