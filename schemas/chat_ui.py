from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatUIRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    attachment_ids: list[str] = Field(default_factory=list, max_length=8)
    attachment_reference_mode: bool = False
    workflow_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,63}$",
    )
    resume_workflow: bool = False
    pause_after: Literal["classify_primary"] | None = None
    conversation_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F-]{36}$",
    )
    client_message_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=96,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,95}$",
    )


class ChatUIResponse(BaseModel):
    answer: str
    intent: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    data: Any = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    router: str = "deepseek"
    reasoning_summary: str = ""
    routing: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None


class ChatHistoryRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
