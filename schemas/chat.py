from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    tool_name: str | None = None
    data: dict[str, Any] | list[Any] | None = None
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
