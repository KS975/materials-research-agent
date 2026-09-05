from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EngineTaskIntent = Literal[
    "engine_prepare_dataset",
    "automl_training",
    "predict_performance",
    "optimize_formula",
    "recommend_next_experiments",
]


class EngineTaskCreateRequest(BaseModel):
    intent: EngineTaskIntent
    tool_name: str = Field(min_length=1, max_length=120)
    tool_args: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,63}$",
    )


class EngineTaskActionRequest(BaseModel):
    reason: str = Field(default="", max_length=500)


class EngineTaskStatusResponse(BaseModel):
    schema_version: int
    task_id: str
    status: str
    intent: str
    tool_name: str
    scenario_plan: dict[str, Any]
    current_stage: str
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    resume_count: int
    approval: dict[str, Any] | None = None
    answer: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
