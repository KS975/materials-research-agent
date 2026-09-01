from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.chat import resolve_user_context
from runtime.v014_ui import (
    V014UIError,
    infer_batch_size,
    infer_bo_target_metric,
    run_inverse_design_for_ui,
    run_next_experiments_for_ui,
)
from schemas.user_context import UserContext


router = APIRouter(prefix="/api/v1/optimization-ui", tags=["optimization-ui"])


def runtime_root() -> Path:
    override = os.getenv("V014_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


class InverseDesignRequest(BaseModel):
    project_id: int = Field(ge=1)
    message: str = Field(min_length=1, max_length=2000)
    candidate_count: int = Field(default=600, ge=50, le=5000)
    random_state: int = 42


class NextExperimentsRequest(BaseModel):
    project_id: int = Field(ge=1)
    message: str = Field(min_length=1, max_length=2000)
    target_metric: str | None = Field(default=None, max_length=80)
    batch_size: int | None = Field(default=None, ge=1, le=20)
    candidate_count: int = Field(default=900, ge=50, le=10000)
    random_state: int = 42


def _check_project(ctx: UserContext, project_id: int) -> None:
    if not ctx.can_access_project(project_id):
        raise HTTPException(status_code=403, detail="当前用户无权执行该项目的 V0.1.4 优化")


@router.post("/inverse-design")
def inverse_design(
    body: InverseDesignRequest,
    ctx: UserContext = Depends(resolve_user_context),
):
    _check_project(ctx, body.project_id)
    try:
        report = run_inverse_design_for_ui(
            runtime_root=runtime_root(),
            project_id=body.project_id,
            message=body.message,
            candidate_count=body.candidate_count,
            random_state=body.random_state,
        )
    except V014UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": report.get("answer", ""), "data": report}


@router.post("/next-experiments")
def next_experiments(
    body: NextExperimentsRequest,
    ctx: UserContext = Depends(resolve_user_context),
):
    _check_project(ctx, body.project_id)
    root = runtime_root()
    try:
        target_metric = (
            body.target_metric.strip()
            if body.target_metric and body.target_metric.strip()
            else infer_bo_target_metric(body.message, root, body.project_id)
        )
        batch_size = body.batch_size or infer_batch_size(body.message, 5)
        report = run_next_experiments_for_ui(
            runtime_root=root,
            project_id=body.project_id,
            target_metric=target_metric,
            batch_size=batch_size,
            candidate_count=body.candidate_count,
            random_state=body.random_state,
        )
    except V014UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": report.get("answer", ""), "data": report}
