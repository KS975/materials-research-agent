from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.chat import resolve_user_context
from runtime.v030_ui import (
    V030UIError,
    build_autonomy_overview,
    operator_override_for_ui,
)
from schemas.user_context import UserContext


router = APIRouter(
    prefix="/api/v1/autonomy-ui",
    tags=["autonomy-ui"],
)


def runtime_root() -> Path:
    override = os.getenv("V030_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


def _check_campaign_access(
    ctx: UserContext,
    campaign_id: str,
) -> dict:
    try:
        view = build_autonomy_overview(
            runtime_root(), campaign_id=campaign_id
        )
    except V030UIError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    project_id = int(view["campaign"]["project_id"])
    if not ctx.can_access_project(project_id):
        raise HTTPException(
            status_code=403,
            detail="当前用户无权访问该 V0.3 autonomous campaign",
        )
    return view


class OperatorOverrideRequest(BaseModel):
    campaign_id: str = Field(min_length=1, max_length=160)
    round_id: str = Field(min_length=1, max_length=180)
    action: str = Field(min_length=1, max_length=32)
    reason: str = Field(min_length=1, max_length=1000)


@router.get("/status")
def status(
    campaign_id: str | None = Query(default=None),
    project_id: int | None = Query(default=None, ge=1),
    ctx: UserContext = Depends(resolve_user_context),
):
    try:
        data = build_autonomy_overview(
            runtime_root(),
            campaign_id=campaign_id,
            project_id=project_id,
        )
    except V030UIError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not ctx.can_access_project(int(data["campaign"]["project_id"])):
        raise HTTPException(
            status_code=403,
            detail="当前用户无权访问该项目 V0.3 autonomous runtime",
        )
    return {"answer": data["answer"], "data": data}


@router.post("/operator")
def operator_override(
    body: OperatorOverrideRequest,
    ctx: UserContext = Depends(resolve_user_context),
):
    _check_campaign_access(ctx, body.campaign_id)
    try:
        data = operator_override_for_ui(
            runtime_root(),
            campaign_id=body.campaign_id,
            round_id=body.round_id,
            action=body.action,
            operator_id=ctx.user_id,
            reason=body.reason,
        )
    except V030UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}
