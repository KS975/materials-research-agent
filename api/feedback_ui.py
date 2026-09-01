from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.chat import resolve_user_context
from runtime.v020_ui import (
    V020UIError,
    advance_campaign_for_ui,
    approve_model_for_ui,
    build_campaign_overview,
    close_round_for_ui,
    start_round_for_ui,
    submit_result_for_ui,
)
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1/feedback-ui", tags=["feedback-ui"])


def runtime_root() -> Path:
    override = os.getenv("V020_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


def _check_campaign_access(ctx: UserContext, campaign_id: str) -> dict[str, Any]:
    try:
        overview = build_campaign_overview(runtime_root(), campaign_id=campaign_id)
    except V020UIError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    project_id = int(overview["campaign"]["project_id"])
    if not ctx.can_access_project(project_id):
        raise HTTPException(status_code=403, detail="当前用户无权访问该 V0.2 Campaign")
    return overview


class RoundActionRequest(BaseModel):
    campaign_id: str = Field(min_length=1, max_length=120)
    round_id: str = Field(min_length=1, max_length=160)


class ResultSubmissionRequest(RoundActionRequest):
    candidate_id: str = Field(min_length=1, max_length=160)
    status: str = Field(min_length=1, max_length=32)
    test_condition_signature: str = Field(min_length=1, max_length=240)
    measurements: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    failure_reason: str = Field(default="", max_length=1000)
    notes: str = Field(default="", max_length=2000)


class CampaignActionRequest(BaseModel):
    campaign_id: str = Field(min_length=1, max_length=120)


@router.get("/status")
def status(
    campaign_id: str | None = Query(default=None),
    project_id: int | None = Query(default=None, ge=1),
    ctx: UserContext = Depends(resolve_user_context),
):
    try:
        data = build_campaign_overview(
            runtime_root(), campaign_id=campaign_id, project_id=project_id
        )
    except V020UIError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not ctx.can_access_project(int(data["campaign"]["project_id"])):
        raise HTTPException(status_code=403, detail="当前用户无权访问该项目 V0.2 闭环")
    return {"answer": data["answer"], "data": data}


@router.post("/start-round")
def start_round(body: RoundActionRequest, ctx: UserContext = Depends(resolve_user_context)):
    _check_campaign_access(ctx, body.campaign_id)
    try:
        data = start_round_for_ui(runtime_root(), campaign_id=body.campaign_id, round_id=body.round_id)
    except V020UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}


@router.post("/result")
def submit_result(body: ResultSubmissionRequest, ctx: UserContext = Depends(resolve_user_context)):
    _check_campaign_access(ctx, body.campaign_id)
    payload = {
        "candidate_id": body.candidate_id,
        "status": body.status,
        "test_condition_signature": body.test_condition_signature,
        "measurements": body.measurements,
        "units": body.units,
        "failure_reason": body.failure_reason,
        "notes": body.notes,
    }
    try:
        data = submit_result_for_ui(
            runtime_root(), campaign_id=body.campaign_id, round_id=body.round_id, payload=payload
        )
    except V020UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}


@router.post("/close-round")
def close_round(body: RoundActionRequest, ctx: UserContext = Depends(resolve_user_context)):
    _check_campaign_access(ctx, body.campaign_id)
    try:
        data = close_round_for_ui(runtime_root(), campaign_id=body.campaign_id, round_id=body.round_id)
    except V020UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}


@router.post("/advance")
def advance(body: CampaignActionRequest, ctx: UserContext = Depends(resolve_user_context)):
    _check_campaign_access(ctx, body.campaign_id)
    try:
        data = advance_campaign_for_ui(runtime_root(), campaign_id=body.campaign_id)
    except V020UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}


@router.post("/approve-model")
def approve_model(body: CampaignActionRequest, ctx: UserContext = Depends(resolve_user_context)):
    _check_campaign_access(ctx, body.campaign_id)
    try:
        data = approve_model_for_ui(
            runtime_root(), campaign_id=body.campaign_id, approved_by=ctx.user_id
        )
    except V020UIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}
