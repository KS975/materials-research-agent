from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.chat import resolve_user_context
from runtime.v013_reports import load_v013_status
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1", tags=["ml-ui"])


def _runtime_root() -> Path:
    override = os.getenv("V013_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime" / "v013"


def _summary_answer(status: dict[str, Any]) -> str:
    gate = status.get("gate") or {}
    decision = gate.get("decision")
    target = status["target_metric"]
    project_id = status["project_id"]

    if not decision:
        return (
            f"Project {project_id} 的“{target}”暂时没有 Modeling Gate 报告。"
            "请先完成 V0.1.3-A/B 数据现实检查与建模准入检查。"
        )
    if decision == "FAIL":
        return (
            f"Project {project_id} 的“{target}”当前建模准入为 FAIL。"
            "系统已禁止训练正式模型，具体原因见下方建模状态卡。"
        )
    if decision == "CONDITIONAL_PASS":
        return (
            f"Project {project_id} 的“{target}”当前为 CONDITIONAL_PASS。"
            "允许受限训练，但不能自动视为正式可发布模型。"
        )
    return (
        f"Project {project_id} 的“{target}”当前 Modeling Gate 为 PASS。"
        "可继续查看模型比较、交叉验证和适用域结果。"
    )


@router.get("/ml-ui/status")
def ml_ui_status(
    project_id: int = Query(..., ge=1),
    target_metric: str = Query(..., min_length=1, max_length=80),
    ctx: UserContext = Depends(resolve_user_context),
):
    if not ctx.can_access_project(project_id):
        raise HTTPException(status_code=403, detail="当前用户无权查看该项目建模状态")

    try:
        status = load_v013_status(_runtime_root(), project_id, target_metric)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500 if not isinstance(exc, ValueError) else 400,
            detail=f"V0.1.3 报告读取失败: {exc}",
        ) from exc

    return {"answer": _summary_answer(status), "data": status}
