from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.chat import resolve_user_context
from company_data import CompanyDataRepository, CompanyDataValidationError
from runtime.company_data_ui import (
    CompanyDataUIError,
    build_company_data_overview,
)
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1/company-data", tags=["company-real-data"])


def runtime_root() -> Path:
    override = os.getenv("COMPANY_DATA_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


class ExportRequest(BaseModel):
    product_name: str = Field(min_length=1, max_length=200)
    target_metric: str = Field(min_length=1, max_length=200)


@router.get("/overview")
def overview(
    product_name: str | None = Query(default=None),
    ctx: UserContext = Depends(resolve_user_context),
):
    del ctx  # authentication/context resolution is still required
    try:
        data = build_company_data_overview(
            runtime_root(),
            product_name=product_name,
        )
    except CompanyDataUIError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"answer": data["answer"], "data": data}


@router.post("/export-modeling")
def export_modeling(
    body: ExportRequest,
    ctx: UserContext = Depends(resolve_user_context),
):
    del ctx
    repo = CompanyDataRepository(runtime_root())
    try:
        result = repo.export_modeling_dataset(
            product_name=body.product_name,
            target_metric=body.target_metric,
        )
    except CompanyDataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "answer": (
            "已导出单位真实数据 Reality/Gate 输入。"
            "当前导入本身不满足正式建模安全条件，请继续运行 V0.1.3 Modeling Gate。"
        ),
        "data": result,
    }
