from __future__ import annotations

from threading import RLock
from time import monotonic
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.field_catalog import normalize_field_name
from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from data.json_utils import decode_json_mapping
from schemas.user_context import UserContext


router = APIRouter(prefix="/api/v1/dashboard", tags=["database-navigator"])

_FIELD_CACHE_TTL_SECONDS = 60.0
_field_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_field_cache_lock = RLock()
_SECTION_LABELS = {
    "sample": "样品基础",
    "formula": "配方",
    "process": "工艺",
    "performance": "性能",
    "conditions": "测试条件",
}


def _scope_payload(ctx: UserContext) -> dict[str, Any]:
    return {
        "company_id": ctx.company_id,
        "project_mode": "company_all_projects" if ctx.all_projects else "authorized_projects",
        "project_ids": [] if ctx.all_projects else list(ctx.project_ids),
        "permission_source": ctx.permission_source,
    }


def _field_cache_key(ctx: UserContext) -> tuple[Any, ...]:
    return (ctx.company_id, ctx.all_projects, *sorted(ctx.project_ids))


def _get_field_catalog(
    container: ApplicationContainer,
    ctx: UserContext,
) -> dict[str, Any]:
    key = _field_cache_key(ctx)
    now = monotonic()
    with _field_cache_lock:
        cached = _field_cache.get(key)
        if cached and now - cached[0] <= _FIELD_CACHE_TTL_SECONDS:
            return cached[1]
    catalog = container.tools.get_material_field_catalog(ctx)
    with _field_cache_lock:
        _field_cache[key] = (monotonic(), catalog)
    return catalog


def _sample_card(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": row.get("name"),
        "project_id": row.get("project_id"),
        "project_name": row.get("project_name"),
        "sample_type": row.get("sample_type"),
        "create_time": row.get("create_time"),
        "update_time": row.get("update_time"),
        "field_counts": {
            "formula": len(decode_json_mapping(row.get("recipes"))),
            "process": len(decode_json_mapping(row.get("craft_param"))),
            "performance": len(decode_json_mapping(row.get("performances"))),
            "service_performance": len(
                decode_json_mapping(row.get("service_performances"))
            ),
            "conditions": len(decode_json_mapping(row.get("conditions"))),
        },
    }


@router.get("/summary")
def dashboard_summary(
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    return {
        "status": "ok",
        "scope": _scope_payload(ctx),
        **container.dashboard.summary(ctx),
    }


@router.get("/projects")
def dashboard_projects(
    q: str = Query(default="", max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100000),
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    return {
        "status": "ok",
        "scope": _scope_payload(ctx),
        **container.dashboard.list_projects(
            ctx,
            query=q,
            limit=limit,
            offset=offset,
        ),
    }


@router.get("/samples")
def dashboard_samples(
    q: str = Query(default="", max_length=200),
    project_id: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
    offset: int = Query(default=0, ge=0, le=100000),
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    if project_id is not None and not ctx.can_access_project(project_id):
        raise HTTPException(status_code=403, detail="当前用户无权浏览该项目")
    result = container.dashboard.list_samples(
        ctx,
        query=q,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    return {
        "status": "ok",
        "scope": _scope_payload(ctx),
        **result,
        "samples": [_sample_card(row) for row in result["samples"]],
    }


@router.get("/samples/{sample_id}")
def dashboard_sample_detail(
    sample_id: int,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    result = container.tools.get_sample_context(sample_id, ctx)
    if result.get("status") != "ok":
        raise HTTPException(status_code=404, detail="当前公司及项目权限范围内未找到该样品")
    return {
        "status": "ok",
        "scope": _scope_payload(ctx),
        "data": result,
    }


@router.get("/fields")
def dashboard_fields(
    section: Literal[
        "all", "sample", "formula", "process", "performance", "conditions"
    ] = Query(default="all"),
    q: str = Query(default="", max_length=200),
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    catalog = _get_field_catalog(container, ctx)
    if catalog.get("status") != "ok":
        raise HTTPException(status_code=500, detail="字段目录读取失败")
    query_key = normalize_field_name(q)
    sections = {}
    for key, items in (catalog.get("sections") or {}).items():
        if section != "all" and key != section:
            continue
        filtered = [
            item
            for item in items or []
            if not query_key or query_key in normalize_field_name(item.get("name"))
        ]
        sections[key] = filtered
    return {
        "status": "ok",
        "scope": _scope_payload(ctx),
        "section_labels": _SECTION_LABELS,
        "sections": sections,
        "field_counts": {key: len(items) for key, items in sections.items()},
        "source_sample_count": catalog.get("source_sample_count", 0),
        "scan_complete": catalog.get("scan_complete", False),
        "unresolved_field_count": catalog.get("unresolved_field_count", 0),
        "value_disclosure": False,
    }
