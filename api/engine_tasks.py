from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from runtime.engine_tasks import (
    EngineTaskConflictError,
    EngineTaskNotFoundError,
    EngineTaskPermissionError,
    EngineTaskRequest,
)
from schemas.engine_tasks import (
    EngineTaskActionRequest,
    EngineTaskCreateRequest,
    EngineTaskStatusResponse,
)
from schemas.user_context import UserContext


router = APIRouter(prefix="/api/v1/engine-tasks", tags=["engine-tasks"])


@router.post("", response_model=EngineTaskStatusResponse, status_code=202)
def create_engine_task(
    body: EngineTaskCreateRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
) -> EngineTaskStatusResponse:
    request = EngineTaskRequest(
        intent=body.intent,
        tool_name=body.tool_name,
        tool_args=dict(body.tool_args),
        conversation_id=str(body.conversation_id or ""),
    )
    try:
        status = container.engine_task_manager.submit(ctx=ctx, request=request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"任务创建失败：{type(exc).__name__}: {exc}",
        ) from exc
    return EngineTaskStatusResponse(**status)


@router.get("/{task_id}", response_model=EngineTaskStatusResponse)
def get_engine_task(
    task_id: str,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
) -> EngineTaskStatusResponse:
    try:
        status = container.engine_task_manager.status(task_id, ctx)
    except EngineTaskPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except EngineTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EngineTaskConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EngineTaskStatusResponse(**status)


@router.post("/{task_id}/cancel", response_model=EngineTaskStatusResponse)
def cancel_engine_task(
    task_id: str,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
) -> EngineTaskStatusResponse:
    try:
        status = container.engine_task_manager.cancel(task_id, ctx)
    except EngineTaskPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except EngineTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EngineTaskConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EngineTaskStatusResponse(**status)


@router.post("/{task_id}/approve", response_model=EngineTaskStatusResponse)
def approve_engine_task(
    task_id: str,
    body: EngineTaskActionRequest | None = None,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
) -> EngineTaskStatusResponse:
    try:
        status = container.engine_task_manager.approve(
            task_id,
            ctx,
            reason=str(body.reason if body is not None else ""),
        )
    except EngineTaskPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except EngineTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EngineTaskConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EngineTaskStatusResponse(**status)


@router.post("/{task_id}/resume", response_model=EngineTaskStatusResponse)
def resume_engine_task(
    task_id: str,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
) -> EngineTaskStatusResponse:
    try:
        status = container.engine_task_manager.resume(task_id, ctx)
    except EngineTaskPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except EngineTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EngineTaskConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EngineTaskStatusResponse(**status)
