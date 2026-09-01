from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import Settings, get_settings
from app.container import ApplicationContainer, get_container
from connectors.permission.factory import create_permission_adapter
from schemas.chat import ChatRequest, ChatResponse
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1")


def resolve_user_context(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> UserContext:
    adapter = create_permission_adapter(settings)
    return adapter.resolve(request)


@router.get("/session-context")
def session_context(ctx: UserContext = Depends(resolve_user_context)):
    """Return the safe, resolved platform context; never return the token."""

    return {
        "user_id": ctx.user_id,
        "company_id": ctx.company_id,
        "organization_id": ctx.organization_id,
        "organization_level": ctx.organization_level,
        "permission_source": ctx.permission_source,
        "project_mode": "company_all_projects" if ctx.all_projects else "project_list",
        "project_ids": list(ctx.project_ids),
    }


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    try:
        state = container.agent.chat(body.message, ctx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent 执行失败：{type(exc).__name__}: {exc}",
        ) from exc

    return ChatResponse(
        answer=state.get("answer", ""),
        intent=state.get("intent", "unknown"),
        tool_name=state.get("tool_name"),
        data=state.get("tool_result"),
        evidence=state.get("evidence", []),
        warnings=state.get("warnings", []),
    )


@router.get("/tools")
def list_tools(container: ApplicationContainer = Depends(get_container)):
    return {"tools": container.registry.list_tools()}
