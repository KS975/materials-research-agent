from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.deepseek_intent_router import DeepSeekIntentRouter
from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1", tags=["chat-ui"])


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatUIRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    attachment_ids: list[str] = Field(default_factory=list, max_length=8)


class ChatUIResponse(BaseModel):
    answer: str
    intent: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    data: Any = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    router: str = "deepseek"
    reasoning_summary: str = ""


def _looks_like_joint_mysql_knowledge(message: str) -> bool:
    has_history = any(
        marker in message
        for marker in (
            "历史", "以前", "过去", "历史报告", "历史资料", "有没有类似"
        )
    )
    has_database = any(
        marker in message
        for marker in (
            "数据库", "样品", "对比", "比较", "结合数据库", "结合数据"
        )
    )
    # Two explicit numeric sample identifiers are a strong joint-analysis hint.
    import re
    has_two_ids = len(re.findall(r"(?<!\d)\d{3,}(?!\d)", message)) >= 2
    return has_history and (has_database or has_two_ids)


def _looks_like_historical_knowledge(message: str) -> bool:
    markers = (
        "历史有没有",
        "历史上有没有",
        "历史资料",
        "历史报告",
        "以前有没有",
        "过去有没有",
        "有没有类似",
        "类似问题",
        "类似情况",
    )
    return any(marker in message for marker in markers)


def _resolve_historical_project_id(
    tool_args: dict[str, Any],
    ctx: UserContext,
) -> int:
    raw = tool_args.get("project_id")
    if raw is not None:
        try:
            project_id = int(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="历史知识检索的 project_id 必须是整数",
            ) from exc
        if not ctx.can_access_project(project_id):
            raise HTTPException(
                status_code=403,
                detail="当前用户无权检索该项目历史知识",
            )
        return project_id

    if len(ctx.project_ids) == 1:
        return int(ctx.project_ids[0])

    raise HTTPException(
        status_code=400,
        detail=(
            "当前用户同时拥有多个项目权限，请在问题中明确项目号，"
            "例如“项目115历史上有没有类似问题？”"
        ),
    )


def _resolve_joint_args(
    tool_args: dict[str, Any],
    ctx: UserContext,
) -> dict[str, Any]:
    project_id = _resolve_historical_project_id(tool_args, ctx)

    required = ("left_identifier", "right_identifier", "target_metric")
    missing = [
        key for key in required
        if tool_args.get(key) is None or str(tool_args.get(key)).strip() == ""
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail="联合分析缺少参数：" + ", ".join(missing),
        )

    return {
        "project_id": project_id,
        "left_identifier": tool_args["left_identifier"],
        "right_identifier": tool_args["right_identifier"],
        "target_metric": str(tool_args["target_metric"]).strip(),
        "direction_claim": str(
            tool_args.get("direction_claim")
            or tool_args.get("direction")
            or ""
        ).strip(),
    }


@router.post("/chat-ui", response_model=ChatUIResponse)
def chat_ui(
    body: ChatUIRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    if not container.settings.llm_enabled:
        raise HTTPException(503, "请先在后端 .env 启用并配置 DeepSeek：LLM_ENABLED=true")

    attachment_meta = []
    for attachment_id in body.attachment_ids:
        try:
            item = container.chat_attachment_store.get(attachment_id, ctx)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        attachment_meta.append(
            {
                "attachment_id": item.attachment_id,
                "filename": item.filename,
                "parser": item.parser,
                "page_count": item.page_count,
            }
        )

    engine = DeepSeekIntentRouter(container.llm)
    history = [{"role": x.role, "content": x.content} for x in body.history[-12:]]

    try:
        decision = engine.route(body.message, history, attachment_meta)
        router_name = "deepseek"
        summary = decision.reasoning_summary
        intent, tool_name, tool_args = decision.intent, decision.tool_name, decision.tool_args
    except Exception as exc:
        # A joint request must never silently degrade to only one evidence source.
        if _looks_like_joint_mysql_knowledge(body.message):
            raise HTTPException(
                status_code=400,
                detail=(
                    "已识别为 MySQL + 历史资料联合分析请求，但 DeepSeek 参数提取失败："
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        # Historical RAG gets a narrow fail-safe keyword fallback.
        if _looks_like_historical_knowledge(body.message):
            intent, tool_name, tool_args = "search_historical_knowledge", None, {}
            router_name = "knowledge_fallback"
            summary = "DeepSeek 路由失败，按明确的历史知识检索请求处理。"
        else:
            # V0.1.1 database questions retain the already validated rule fallback.
            fallback = container.core.rule_router.route(body.message)
            if fallback is None:
                if body.attachment_ids:
                    # Safe current-attachment fallback: it still cannot access MySQL directly.
                    intent, tool_name, tool_args = "ask_current_attachment", None, {}
                    router_name = "attachment_fallback"
                    summary = "DeepSeek 路由失败，按当前 Chat 附件问题处理。"
                else:
                    raise HTTPException(
                        400,
                        f"DeepSeek 意图识别失败：{type(exc).__name__}: {exc}",
                    ) from exc
            else:
                intent, tool_name, tool_args = (
                    fallback.intent,
                    fallback.tool_name,
                    fallback.tool_args,
                )
                router_name = "rule_fallback"
                summary = "DeepSeek 路由失败，使用 V0.1.1 规则路由兜底。"

    if intent in {"analyze_current_attachment", "ask_current_attachment"}:
        if not body.attachment_ids:
            raise HTTPException(status_code=400, detail="请先上传 PDF 或 DOCX 附件")
        try:
            result = container.current_attachment_skill.answer(
                message=body.message,
                attachment_ids=body.attachment_ids,
                ctx=ctx,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"当前附件分析失败：{type(exc).__name__}: {exc}",
            ) from exc
        return ChatUIResponse(
            answer=result.get("answer", ""),
            intent=intent,
            tool_name=None,
            tool_args={},
            data=result,
            evidence=result.get("evidence", []),
            warnings=result.get("warnings", []),
            router=router_name,
            reasoning_summary=summary,
        )

    if intent == "joint_mysql_knowledge_analysis":
        args = _resolve_joint_args(tool_args, ctx)
        try:
            result = container.joint_mysql_knowledge_skill.answer(
                message=body.message,
                ctx=ctx,
                **args,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"MySQL + 历史资料联合分析失败：{type(exc).__name__}: {exc}",
            ) from exc

        return ChatUIResponse(
            answer=result.get("answer", ""),
            intent=intent,
            tool_name=None,
            tool_args=args,
            data=result,
            evidence=result.get("evidence", []),
            warnings=result.get("warnings", []),
            router=router_name,
            reasoning_summary=summary,
        )

    if intent == "search_historical_knowledge":
        project_id = _resolve_historical_project_id(tool_args, ctx)
        try:
            result = container.historical_knowledge_skill.answer(
                message=body.message,
                project_id=project_id,
                ctx=ctx,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"历史知识 RAG 失败：{type(exc).__name__}: {exc}",
            ) from exc

        return ChatUIResponse(
            answer=result.get("answer", ""),
            intent=intent,
            tool_name=None,
            tool_args={"project_id": project_id},
            data=result,
            evidence=result.get("evidence", []),
            warnings=result.get("warnings", []),
            router=router_name,
            reasoning_summary=summary,
        )

    if intent == "unsupported_future_feature":
        return ChatUIResponse(
            answer=(
                "当前 V0.1.2 已开放当前附件分析、历史 Knowledge Index / Qdrant RAG，"
                "以及 T07 MySQL + 历史资料联合分析。"
            ),
            intent=intent,
            tool_name=None,
            tool_args=tool_args,
            data={"available_version": "V0.1.2-T07"},
            router=router_name,
            reasoning_summary=summary,
        )

    if tool_name is None:
        raise HTTPException(400, "已识别意图，但没有可执行 Tool")

    try:
        result = container.core.execute(intent, tool_name, tool_args, ctx)
        answer = container.core.answer(body.message, intent, result)
    except Exception as exc:
        raise HTTPException(500, f"Agent 执行失败：{type(exc).__name__}: {exc}") from exc

    evidence = result.get("evidence", []) if isinstance(result, dict) else []
    warnings = result.get("warnings", []) if isinstance(result, dict) else []
    return ChatUIResponse(
        answer=answer,
        intent=intent,
        tool_name=tool_name,
        tool_args=tool_args,
        data=result,
        evidence=evidence,
        warnings=warnings,
        router=router_name,
        reasoning_summary=summary,
    )
