from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.deepseek_intent_router import DeepSeekIntentRouter
from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from schemas.user_context import UserContext
from runtime.v030_ui import (
    V030UIError,
    build_autonomy_overview,
)

from runtime.v020_ui import (
    V020UIError,
    build_campaign_overview,
    latest_campaign_id_for_project,
)

from runtime.company_data_ui import (
    CompanyDataUIError,
    build_company_data_overview,
)
from runtime.company_data_inspection import (
    classify_company_data_request,
)
from runtime.company_data_conversation import (
    classify_company_data_turn,
)

from runtime.v014_ui import (
    V014UIError,
    infer_batch_size,
    infer_bo_target_metric,
    looks_like_inverse_design,
    looks_like_next_experiments,
    run_inverse_design_for_ui,
    run_next_experiments_for_ui,
)

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
) -> int | None:
    """Resolve an optional explicit project restriction.

    No project in the user request no longer means an error. The downstream
    RAG skill will search the caller's full authorized scope:
    - current-company all projects when ctx.all_projects=True; or
    - every project listed in ctx.project_ids otherwise.
    """
    raw = tool_args.get("project_id")
    if raw is None or str(raw).strip() == "":
        return None

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




def _company_data_runtime_root():
    import os
    from pathlib import Path

    override = os.getenv("COMPANY_DATA_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


def _classify_company_real_data_turn(
    message: str,
    history,
) -> dict[str, Any]:
    return classify_company_data_turn(
        _company_data_runtime_root(),
        message=message,
        history=history,
    )


def _looks_like_company_real_data(
    message: str,
    history=(),
) -> bool:
    decision = _classify_company_real_data_turn(
        message,
        history,
    )
    return bool(decision["route"])



def _v030_runtime_root():
    import os
    from pathlib import Path

    override = os.getenv("V030_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


def _extract_v030_campaign_id(message: str) -> str | None:
    import re

    match = re.search(
        r"\b(V030_[A-Za-z0-9_.\-]+)\b",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1)
    return re.sub(r"-R\d+$", "", value, flags=re.IGNORECASE)


def _looks_like_v030_autonomy(message: str) -> bool:
    lowered = str(message or "").lower()
    explicit = _extract_v030_campaign_id(message) is not None
    markers = (
        "v0.3",
        "自主实验",
        "自动实验",
        "自主闭环",
        "autonomous",
        "设备状态",
        "scheduler",
        "telemetry",
        "safety stop",
        "安全联锁",
        "crash/resume",
        "crash resume",
        "崩溃恢复",
        "operator override",
        "自动结果回流",
    )
    return any(marker in lowered for marker in markers) or (
        explicit
        and any(
            marker in lowered
            for marker in ("状态", "进度", "round", "设备", "安全")
        )
    )


def _resolve_v030_autonomy_request(
    message: str,
    ctx: UserContext,
):
    import re

    root = _v030_runtime_root()
    campaign_id = _extract_v030_campaign_id(message)
    if campaign_id:
        report = build_autonomy_overview(
            root, campaign_id=campaign_id
        )
        project_id = int(report["campaign"]["project_id"])
        if not ctx.can_access_project(project_id):
            raise HTTPException(
                status_code=403,
                detail="当前用户无权访问该 V0.3 Campaign",
            )
        return report

    match = re.search(
        r"(?:Project|项目)\s*#?\s*(\d+)",
        message,
        re.IGNORECASE,
    )
    if match:
        project_id = int(match.group(1))
        if not ctx.can_access_project(project_id):
            raise HTTPException(
                status_code=403,
                detail="当前用户无权访问该项目 V0.3 autonomous runtime",
            )
    elif len(ctx.project_ids) == 1:
        project_id = int(ctx.project_ids[0])
    else:
        raise HTTPException(
            status_code=400,
            detail="请在问题中写明 Project/项目号或 V030 Campaign ID。",
        )
    return build_autonomy_overview(root, project_id=project_id)


def _v020_runtime_root():
    import os
    from pathlib import Path

    override = os.getenv("V020_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


def _extract_v020_campaign_id(message: str) -> str | None:
    import re
    match = re.search(r"\b(V020_[A-Za-z0-9_.\-]+)\b", message, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1)
    return re.sub(r"-R\d+$", "", value, flags=re.IGNORECASE)


def _looks_like_v020_feedback(message: str) -> bool:
    lowered = message.lower()
    explicit_campaign = _extract_v020_campaign_id(message) is not None
    markers = (
        "闭环状态", "闭环进度", "实验进度", "实验回填", "当前round",
        "当前 round", "数据集版本", "dataset版本", "dataset version",
        "模型晋级", "promotion", "checkpoint", "断点恢复", "campaign状态",
        "campaign 状态", "v0.2 闭环", "v0.2反馈", "v0.2 反馈",
    )
    return any(x in lowered for x in markers) or (
        explicit_campaign and any(x in lowered for x in ("状态", "进度", "round", "campaign", "闭环"))
    )


def _resolve_v020_feedback_request(message: str, ctx: UserContext):
    import re
    root = _v020_runtime_root()
    campaign_id = _extract_v020_campaign_id(message)
    if campaign_id:
        report = build_campaign_overview(root, campaign_id=campaign_id)
        project_id = int(report["campaign"]["project_id"])
        if not ctx.can_access_project(project_id):
            raise HTTPException(status_code=403, detail="当前用户无权访问该 V0.2 Campaign")
        return report

    match = re.search(r"(?:Project|项目)\s*#?\s*(\d+)", message, re.IGNORECASE)
    if match:
        project_id = int(match.group(1))
        if not ctx.can_access_project(project_id):
            raise HTTPException(status_code=403, detail="当前用户无权访问该项目 V0.2 闭环")
    elif len(ctx.project_ids) == 1:
        project_id = int(ctx.project_ids[0])
    else:
        raise HTTPException(status_code=400, detail="请在问题中写明 Project/项目号或 Campaign ID。")

    return build_campaign_overview(root, project_id=project_id)

def _v014_runtime_root():
    import os
    from pathlib import Path

    override = os.getenv("V014_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".runtime"


def _resolve_optimization_project_id(message: str, ctx: UserContext) -> int:
    import re

    match = re.search(r"(?:Project|项目)\s*#?\s*(\d+)", message, re.IGNORECASE)
    if match:
        project_id = int(match.group(1))
        if not ctx.can_access_project(project_id):
            raise HTTPException(status_code=403, detail="当前用户无权执行该项目优化")
        return project_id

    if len(ctx.project_ids) == 1:
        return int(ctx.project_ids[0])

    raise HTTPException(
        status_code=400,
        detail="当前权限包含多个项目，请在问题中写明 Project/项目号。",
    )


@router.post("/chat-ui", response_model=ChatUIResponse)
def chat_ui(
    body: ChatUIRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    # User-provided company real data is a deterministic local runtime route.
    # It never mutates the read-only business MySQL.
    company_decision = _classify_company_real_data_turn(
        body.message,
        body.history,
    )
    if company_decision["route"]:
        company_scope = (
            company_decision.get(
                "conversation_scope"
            )
            or {}
        )
        try:
            report = build_company_data_overview(
                _company_data_runtime_root(),
                message=body.message,
                product_name=company_scope.get(
                    "product_type"
                ),
                classification_override=company_decision,
                conversation_scope=company_scope,
            )
        except CompanyDataUIError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc
        return ChatUIResponse(
            answer=report.get("answer", ""),
            intent="company_real_data_status",
            tool_name="company_real_data_runtime",
            tool_args={
                "dataset_id": report.get("dataset_id"),
                "product_type": (
                    report.get("selected_product") or {}
                ).get("product_type"),
                "scope_source": (
                    report.get("conversation_scope") or {}
                ).get("source"),
                "requested_checks": (
                    report.get("inspection") or {}
                ).get("requested_checks") or [],
            },
            data=report,
            evidence=[],
            warnings=report.get("warnings") or [],
            router="company_data_deterministic",
            reasoning_summary=(
                "确定性读取单位真实数据，并从最近对话继承当前产品作用域；"
                "回答采用 Answer-first，卡片只展示当前问题相关指标。"
                "Reality Check：样品 → 缺失/覆盖 → 重复/异常 → 可建模性；"
                "不合并歧义性能字段。"
                "数据链：样品 → 配方 → 性能 → "
                "数据覆盖 → 建模安全边界；不写业务 MySQL，不绕过 Modeling Gate。"
            ),
        )

    # V0.3 autonomous runtime is deterministic and takes precedence over
    # generic "closed-loop" wording used by V0.2.
    if _looks_like_v030_autonomy(body.message):
        try:
            report = _resolve_v030_autonomy_request(
                body.message, ctx
            )
        except V030UIError as exc:
            raise HTTPException(
                status_code=404, detail=str(exc)
            ) from exc
        return ChatUIResponse(
            answer=report.get("answer", ""),
            intent="v030_autonomy_status",
            tool_name="v030_autonomy_runtime",
            tool_args={
                "campaign_id": (
                    report.get("campaign") or {}
                ).get("campaign_id"),
                "project_id": (
                    report.get("campaign") or {}
                ).get("project_id"),
            },
            data=report,
            evidence=[],
            warnings=[],
            router="v030_deterministic",
            reasoning_summary=(
                "确定性读取 T27-T36：Protocol → Scheduler/Device → "
                "Telemetry → Safety → Automatic Result Capture → "
                "Autonomous Round/Loop → Crash/Resume / Operator Override。"
            ),
        )

    # V0.2 feedback-loop status is a deterministic runtime route.
    # It runs before LLM and before V0.1.4 optimization intent matching.
    if _looks_like_v020_feedback(body.message):
        try:
            report = _resolve_v020_feedback_request(body.message, ctx)
        except V020UIError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ChatUIResponse(
            answer=report.get("answer", ""),
            intent="v020_feedback_loop_status",
            tool_name="v020_campaign_runtime",
            tool_args={
                "campaign_id": report.get("campaign", {}).get("campaign_id"),
                "project_id": report.get("campaign", {}).get("project_id"),
            },
            data=report,
            evidence=[],
            warnings=[],
            router="v020_deterministic",
            reasoning_summary=(
                "确定性读取 T19-T26：Campaign/Round → 实验反馈 → Dataset lineage → "
                "Prediction Evaluation → Model Promotion → Checkpoint → Closed-loop BO。"
            ),
        )

    # V0.1.4 optimization requests are deterministic algorithm routes.
    # They intentionally run before the LLM check so candidate values are
    # never dependent on an LLM being enabled.
    if looks_like_inverse_design(body.message):
        project_id = _resolve_optimization_project_id(body.message, ctx)
        try:
            report = run_inverse_design_for_ui(
                runtime_root=_v014_runtime_root(),
                project_id=project_id,
                message=body.message,
            )
        except V014UIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatUIResponse(
            answer=report.get("answer", ""),
            intent="v014_inverse_design",
            tool_name="inverse_design_engine",
            tool_args={"project_id": project_id},
            data=report,
            evidence=[],
            warnings=[],
            router="v014_deterministic",
            reasoning_summary=(
                "确定性调用 T14-T17：Search Space → Constraints → "
                "models → AD → thresholds → Pareto → diversity。"
            ),
        )

    if looks_like_next_experiments(body.message):
        project_id = _resolve_optimization_project_id(body.message, ctx)
        root = _v014_runtime_root()
        try:
            target_metric = infer_bo_target_metric(body.message, root, project_id)
            batch_size = infer_batch_size(body.message, 5)
            report = run_next_experiments_for_ui(
                runtime_root=root,
                project_id=project_id,
                target_metric=target_metric,
                batch_size=batch_size,
            )
        except V014UIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatUIResponse(
            answer=report.get("answer", ""),
            intent="v014_next_experiments",
            tool_name="gaussian_process_bo",
            tool_args={
                "project_id": project_id,
                "target_metric": target_metric,
                "batch_size": batch_size,
            },
            data=report,
            evidence=[],
            warnings=[],
            router="v014_deterministic",
            reasoning_summary=(
                "确定性调用 T18：历史实验 → GP posterior → adjusted acquisition "
                "→ Kriging Believer batch recommendation。"
            ),
        )

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
                "当前系统已进入 V0.3：支持附件/历史知识分析、真实公司数据查询与 "
                "Reality Check、Dataset/ML/BO、V0.2 实验反馈闭环，以及 V0.3 "
                "Simulator 自主实验编排。当前请求尚未匹配到可执行能力；"
                "如果你在问公司真实数据，可以直接说“真实样本多少”“查看公司真实数据”"
                "或写明具体产品名称。"
            ),
            intent=intent,
            tool_name=None,
            tool_args=tool_args,
            data={"available_version": "V0.3"},
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
