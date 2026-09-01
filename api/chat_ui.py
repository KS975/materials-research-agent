from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent.deepseek_intent_router import DeepSeekIntentRouter
from agent.multi_condition import looks_like_multi_condition_request
from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from orchestration.chat_ui_graph import (
    build_chat_ui_graph,
    invoke_chat_ui_graph,
)
from schemas.chat_ui import ChatUIRequest, ChatUIResponse, HistoryMessage
from schemas.user_context import UserContext
from runtime.progress import emit_progress, progress_context
from runtime.chat_ui_workflow import (
    ChatUIWorkflowCheckpointError,
    ChatUIWorkflowConflictError,
    ChatUIWorkflowNotFoundError,
    ChatUIWorkflowPermissionError,
)
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
    company_data_has_priority,
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


_ROUND2A_DATABASE_INTENTS = {
    "sample_full_profile",
    "formula_difference",
    "process_difference",
    "comparability_check",
    "performance_rank",
    "experiment_series_analysis",
    "data_quality_check",
    "similar_samples",
}

_EXPLICIT_COMPANY_DATA_MARKERS = (
    "单位真实数据",
    "公司真实数据",
    "真实数据",
    "真实样本",
    "真实样品",
    "单位数据",
    "公司数据",
    "海科数据",
    "海科总库",
    "总库",
    "导入的真实数据",
)

_INTENT_PROGRESS_NAMES = {
    "sample_full_profile": "读取样品完整档案",
    "formula_difference": "比较样品配方差异",
    "process_difference": "比较样品工艺差异",
    "comparability_check": "检查样品测试可比性",
    "performance_rank": "按性能筛选与排序",
    "experiment_series_analysis": "分析实验系列",
    "data_quality_check": "检查数据质量",
    "find_samples_multi_condition": "按多个条件筛选样品",
    "similar_samples": "计算相似样品",
    "joint_mysql_knowledge_analysis": "样品对比 + 历史资料联合分析",
    "sample_historical_similarity": "当前样品 + 历史案例联合分析",
    "search_historical_knowledge": "检索历史知识库",
    "historical_similar_case": "检索相似历史案例",
    "database_explorer": "DeepSeek 授权数据库探索",
    "general_conversation": "通用问答",
    "v014_inverse_design": "多目标逆向设计",
    "v014_next_experiments": "下一轮实验推荐",
}


def _intent_progress_details(intent: str, tool_args: dict[str, Any]) -> list[dict[str, str]]:
    labels = {
        "identifier": "样品",
        "left_identifier": "左侧样品",
        "right_identifier": "右侧样品",
        "target_metric": "关注指标",
        "keyword": "检索关键词",
        "similarity_scope": "相似范围",
        "top_n": "返回数量",
        "project_id": "Project",
        "history_query": "历史检索主题",
    }
    output = [{
        "label": "任务类型",
        "value": _INTENT_PROGRESS_NAMES.get(intent, intent),
    }]
    for key, label in labels.items():
        value = tool_args.get(key)
        if value in (None, "", []):
            continue
        output.append({"label": label, "value": str(value)[:500]})
    return output


def initial_stream_progress_event() -> dict[str, Any]:
    """Return the first user-safe SSE event before domain work begins."""
    return {
        "schema_version": "1.1",
        "source": "backend",
        "stage": "stream_connected",
        "status": "completed",
        "title": "实时分析通道已建立",
        "message": "后端已接受请求，后续执行阶段将通过当前通道持续返回。",
        "elapsed_ms": 0,
    }


def _route_round2a2_database_intent(
    message: str,
    rule_router,
    history: list[dict[str, str]] | None = None,
):
    """Reserve explicit Round 2A R&D questions for business MySQL.

    The imported company-data runtime remains available only when the current
    turn explicitly names that source. Pair follow-ups are resolved from user
    history before the low-priority imported-data overview classifier.
    """
    text = str(message or "").strip()
    if any(marker in text for marker in _EXPLICIT_COMPANY_DATA_MARKERS):
        return None
    decision = rule_router.route(text)
    if decision is not None and decision.intent in _ROUND2A_DATABASE_INTENTS:
        return decision
    return DeepSeekIntentRouter.deterministic_material_followup_decision(
        text,
        history or [],
    )


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


def _looks_like_unmatched_database_question(message: str) -> bool:
    """Conservative fail-safe used only when the JSON intent router crashes."""
    text = str(message or "").strip()
    strong_scope = any(marker in text for marker in (
        "数据库", "库里", "数据表", "表中", "表里", "样品记录", "项目记录",
    ))
    material_scope_count = sum(
        marker in text
        for marker in (
            "样品", "样本", "项目", "配方", "工艺", "性能", "实验", "批次",
        )
    )
    query_request = any(marker in text for marker in (
        "查", "找", "哪些", "哪个", "多少", "有没有", "统计", "分布",
        "趋势", "关系", "最近", "最常见", "汇总", "分析",
    ))
    return query_request and (strong_scope or material_scope_count >= 2)


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


def _resolve_sample_history_args(
    tool_args: dict[str, Any],
    ctx: UserContext,
) -> dict[str, Any]:
    project_id = _resolve_historical_project_id(tool_args, ctx)
    identifier = tool_args.get("identifier")
    if identifier is None or str(identifier).strip() == "":
        raise HTTPException(
            status_code=400,
            detail="样品历史相似分析缺少 identifier",
        )
    return {
        "project_id": project_id,
        "identifier": identifier,
        "target_metric": str(tool_args.get("target_metric") or "").strip(),
        "history_query": str(tool_args.get("history_query") or "").strip(),
    }


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


def _classify_chat_ui_primary_family(state: dict[str, Any]) -> dict[str, Any]:
    """Select the first V2 graph family without executing DB/LLM work.

    The order mirrors the protected production dispatcher exactly. All checks
    here are deterministic and side-effect free; the dispatcher remains the
    final authority and repeats the same checks before executing a capability.
    """

    body: ChatUIRequest = state["body"]
    ctx: UserContext = state["user_context"]
    container: ApplicationContainer = state["container"]

    if body.attachment_reference_mode:
        return {
            "primary_family": "direct_attachment",
            "deterministic_kind": "deepseek_attachment_passthrough",
        }

    history = [{"role": x.role, "content": x.content} for x in body.history[-12:]]
    round2a2_decision = _route_round2a2_database_intent(
        body.message,
        container.core.rule_router,
        history,
    )
    if round2a2_decision is not None:
        return {
            "primary_family": "deterministic",
            "deterministic_kind": round2a2_decision.intent,
        }

    company_decision = _classify_company_real_data_turn(body.message, body.history)
    if company_data_has_priority(company_decision):
        return {
            "primary_family": "deterministic",
            "deterministic_kind": "company_real_data_status",
        }
    if _looks_like_v030_autonomy(body.message):
        return {
            "primary_family": "deterministic",
            "deterministic_kind": "v030_autonomy_status",
        }
    if _looks_like_v020_feedback(body.message):
        return {
            "primary_family": "deterministic",
            "deterministic_kind": "v020_feedback_loop_status",
        }
    if looks_like_inverse_design(body.message):
        return {
            "primary_family": "deterministic",
            "deterministic_kind": "v014_inverse_design",
        }
    if looks_like_next_experiments(body.message):
        return {
            "primary_family": "deterministic",
            "deterministic_kind": "v014_next_experiments",
        }
    return {
        "primary_family": "semantic",
        "deterministic_kind": "",
    }


def _classify_chat_ui_semantic_family(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the evidence family from the actual semantic response.

    This runs after the protected dispatcher, so it never invokes DeepSeek a
    second time and cannot disagree with the response returned to the user.
    """

    response = state.get("response")
    if isinstance(response, dict):
        response = ChatUIResponse.model_validate(response)
    if not isinstance(response, ChatUIResponse):
        raise RuntimeError("LangGraph V2 无法识别语义响应")

    intent = str(response.intent or "").strip()
    router_name = str(response.router or "").strip()
    return {"semantic_family": _semantic_family_for_intent(intent, router_name)}


def _semantic_family_for_intent(intent: str, router_name: str = "") -> str:
    if intent == "database_explorer" or router_name == "deepseek_database_explorer":
        return "database_explorer"
    if intent in {
        "sample_historical_similarity",
        "joint_mysql_knowledge_analysis",
        "search_historical_knowledge",
        "historical_similar_case",
    }:
        return "rag"
    if intent in {
        "analyze_current_attachment",
        "ask_current_attachment",
    }:
        return "current_attachment"
    if intent in {
        "general_conversation",
        "unsupported_future_feature",
        "clarification_required",
    }:
        return "general_conversation"
    return "material_tool"


def _plan_chat_ui_semantic(state: dict[str, Any]) -> dict[str, Any]:
    """Run the V4 DeepSeek semantic planner exactly once.

    This node may inspect authorized field names, but it does not execute the
    selected material/RAG/database capability. Its JSON-safe result becomes
    native LangGraph state and selects one of five execution nodes.
    """

    body: ChatUIRequest = state["body"]
    ctx: UserContext = state["user_context"]
    container: ApplicationContainer = state["container"]

    if not container.settings.llm_enabled:
        raise HTTPException(503, "请先在后端 .env 启用并配置 DeepSeek：LLM_ENABLED=true")

    history = [{"role": x.role, "content": x.content} for x in body.history[-12:]]
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

    field_catalog = None
    if looks_like_multi_condition_request(body.message):
        emit_progress(
            "schema_loading",
            "running",
            "读取授权字段目录",
            "正在读取当前权限范围内实际存在的字段名称和单位。",
        )
        try:
            field_catalog = (
                container.core.material_intelligence_skill.get_field_catalog(ctx)
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Round 2B-1.1 授权字段目录读取失败："
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        if not isinstance(field_catalog, dict) or field_catalog.get("status") != "ok":
            raise HTTPException(
                status_code=500,
                detail="Round 2B-1.1 授权字段目录不完整，已停止 Schema-Aware 路由。",
            )
        emit_progress(
            "schema_loading",
            "completed",
            "字段目录已加载",
            f"已加载 {field_catalog.get('total_field_count', 0)} 个授权字段定义。",
            field_count=field_catalog.get("total_field_count", 0),
        )

    engine = DeepSeekIntentRouter(container.llm)
    database_explorer_skill = getattr(
        container,
        "database_explorer_skill",
        None,
    )
    database_explorer_enabled = bool(
        database_explorer_skill is not None
        and getattr(database_explorer_skill, "enabled", False)
    )
    routing_meta: dict[str, Any] = {}
    needs_clarification = False
    clarification_question = ""

    try:
        emit_progress(
            "intent_routing",
            "running",
            "DeepSeek 语义路由",
            "正在结合当前问题和对话上下文提取业务意图。",
        )
        decision = engine.route(
            body.message,
            history,
            attachment_meta,
            field_catalog=field_catalog,
            database_explorer_enabled=database_explorer_enabled,
            database_explorer_mode=str(
                getattr(database_explorer_skill, "mode", "off")
            ),
        )
        router_name = "deepseek"
        summary = decision.reasoning_summary
        intent, tool_name, tool_args = decision.intent, decision.tool_name, decision.tool_args
        routing_meta = decision.to_routing_meta()
        needs_clarification = decision.needs_clarification
        clarification_question = decision.clarification_question
        emit_progress(
            "intent_routing",
            "completed",
            "问题理解完成",
            (
                f"已识别为“{_INTENT_PROGRESS_NAMES.get(intent, intent)}”。"
                + (f" {summary}" if summary else "")
            ),
            intent=intent,
            detail_items=_intent_progress_details(intent, tool_args),
            plan_summary=summary,
        )
    except Exception as exc:
        if _looks_like_joint_mysql_knowledge(body.message):
            raise HTTPException(
                status_code=400,
                detail=(
                    "已识别为 MySQL + 历史资料联合分析请求，但 DeepSeek 参数提取失败："
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        if _looks_like_historical_knowledge(body.message):
            intent, tool_name, tool_args = "search_historical_knowledge", None, {}
            router_name = "knowledge_fallback"
            summary = "DeepSeek 路由失败，按明确的历史知识检索请求处理。"
        else:
            fallback = container.core.rule_router.route(body.message)
            if fallback is None:
                if body.attachment_ids:
                    intent, tool_name, tool_args = "ask_current_attachment", None, {}
                    router_name = "attachment_fallback"
                    summary = "DeepSeek 路由失败，按当前 Chat 附件问题处理。"
                elif (
                    database_explorer_enabled
                    and _looks_like_unmatched_database_question(body.message)
                ):
                    intent, tool_name, tool_args = "database_explorer", None, {}
                    router_name = "database_explorer_fail_safe"
                    summary = (
                        "DeepSeek JSON 意图路由失败，但当前问题明确要求业务数据库事实；"
                        "转入授权只读 Database Explorer。"
                    )
                else:
                    intent, tool_name, tool_args = "general_conversation", None, {}
                    router_name = "deepseek_answer_fallback"
                    summary = (
                        "DeepSeek JSON 意图路由失败，转入无 Tool 的受约束通用回答；"
                        "本轮不得声称使用数据库、附件或历史知识证据。"
                    )
            else:
                intent, tool_name, tool_args = (
                    fallback.intent,
                    fallback.tool_name,
                    fallback.tool_args,
                )
                router_name = "rule_fallback"
                summary = "DeepSeek 路由失败，使用 V0.1.1 规则路由兜底。"

    tool_args = dict(tool_args or {})
    if not routing_meta:
        is_database_explorer = intent == "database_explorer"
        routing_meta = {
            "version": "DBE-0.1" if is_database_explorer else "fallback",
            "domain": (
                "retrieve"
                if is_database_explorer or tool_name is not None
                else "conversation"
            ),
            "primary_intent": intent,
            "secondary_intents": [],
            "entities": {},
            "scope": {"company": "current", "projects": "authorized"},
            "constraints": ({
                "read_only": True,
                "authorized_virtual_sources_only": True,
                "bounded_sql_retry": True,
            } if is_database_explorer else {}),
            "context_reference": {"action": "fallback"},
            "tool_plan": ([{
                "kind": "tool",
                "name": tool_name,
                "args": dict(tool_args),
                "purpose": "fallback execution",
            }] if tool_name else ([{
                "kind": "skill",
                "name": "database_explorer",
                "args": {},
                "purpose": "授权只读数据库探索兜底",
            }] if is_database_explorer else [])),
            "needs_clarification": False,
            "clarification_question": "",
        }

    planned_intent = "clarification_required" if needs_clarification else intent
    semantic_family = _semantic_family_for_intent(planned_intent, router_name)
    family_labels = {
        "database_explorer": "授权数据库探索",
        "rag": "历史知识与联合分析",
        "current_attachment": "当前附件问答",
        "general_conversation": "通用回答或澄清",
        "material_tool": "确定性材料工具",
    }
    emit_progress(
        "semantic_plan",
        "completed",
        "执行路线已确定",
        f"LangGraph 将进入“{family_labels[semantic_family]}”执行节点。",
        intent=intent,
        semantic_family=semantic_family,
        detail_items=[
            {"label": "语义执行分支", "value": family_labels[semantic_family]},
            {"label": "业务意图", "value": str(intent)},
        ],
    )
    return {
        "semantic_family": semantic_family,
        "history": history,
        "attachment_meta": attachment_meta,
        "database_explorer_enabled": database_explorer_enabled,
        "intent": intent,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "router_name": router_name,
        "reasoning_summary": summary,
        "routing_meta": dict(routing_meta),
        "needs_clarification": bool(needs_clarification),
        "clarification_question": str(clarification_question or ""),
    }


def _semantic_state(state: dict[str, Any]):
    return (
        state["body"],
        state["user_context"],
        state["container"],
        str(state.get("intent") or ""),
        state.get("tool_name"),
        dict(state.get("tool_args") or {}),
        str(state.get("router_name") or "deepseek"),
        str(state.get("reasoning_summary") or ""),
        dict(state.get("routing_meta") or {}),
    )


def _execute_semantic_current_attachment(state: dict[str, Any]) -> ChatUIResponse:
    body, ctx, container, intent, _, _, router_name, summary, routing_meta = (
        _semantic_state(state)
    )
    if intent not in {"analyze_current_attachment", "ask_current_attachment"}:
        raise HTTPException(400, f"附件执行节点收到不支持的意图：{intent or '-'}")
    if not body.attachment_ids:
        raise HTTPException(status_code=400, detail="请先上传 PDF、DOCX 或 XLSX 附件")
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
        routing=routing_meta,
    )


def _execute_semantic_rag(state: dict[str, Any]) -> ChatUIResponse:
    body, ctx, container, intent, _, tool_args, router_name, summary, routing_meta = (
        _semantic_state(state)
    )
    if intent == "sample_historical_similarity":
        args = _resolve_sample_history_args(tool_args, ctx)
        try:
            result = container.sample_historical_similarity_skill.answer(
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
                detail=f"样品 + 历史资料相似分析失败：{type(exc).__name__}: {exc}",
            ) from exc
        response_args = args
    elif intent == "joint_mysql_knowledge_analysis":
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
        response_args = args
    elif intent in {"search_historical_knowledge", "historical_similar_case"}:
        project_id = _resolve_historical_project_id(tool_args, ctx)
        history_query = str(tool_args.get("history_query") or "").strip()
        try:
            result = container.historical_knowledge_skill.answer(
                message=(history_query or body.message),
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
        response_args = {
            "project_id": project_id,
            "history_query": history_query,
        }
    else:
        raise HTTPException(400, f"RAG 执行节点收到不支持的意图：{intent or '-'}")

    return ChatUIResponse(
        answer=result.get("answer", ""),
        intent=intent,
        tool_name=None,
        tool_args=response_args,
        data=result,
        evidence=result.get("evidence", []),
        warnings=result.get("warnings", []),
        router=router_name,
        reasoning_summary=summary,
        routing=routing_meta,
    )


def _execute_semantic_database_explorer(state: dict[str, Any]) -> ChatUIResponse:
    body, ctx, container, intent, _, _, _, summary, routing_meta = _semantic_state(state)
    database_explorer_skill = getattr(container, "database_explorer_skill", None)
    enabled = bool(
        state.get("database_explorer_enabled")
        and database_explorer_skill is not None
        and getattr(database_explorer_skill, "enabled", False)
    )
    if intent != "database_explorer":
        raise HTTPException(400, f"数据库探索节点收到不支持的意图：{intent or '-'}")
    if not enabled:
        raise HTTPException(status_code=503, detail="Database Explorer 当前未启用。")
    try:
        result = database_explorer_skill.answer(
            message=body.message,
            history=list(state.get("history") or []),
            ctx=ctx,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database Explorer 执行失败：{type(exc).__name__}: {exc}",
        ) from exc
    return ChatUIResponse(
        answer=result.get("answer", ""),
        intent=intent,
        tool_name="database_explorer",
        tool_args={},
        data=result,
        evidence=result.get("evidence", []),
        warnings=result.get("warnings", []),
        router="deepseek_database_explorer",
        reasoning_summary=(
            summary or "未命中高精度意图，转入授权只读 Database Explorer。"
        ),
        routing=routing_meta,
    )


def _execute_semantic_general_conversation(state: dict[str, Any]) -> ChatUIResponse:
    body, _, container, intent, _, tool_args, router_name, summary, routing_meta = (
        _semantic_state(state)
    )
    if state.get("needs_clarification"):
        return ChatUIResponse(
            answer=(
                str(state.get("clarification_question") or "").strip()
                or "当前信息不足，请补充样品、指标或分析范围。"
            ),
            intent="clarification_required",
            tool_name=None,
            tool_args=tool_args,
            data={
                "requested_intent": intent,
                "needs_clarification": True,
            },
            evidence=[],
            warnings=[],
            router=router_name,
            reasoning_summary=summary,
            routing=routing_meta,
        )
    if intent not in {"general_conversation", "unsupported_future_feature"}:
        raise HTTPException(400, f"通用回答节点收到不支持的意图：{intent or '-'}")
    try:
        result = container.general_conversation_skill.answer(
            message=body.message,
            history=list(state.get("history") or []),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek 通用回答失败：{type(exc).__name__}: {exc}",
        ) from exc
    return ChatUIResponse(
        answer=result.get("answer", ""),
        intent=intent,
        tool_name=None,
        tool_args={},
        data=result,
        evidence=[],
        warnings=result.get("warnings", []),
        router=(
            "deepseek_general_answer"
            if router_name == "deepseek"
            else router_name
        ),
        reasoning_summary=summary,
        routing=routing_meta,
    )


def _execute_semantic_material_tool(state: dict[str, Any]) -> ChatUIResponse:
    body, ctx, container, intent, tool_name, tool_args, router_name, summary, routing_meta = (
        _semantic_state(state)
    )
    if tool_name is None:
        raise HTTPException(400, "已识别意图，但没有可执行 Tool")
    try:
        emit_progress(
            "tool_execution",
            "running",
            "执行材料工具",
            f"正在执行只读工具：{tool_name}。",
            intent=intent,
        )
        result = container.core.execute(intent, tool_name, tool_args, ctx)
        emit_progress(
            "tool_execution",
            "completed",
            "材料工具执行完成",
            "结构化数据库证据已返回。",
            intent=intent,
        )
        emit_progress(
            "answer_generation",
            "running",
            "组织回答",
            "正在将结构化事实转换为中文回答。",
        )
        answer = container.core.answer(body.message, intent, result)
        emit_progress(
            "answer_generation",
            "completed",
            "回答已生成",
            "最终回答已依据本轮工具证据生成。",
        )
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
        routing=routing_meta,
    )


def _execute_chat_ui_legacy(
    body: ChatUIRequest,
    ctx: UserContext,
    container: ApplicationContainer,
):
    emit_progress(
        "intent_routing",
        "running",
        "识别问题",
        "正在判断应使用确定性材料能力、数据库探索或其它证据源。",
    )

    # Explicit UI mode: act as a pure DeepSeek attachment Q&A relay. The
    # parsed attachment body and the user's original question go straight to
    # the LLM before any business intent, DB or optimization route runs. With
    # the flag off, all historical routing behavior remains unchanged.
    if body.attachment_reference_mode:
        if not body.attachment_ids:
            raise HTTPException(
                status_code=400,
                detail="DeepSeek 附件问答已开启，请先上传 PDF、DOCX 或 XLSX 附件",
            )
        if not container.settings.llm_enabled:
            raise HTTPException(
                status_code=503,
                detail="DeepSeek 附件问答需要先启用模型：LLM_ENABLED=true",
            )

        emit_progress(
            "deepseek_attachment_passthrough",
            "running",
            "向 DeepSeek 提交附件",
            "正在读取当前附件正文，并与用户问题一并发送给 DeepSeek。",
            attachment_count=len(body.attachment_ids),
            detail_items=[{
                "label": "发送内容",
                "value": f"当前 Chat 的 {len(body.attachment_ids)} 个附件",
            }],
        )
        try:
            result = container.current_attachment_skill.answer(
                message=body.message,
                attachment_ids=body.attachment_ids,
                ctx=ctx,
                reference_mode=True,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"DeepSeek 附件问答失败：{type(exc).__name__}: {exc}",
            ) from exc

        raw_answer = str(result.get("answer") or "").strip()
        result = dict(result)
        result["answer"] = raw_answer

        emit_progress(
            "deepseek_attachment_passthrough",
            "completed",
            "DeepSeek 答案已返回",
            "已将 DeepSeek 对附件和用户问题的回答直接返回；本轮未进入业务路由。",
            attachment_count=len(body.attachment_ids),
            evidence_count=len(result.get("evidence") or []),
        )
        return ChatUIResponse(
            answer=result.get("answer", ""),
            intent="deepseek_attachment_answer",
            tool_name=None,
            tool_args={"attachment_count": len(body.attachment_ids)},
            data=result,
            evidence=result.get("evidence", []),
            warnings=result.get("warnings", []),
            router="deepseek_attachment_passthrough",
            reasoning_summary=(
                "纯附件转接模式：后端将当前附件解析正文和用户原问题直接提交给 DeepSeek，"
                "答案不经过业务意图、数据库或 T17/T18 处理。"
            ),
            routing={
                "version": "DEEPSEEK-ATTACHMENT-PASSTHROUGH-0.2",
                "domain": "direct_attachment_qa",
                "primary_intent": "deepseek_attachment_answer",
                "secondary_intents": [],
                "entities": {"attachment_count": len(body.attachment_ids)},
                "scope": {
                    "company": "current",
                    "projects": "attachment_owner_scope",
                    "data_source": "current_chat_attachments",
                },
                "constraints": {
                    "attachment_only_evidence": True,
                    "business_intent_bypassed": True,
                    "database_not_queried": True,
                    "optimization_engine_not_run": True,
                    "answer_postprocessed": False,
                },
                "context_reference": {"action": "explicit_deepseek_attachment_mode"},
                "tool_plan": [{
                    "kind": "llm",
                    "name": "deepseek_attachment_passthrough",
                    "args": {"reference_mode": True},
                    "purpose": "将附件正文与用户问题直接提交给 DeepSeek",
                }],
                "needs_clarification": False,
                "clarification_question": "",
            },
        )

    # Materials Intent Round 2A is a business-MySQL capability. Explicit and
    # context-resolved material requests must execute before the low-priority
    # imported company-data overview router.
    history = [{"role": x.role, "content": x.content} for x in body.history[-12:]]
    round2a2_decision = _route_round2a2_database_intent(
        body.message,
        container.core.rule_router,
        history,
    )
    if round2a2_decision is not None:
        emit_progress(
            "intent_routing",
            "completed",
            "意图已识别",
            "已匹配高精度能力："
            f"{_INTENT_PROGRESS_NAMES.get(round2a2_decision.intent, round2a2_decision.intent)}。",
            intent=round2a2_decision.intent,
            detail_items=_intent_progress_details(
                round2a2_decision.intent,
                dict(round2a2_decision.tool_args),
            ),
        )
        try:
            emit_progress(
                "deterministic_analysis",
                "running",
                "执行确定性分析",
                "正在读取业务 MySQL 证据并执行后端计算。",
            )
            result = container.core.execute(
                round2a2_decision.intent,
                round2a2_decision.tool_name,
                dict(round2a2_decision.tool_args),
                ctx,
            )
            answer = container.core.answer(
                body.message,
                round2a2_decision.intent,
                result,
            )
            emit_progress(
                "answer_generation",
                "completed",
                "组织回答",
                "数据库事实和确定性计算结果已整理完成。",
            )
        except Exception as exc:
            raise HTTPException(
                500,
                f"Round 2A 数据库分析失败：{type(exc).__name__}: {exc}",
            ) from exc

        evidence = result.get("evidence", []) if isinstance(result, dict) else []
        warnings = result.get("warnings", []) if isinstance(result, dict) else []
        is_round2b_similarity = round2a2_decision.intent == "similar_samples"
        routing = {
            "version": "2B-2.1" if is_round2b_similarity else "2A-2.6",
            "domain": (
                "validate"
                if round2a2_decision.intent == "data_quality_check"
                else "analyze"
            ),
            "primary_intent": round2a2_decision.intent,
            "secondary_intents": [],
            "entities": {
                "samples": [round2a2_decision.tool_args["identifier"]]
                if round2a2_decision.tool_args.get("identifier") is not None
                else [],
                "metrics": [round2a2_decision.tool_args["target_metric"]]
                if round2a2_decision.tool_args.get("target_metric")
                else [],
                "series_keyword": round2a2_decision.tool_args.get("keyword") or None,
            },
            "scope": {
                "company": "current",
                "projects": "authorized",
                "data_source": "business_mysql",
            },
            "constraints": {"read_only": True, "deterministic_calculation": True},
            "context_reference": {"action": "new_request"},
            "tool_plan": [{
                "kind": "tool",
                "name": round2a2_decision.tool_name,
                "args": dict(round2a2_decision.tool_args),
                "purpose": "读取授权范围内的 MySQL 样品并执行确定性材料分析",
            }],
            "needs_clarification": False,
            "clarification_question": "",
        }
        return ChatUIResponse(
            answer=answer,
            intent=round2a2_decision.intent,
            tool_name=round2a2_decision.tool_name,
            tool_args=dict(round2a2_decision.tool_args),
            data=result,
            evidence=evidence,
            warnings=warnings,
            router=(
                "materials_round2b_similarity"
                if is_round2b_similarity
                else "materials_round2a_mysql"
            ),
            reasoning_summary=(
                (
                    "Round 2B-2.1：相似度由后端按同名同单位数值字段、"
                    "字段覆盖率和归一化距离确定性计算；海科导入数据未参与。"
                )
                if is_round2b_similarity
                else (
                    "明确或由上下文恢复的 Round 2A 材料研发意图优先读取业务 MySQL；"
                    "海科导入数据概览未参与本次执行。"
                )
            ),
            routing=routing,
        )

    # User-provided company real data is a deterministic local runtime route.
    # It never mutates the read-only business MySQL.
    company_decision = _classify_company_real_data_turn(
        body.message,
        body.history,
    )
    if company_data_has_priority(company_decision):
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
        emit_progress(
            "intent_routing",
            "completed",
            "命中逆向设计",
            f"已识别为 Project {project_id} 的 T17 多目标逆向设计。",
            intent="v014_inverse_design",
            project_id=project_id,
        )
        try:
            report = run_inverse_design_for_ui(
                runtime_root=_v014_runtime_root(),
                project_id=project_id,
                message=body.message,
            )
        except V014UIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        emit_progress(
            "inverse_design",
            "completed",
            "逆向设计完成",
            (
                f"计算状态为 {report.get('status', '-')}；"
                f"返回 {len(report.get('design_cards') or [])} 组可信设计。"
            ),
            result_status=report.get("status"),
            recommendation_count=len(report.get("design_cards") or []),
        )
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
        emit_progress(
            "intent_routing",
            "completed",
            "命中下一轮实验推荐",
            f"已识别为 Project {project_id} 的 T18 贝叶斯优化。",
            intent="v014_next_experiments",
            project_id=project_id,
        )
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

    field_catalog = None
    if looks_like_multi_condition_request(body.message):
        emit_progress(
            "schema_loading",
            "running",
            "读取授权字段目录",
            "正在读取当前权限范围内实际存在的字段名称和单位。",
        )
        try:
            field_catalog = (
                container.core.material_intelligence_skill.get_field_catalog(ctx)
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Round 2B-1.1 授权字段目录读取失败："
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        if not isinstance(field_catalog, dict) or field_catalog.get("status") != "ok":
            raise HTTPException(
                status_code=500,
                detail="Round 2B-1.1 授权字段目录不完整，已停止 Schema-Aware 路由。",
            )
        emit_progress(
            "schema_loading",
            "completed",
            "字段目录已加载",
            f"已加载 {field_catalog.get('total_field_count', 0)} 个授权字段定义。",
            field_count=field_catalog.get("total_field_count", 0),
        )

    engine = DeepSeekIntentRouter(container.llm)
    database_explorer_skill = getattr(
        container,
        "database_explorer_skill",
        None,
    )
    database_explorer_enabled = bool(
        database_explorer_skill is not None
        and getattr(database_explorer_skill, "enabled", False)
    )
    routing_meta: dict[str, Any] = {}
    needs_clarification = False
    clarification_question = ""

    try:
        emit_progress(
            "intent_routing",
            "running",
            "DeepSeek 语义路由",
            "正在结合当前问题和对话上下文提取业务意图。",
        )
        decision = engine.route(
            body.message,
            history,
            attachment_meta,
            field_catalog=field_catalog,
            database_explorer_enabled=database_explorer_enabled,
            database_explorer_mode=str(
                getattr(database_explorer_skill, "mode", "off")
            ),
        )
        router_name = "deepseek"
        summary = decision.reasoning_summary
        intent, tool_name, tool_args = decision.intent, decision.tool_name, decision.tool_args
        routing_meta = decision.to_routing_meta()
        needs_clarification = decision.needs_clarification
        clarification_question = decision.clarification_question
        emit_progress(
            "intent_routing",
            "completed",
            "问题理解完成",
            (
                f"已识别为“{_INTENT_PROGRESS_NAMES.get(intent, intent)}”。"
                + (f" {summary}" if summary else "")
            ),
            intent=intent,
            detail_items=_intent_progress_details(intent, tool_args),
            plan_summary=summary,
        )
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
                elif (
                    database_explorer_enabled
                    and _looks_like_unmatched_database_question(body.message)
                ):
                    intent, tool_name, tool_args = "database_explorer", None, {}
                    router_name = "database_explorer_fail_safe"
                    summary = (
                        "DeepSeek JSON 意图路由失败，但当前问题明确要求业务数据库事实；"
                        "转入授权只读 Database Explorer。"
                    )
                else:
                    # If the JSON router fails and no deterministic Tool rule
                    # applies, give DeepSeek one guarded natural-language answer
                    # attempt instead of returning the old capability template.
                    intent, tool_name, tool_args = "general_conversation", None, {}
                    router_name = "deepseek_answer_fallback"
                    summary = (
                        "DeepSeek JSON 意图路由失败，转入无 Tool 的受约束通用回答；"
                        "本轮不得声称使用数据库、附件或历史知识证据。"
                    )
            else:
                intent, tool_name, tool_args = (
                    fallback.intent,
                    fallback.tool_name,
                    fallback.tool_args,
                )
                router_name = "rule_fallback"
                summary = "DeepSeek 路由失败，使用 V0.1.1 规则路由兜底。"

    if not routing_meta:
        is_database_explorer = intent == "database_explorer"
        routing_meta = {
            "version": "DBE-0.1" if is_database_explorer else "fallback",
            "domain": (
                "retrieve"
                if is_database_explorer or tool_name is not None
                else "conversation"
            ),
            "primary_intent": intent,
            "secondary_intents": [],
            "entities": {},
            "scope": {"company": "current", "projects": "authorized"},
            "constraints": ({
                "read_only": True,
                "authorized_virtual_sources_only": True,
                "bounded_sql_retry": True,
            } if is_database_explorer else {}),
            "context_reference": {"action": "fallback"},
            "tool_plan": ([{
                "kind": "tool",
                "name": tool_name,
                "args": dict(tool_args),
                "purpose": "fallback execution",
            }] if tool_name else ([{
                "kind": "skill",
                "name": "database_explorer",
                "args": {},
                "purpose": "授权只读数据库探索兜底",
            }] if is_database_explorer else [])),
            "needs_clarification": False,
            "clarification_question": "",
        }

    if needs_clarification:
        return ChatUIResponse(
            answer=(clarification_question or "当前信息不足，请补充样品、指标或分析范围。"),
            intent="clarification_required",
            tool_name=None,
            tool_args=tool_args,
            data={
                "requested_intent": intent,
                "needs_clarification": True,
            },
            evidence=[],
            warnings=[],
            router=router_name,
            reasoning_summary=summary,
            routing=routing_meta,
        )

    if intent in {"analyze_current_attachment", "ask_current_attachment"}:
        if not body.attachment_ids:
            raise HTTPException(status_code=400, detail="请先上传 PDF、DOCX 或 XLSX 附件")
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
            routing=routing_meta,
        )

    if intent == "sample_historical_similarity":
        args = _resolve_sample_history_args(tool_args, ctx)
        try:
            result = container.sample_historical_similarity_skill.answer(
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
                detail=f"样品 + 历史资料相似分析失败：{type(exc).__name__}: {exc}",
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
            routing=routing_meta,
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
            routing=routing_meta,
        )

    if intent in {"search_historical_knowledge", "historical_similar_case"}:
        project_id = _resolve_historical_project_id(tool_args, ctx)
        try:
            result = container.historical_knowledge_skill.answer(
                message=(str(tool_args.get("history_query") or "").strip() or body.message),
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
            tool_args={
                "project_id": project_id,
                "history_query": str(tool_args.get("history_query") or "").strip(),
            },
            data=result,
            evidence=result.get("evidence", []),
            warnings=result.get("warnings", []),
            router=router_name,
            reasoning_summary=summary,
            routing=routing_meta,
        )

    if intent == "database_explorer":
        if not database_explorer_enabled or database_explorer_skill is None:
            raise HTTPException(
                status_code=503,
                detail="Database Explorer 当前未启用。",
            )
        try:
            result = database_explorer_skill.answer(
                message=body.message,
                history=history,
                ctx=ctx,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Database Explorer 执行失败：{type(exc).__name__}: {exc}",
            ) from exc
        return ChatUIResponse(
            answer=result.get("answer", ""),
            intent=intent,
            tool_name="database_explorer",
            tool_args={},
            data=result,
            evidence=result.get("evidence", []),
            warnings=result.get("warnings", []),
            router="deepseek_database_explorer",
            reasoning_summary=(
                summary
                or "未命中高精度意图，转入授权只读 Database Explorer。"
            ),
            routing=routing_meta,
        )

    if intent in {"general_conversation", "unsupported_future_feature"}:
        try:
            result = container.general_conversation_skill.answer(
                message=body.message,
                history=history,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"DeepSeek 通用回答失败：{type(exc).__name__}: {exc}",
            ) from exc
        return ChatUIResponse(
            answer=result.get("answer", ""),
            intent=intent,
            tool_name=None,
            tool_args={},
            data=result,
            evidence=[],
            warnings=result.get("warnings", []),
            router=(
                "deepseek_general_answer"
                if router_name == "deepseek"
                else router_name
            ),
            reasoning_summary=summary,
            routing=routing_meta,
        )

    if tool_name is None:
        raise HTTPException(400, "已识别意图，但没有可执行 Tool")

    try:
        emit_progress(
            "tool_execution",
            "running",
            "执行材料工具",
            f"正在执行只读工具：{tool_name}。",
            intent=intent,
        )
        result = container.core.execute(intent, tool_name, tool_args, ctx)
        emit_progress(
            "tool_execution",
            "completed",
            "材料工具执行完成",
            "结构化数据库证据已返回。",
            intent=intent,
        )
        emit_progress(
            "answer_generation",
            "running",
            "组织回答",
            "正在将结构化事实转换为中文回答。",
        )
        answer = container.core.answer(body.message, intent, result)
        emit_progress(
            "answer_generation",
            "completed",
            "回答已生成",
            "最终回答已依据本轮工具证据生成。",
        )
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
        routing=routing_meta,
    )


_chat_ui_graph = build_chat_ui_graph(
    _execute_chat_ui_legacy,
    primary_classifier=_classify_chat_ui_primary_family,
    semantic_planner=_plan_chat_ui_semantic,
    semantic_executors={
        "database_explorer": _execute_semantic_database_explorer,
        "rag": _execute_semantic_rag,
        "current_attachment": _execute_semantic_current_attachment,
        "general_conversation": _execute_semantic_general_conversation,
        "material_tool": _execute_semantic_material_tool,
    },
)


@router.post("/chat-ui", response_model=ChatUIResponse)
def chat_ui(
    body: ChatUIRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    """Run the production Chat UI request through LangGraph V4.

    V4 keeps V3 checkpoints and moves semantic planning plus the five semantic
    execution families into native graph nodes.
    """

    try:
        return invoke_chat_ui_graph(
            _chat_ui_graph,
            body=body,
            user_context=ctx,
            container=container,
        )
    except ChatUIWorkflowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatUIWorkflowPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ChatUIWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ChatUIWorkflowCheckpointError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/chat-ui/workflows/{workflow_id}")
def chat_ui_workflow_status(
    workflow_id: str,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    """Return a scoped, redacted V4 checkpoint status without answer data."""

    try:
        return container.chat_ui_workflow_store.status(workflow_id, ctx)
    except ChatUIWorkflowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatUIWorkflowPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ChatUIWorkflowConflictError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChatUIWorkflowCheckpointError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat-ui/stream")
async def chat_ui_stream(
    body: ChatUIRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    """SSE companion endpoint for auditable, user-safe execution progress.

    The existing `/chat-ui` response contract stays unchanged.  Frontends may
    use this endpoint to receive progress events and one final ChatUIResponse.
    """
    event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Queue one event before starting the worker. This makes the streaming
    # contract observable immediately, even when the selected DB/ML operation
    # takes a long time before it can emit its first domain-specific update.
    event_queue.put_nowait((
        "progress",
        initial_stream_progress_event(),
    ))

    def publish(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, ("progress", event))

    def run_request() -> None:
        try:
            with progress_context(publish):
                response = chat_ui(body=body, ctx=ctx, container=container)
                emit_progress(
                    "request_complete",
                    "completed",
                    "分析完成",
                    "已完成本轮受控分析并生成最终结果。",
                )
            if hasattr(response, "model_dump"):
                payload = response.model_dump(mode="json")
            elif hasattr(response, "dict"):
                payload = response.dict()
            else:
                payload = dict(response)
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                ("result", payload),
            )
        except HTTPException as exc:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                ("error", {
                    "status_code": exc.status_code,
                    "detail": str(exc.detail),
                }),
            )
        except Exception as exc:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                ("error", {
                    "status_code": 500,
                    "detail": f"{type(exc).__name__}: {exc}",
                }),
            )

    task = asyncio.create_task(asyncio.to_thread(run_request))

    async def event_stream():
        try:
            while True:
                event_name, payload = await event_queue.get()
                encoded = json.dumps(payload, ensure_ascii=False, default=str)
                yield f"event: {event_name}\ndata: {encoded}\n\n"
                if event_name in {"result", "error"}:
                    break
        finally:
            if not task.done():
                # The underlying read-only worker may already be inside a DB/LLM
                # call and cannot be force-killed safely. Cancelling only drops
                # the asyncio waiter when the browser disconnects.
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Materials-Progress": "sse-v1.1",
        },
    )
