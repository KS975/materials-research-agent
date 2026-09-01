from __future__ import annotations

import json
import re
from typing import Any

from agent.conversation_context import (
    CONVERSATION_CONTEXT_SCHEMA_VERSION,
    ConversationHints,
    build_conversation_hints,
)
from agent.intent_v2 import DeepSeekIntentDecision, IntentToolPlanStep
from agent.field_catalog import (
    bind_filters_to_catalog,
    field_catalog_for_prompt,
)
from agent.multi_condition import (
    looks_like_multi_condition_request,
    normalize_filters,
    normalize_logic,
    unit_is_explicit_in_text,
)
from agent.router import IntentDecision, RuleIntentRouter
from llm.base import LLMProvider


INTENT_ROUTER_CONTEXT_SCHEMA_VERSION = "2.1.2"


_ALLOWED_TOOLS = {
    "get_sample_context",
    "get_formula",
    "get_process",
    "get_performance",
    "compare_samples",
    "find_samples",
    "list_samples_for_analysis",
}

_ALLOWED_INTENTS = _ALLOWED_TOOLS | {
    "sample_full_profile",
    "formula_difference",
    "process_difference",
    "comparability_check",
    "performance_rank",
    "performance_statistics",
    "experiment_series_analysis",
    "data_quality_check",
    "find_samples_multi_condition",
    "similar_samples",
    "historical_similar_case",
    "analyze_cause",
    "analyze_performance_difference",
    "analyze_current_attachment",
    "ask_current_attachment",
    "search_historical_knowledge",
    "historical_similar_case",
    "sample_historical_similarity",
    "joint_mysql_knowledge_analysis",
    "database_explorer",
    "general_conversation",
    "unsupported_future_feature",
    "clarification_required",
}

_SPECIAL_NO_TOOL_INTENTS = {
    "analyze_current_attachment",
    "ask_current_attachment",
    "search_historical_knowledge",
    "sample_historical_similarity",
    "joint_mysql_knowledge_analysis",
    "database_explorer",
    "general_conversation",
    "unsupported_future_feature",
    "clarification_required",
}

_INTENT_ALIASES = {
    # Common LLM inventions / synonyms seen in practice.
    "query_sample_data": "get_sample_context",
    "query_sample": "get_sample_context",
    "sample_lookup": "get_sample_context",
    "sample_query": "get_sample_context",
    "full_sample_profile": "sample_full_profile",
    "sample_profile": "sample_full_profile",
    "query_formula": "get_formula",
    "sample_formula": "get_formula",
    "formula_lookup": "get_formula",
    "query_process": "get_process",
    "sample_process": "get_process",
    "process_lookup": "get_process",
    "query_performance": "get_performance",
    "sample_performance": "get_performance",
    "performance_lookup": "get_performance",
    "search_samples": "find_samples",
    "sample_search": "find_samples",
    "filter_samples": "list_samples_for_analysis",
    "multi_condition_sample_search": "list_samples_for_analysis",
    "filter_samples": "find_samples_multi_condition",
    "multi_condition_sample_search": "find_samples_multi_condition",
    "find_similar_samples": "similar_samples",
    "sample_similarity": "similar_samples",
    "formula_similarity": "similar_samples",
    "process_similarity": "similar_samples",
    "metric_stats": "performance_statistics",
    "performance_stats": "performance_statistics",
    "performance_average": "performance_statistics",
    "sample_compare": "compare_samples",
    "compare_sample": "compare_samples",
    "formula_compare": "formula_difference",
    "compare_formula": "formula_difference",
    "formula_diff": "formula_difference",
    "process_compare": "process_difference",
    "compare_process": "process_difference",
    "process_diff": "process_difference",
    "comparability": "comparability_check",
    "can_compare": "comparability_check",
    "performance_difference_analysis": "analyze_performance_difference",
    "root_cause_analysis": "analyze_performance_difference",
    "historical_search": "search_historical_knowledge",
    "historical_similarity_search": "search_historical_knowledge",
    "historical_case": "historical_similar_case",
    "cross_project_search": "search_historical_knowledge",
    "sample_history": "sample_historical_similarity",
    "sample_historical_case": "sample_historical_similarity",
    "sample_historical_similarity": "sample_historical_similarity",
    "historical_similar_sample": "sample_historical_similarity",
    "joint_sample_history": "sample_historical_similarity",
    "joint_analysis": "joint_mysql_knowledge_analysis",
    "joint_database_knowledge_analysis": "joint_mysql_knowledge_analysis",
    "attachment_summary": "analyze_current_attachment",
    "attachment_analysis": "analyze_current_attachment",
    "attachment_qa": "ask_current_attachment",
    "ask_attachment": "ask_current_attachment",
    "general_answer": "general_conversation",
    "general_chat": "general_conversation",
    "general_knowledge": "general_conversation",
    "conversation": "general_conversation",
}

_TOOL_ALIASES = {
    "query_sample_data": "get_sample_context",
    "query_sample": "get_sample_context",
    "sample_lookup": "get_sample_context",
    "sample_full_profile": "get_sample_context",
    "sample_profile": "get_sample_context",
    "query_formula": "get_formula",
    "formula_lookup": "get_formula",
    "query_process": "get_process",
    "process_lookup": "get_process",
    "query_performance": "get_performance",
    "performance_lookup": "get_performance",
    "sample_compare": "compare_samples",
    "compare_sample": "compare_samples",
    "formula_compare": "compare_samples",
    "compare_formula": "compare_samples",
    "formula_difference": "compare_samples",
    "process_compare": "compare_samples",
    "compare_process": "compare_samples",
    "process_difference": "compare_samples",
    "comparability": "compare_samples",
    "can_compare": "compare_samples",
    "comparability_check": "compare_samples",
    "search_samples": "find_samples",
    "sample_search": "find_samples",
    "performance_rank": "list_samples_for_analysis",
    "performance_statistics": "list_samples_for_analysis",
    "experiment_series_analysis": "list_samples_for_analysis",
    "data_quality_check": "list_samples_for_analysis",
    "find_samples_multi_condition": "list_samples_for_analysis",
    "similar_samples": "list_samples_for_analysis",
}

_DOMAIN_BY_INTENT = {
    "get_sample_context": "retrieve",
    "get_formula": "retrieve",
    "get_process": "retrieve",
    "get_performance": "retrieve",
    "find_samples": "retrieve",
    "compare_samples": "compare",
    "sample_full_profile": "retrieve",
    "formula_difference": "compare",
    "process_difference": "compare",
    "comparability_check": "validate",
    "performance_rank": "analyze",
    "performance_statistics": "analyze",
    "experiment_series_analysis": "analyze",
    "data_quality_check": "validate",
    "find_samples_multi_condition": "retrieve",
    "similar_samples": "analyze",
    "historical_similar_case": "knowledge",
    "analyze_cause": "analyze",
    "analyze_performance_difference": "diagnosis",
    "analyze_current_attachment": "attachment",
    "ask_current_attachment": "attachment",
    "search_historical_knowledge": "knowledge",
    "sample_historical_similarity": "diagnosis",
    "joint_mysql_knowledge_analysis": "diagnosis",
    "database_explorer": "retrieve",
    "general_conversation": "conversation",
    "unsupported_future_feature": "system",
    "clarification_required": "conversation",
}

_REQUIRED_ARGS = {
    "get_sample_context": ("identifier",),
    "get_formula": ("identifier",),
    "get_process": ("identifier",),
    "get_performance": ("identifier",),
    "find_samples": ("keyword",),
    "compare_samples": ("left_identifier", "right_identifier"),
    "sample_full_profile": ("identifier",),
    "formula_difference": ("left_identifier", "right_identifier"),
    "process_difference": ("left_identifier", "right_identifier"),
    "comparability_check": ("left_identifier", "right_identifier"),
    "performance_rank": ("target_metric",),
    "performance_statistics": ("target_metric",),
    "experiment_series_analysis": ("keyword",),
    "find_samples_multi_condition": ("filters",),
    "similar_samples": ("identifier",),
    "analyze_cause": ("identifier",),
    "analyze_performance_difference": (
        "left_identifier",
        "right_identifier",
        "target_metric",
    ),
    "sample_historical_similarity": ("identifier",),
    "joint_mysql_knowledge_analysis": (
        "left_identifier",
        "right_identifier",
        "target_metric",
    ),
}

_EXPECTED_TOOL_BY_INTENT = {
    "get_sample_context": "get_sample_context",
    "get_formula": "get_formula",
    "get_process": "get_process",
    "get_performance": "get_performance",
    "compare_samples": "compare_samples",
    "sample_full_profile": "get_sample_context",
    "formula_difference": "compare_samples",
    "process_difference": "compare_samples",
    "comparability_check": "compare_samples",
    "performance_rank": "list_samples_for_analysis",
    "performance_statistics": "list_samples_for_analysis",
    "experiment_series_analysis": "list_samples_for_analysis",
    "data_quality_check": "list_samples_for_analysis",
    "find_samples_multi_condition": "list_samples_for_analysis",
    "similar_samples": "list_samples_for_analysis",
    "find_samples": "find_samples",
    "analyze_cause": "get_sample_context",
    "analyze_performance_difference": "compare_samples",
}

_ALLOWED_DOMAINS = {
    "retrieve",
    "compare",
    "analyze",
    "diagnosis",
    "attachment",
    "knowledge",
    "validate",
    "predict_optimize",
    "plan_execute",
    "conversation",
    "system",
}


class DeepSeekIntentRouter:
    """Intent Router V2.

    DeepSeek only classifies intent and extracts parameters. It never generates
    SQL and it never gets to execute arbitrary tool names. The backend
    normalizes aliases, validates the primary intent/tool pair, derives a safe
    tool plan, and preserves the V1 ``intent/tool_name/tool_args`` interface.
    """

    def __init__(self, llm: LLMProvider):
        if CONVERSATION_CONTEXT_SCHEMA_VERSION != INTENT_ROUTER_CONTEXT_SCHEMA_VERSION:
            raise RuntimeError(
                "Intent Router 与 Conversation Context 版本不一致："
                f"router={INTENT_ROUTER_CONTEXT_SCHEMA_VERSION}, "
                f"context={CONVERSATION_CONTEXT_SCHEMA_VERSION}"
            )
        self.llm = llm

    def route(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        field_catalog: dict[str, Any] | None = None,
        database_explorer_enabled: bool = False,
        database_explorer_mode: str = "off",
    ) -> DeepSeekIntentDecision:
        hints = build_conversation_hints(message, history)
        system_prompt = self._system_prompt()
        payload = {
            "conversation_history": (history or [])[-12:],
            "current_message": message,
            "current_chat_attachments": attachments or [],
            "backend_context_hints": hints.to_dict(),
            "authorized_material_field_catalog": field_catalog_for_prompt(
                field_catalog,
                message,
            ),
            "backend_capabilities": {
                "database_explorer": {
                    "enabled": bool(database_explorer_enabled),
                    "mode": str(database_explorer_mode or "off"),
                },
            },
        }
        raw = self.llm.complete(
            system_prompt,
            json.dumps(payload, ensure_ascii=False),
        ).strip()
        data = self._parse_json_object(raw)
        return self._normalize_decision(
            data=data,
            message=message,
            history=history or [],
            attachments=attachments or [],
            hints=hints,
            field_catalog=field_catalog,
            database_explorer_enabled=database_explorer_enabled,
        )

    @staticmethod
    def _system_prompt() -> str:
        return """
你是“材数智能体”的 Intent Router V2。你只负责理解用户意图、参数、上下文指代和执行计划，不回答材料问题，不生成 SQL，不直接控制设备。

当前此路由器可直接落地的业务能力：
1) 数据库只读 Tool：
- get_sample_context(identifier)
- get_formula(identifier)
- get_process(identifier)
- get_performance(identifier)
- compare_samples(left_identifier, right_identifier)
- find_samples(keyword)

1A) Materials Intent Round 2A-1（确定性材料研发分析）：
- sample_full_profile：用户明确要求“完整/全面/所有信息”查看一个样品；tool_name=get_sample_context。
- formula_difference：比较两个样品“配方差在哪/改了什么配方”；tool_name=compare_samples。
- process_difference：比较两个样品“工艺差在哪/改了什么工艺”；tool_name=compare_samples。
- comparability_check：判断两个样品“能不能直接比较/是否可比”；tool_name=compare_samples；target_metric 可选。
这些意图的差值、相对变化、配方算术和、可比性等级由后端确定性计算，Router 不负责计算。

1B) Materials Intent Round 2A-2（确定性集合分析）：
- performance_rank：按明确性能指标对授权范围样品排序；参数 target_metric，可选 keyword/top_n/order；tool_name=list_samples_for_analysis。
- performance_statistics：计算授权范围样品某个明确性能指标的平均值；参数 target_metric，requested_statistics=["mean"]，keyword 默认空字符串；tool_name=list_samples_for_analysis。缺失值、非数值、单位检查和平均值全部由后端计算。
- experiment_series_analysis：分析实验系列中的变量、常量和缺失；参数 keyword（如 N20260305）；tool_name=list_samples_for_analysis。
- data_quality_check：检查授权范围或 keyword 范围内的缺失、重复、非数值和配方记录值算术和；tool_name=list_samples_for_analysis。
- historical_similar_case：仅检索“以前有没有类似情况/案例”的历史 Knowledge；tool_name=null。若明确针对单一样品，使用 sample_historical_similarity。
排序、计数、比例和异常标记全部由后端 Python 计算，Router 不计算。

1C) Materials Intent Round 2B-1（多条件样品筛选）：
- find_samples_multi_condition：用户按一个或多个样品基础信息、配方、工艺、性能或测试条件查找样品；tool_name=list_samples_for_analysis。
- tool_args 结构：
  {
    "filters":[
      {"section":"sample|formula|process|performance|conditions", "field":"字段中文名", "operator":"eq|ne|gt|gte|lt|lte|between|in|contains|exists|missing", "value":单值, "values":数组, "unit":"用户明确写出的单位"}
    ],
    "logic":"and|or",
    "keyword":"可选的样品名/实验系列关键词",
    "result_limit":50
  }
- between 使用恰好两个 values；in 使用 values；exists/missing 不要输出 value；其它 operator 使用 value。
- section=sample 只允许 field=id/name/project_id/sample_type/create_time；测试条件整体是否有记录时用 section=conditions, field="*", operator=exists/missing。
- unit 只有用户原句明确写出时才能输出，绝对不要根据常识或数据库字段名补单位。后端不自动换算单位。
- keyword 只用于用户明确给出的样品名称或系列名称范围（如 N20260305），不要把整句筛选要求塞进 keyword。
- 只生成上述结构化条件；严禁生成 SQL、表名、列名、JOIN、WHERE 或数据库连接信息。实际读取、权限过滤、数值比较、缺失处理和计数均由后端确定性执行。
- 若输入中的 authorized_material_field_catalog 非空，它是当前用户授权范围内的权威字段目录。section 和 field 必须从目录选择并使用 canonical field 原名：例如目录存在 formula.PC 时，“PC含量”必须输出 field="PC"；目录只在 performance 中存在“成本”时，必须输出 section="performance", field="成本"。不得把“含量/数值/指标”等口语后缀拼进新字段名。

1D) Materials Intent Round 2B-2.1（确定性相似样品）：
- similar_samples：查找与一个明确参照样品结构化数值最接近的样品；tool_name=list_samples_for_analysis。
- 参数：identifier；similarity_scope="formula|process|combined"；top_n 默认5，最大20；keyword 默认空字符串。
- “配方最像/配方相似”使用 formula；“工艺最像/工艺相似”使用 process；未限定或说“综合最像”使用 combined。
- 与“历史上有没有类似案例”严格区分：历史资料问题仍走 sample_historical_similarity / historical_similar_case，不得走 similar_samples。
- Router 不计算相似度；字段单位对齐、缺失覆盖、归一化距离和排名全部由后端确定性执行。

2) 性能差异分析：
- primary_intent=analyze_performance_difference
- tool_name=compare_samples
- 参数：left_identifier、right_identifier、target_metric、direction_claim

3) 当前 Chat 临时附件：
- 总结/分析当前附件：analyze_current_attachment，tool_name=null
- 针对附件问具体问题：ask_current_attachment，tool_name=null
- 当前附件与长期知识库是不同证据源。

4) 历史 Knowledge / RAG：
- search_historical_knowledge，tool_name=null：仅按历史问题检索。
- sample_historical_similarity，tool_name=null：当用户问“这个样品/3811 历史上有没有类似问题、类似异常、类似案例”时使用；至少需要 identifier，可选 target_metric。该能力先读取当前样品数据库事实，再与历史资料联合判断相似点。
- 用户明确项目号时提取 project_id；没有明确项目号时不要编造，默认由后端在当前 Company 授权范围内检索全部项目。

5) MySQL + 历史知识联合分析：
- joint_mysql_knowledge_analysis，tool_name=null
- 必须提取 left_identifier、right_identifier、target_metric、direction_claim；只有用户明确项目号时才给 project_id。

6) Database Explorer 兜底：
- database_explorer，tool_name=null。
- 只有输入的 backend_capabilities.database_explorer.enabled=true 时才允许使用。
- 仅用于“需要当前业务 MySQL 事实、但不属于任何已定义高精度意图”的开放数据库问题。
- 已定义的样品查看、比较、排名、实验系列、多条件筛选、历史知识、附件等意图永远优先，不能为了自由查询而降级到 database_explorer。
- 此 Intent Router 仍然严禁生成 SQL；后续独立的受控 Database Explorer 会读取授权虚拟 Schema、生成并校验只读 SQL。
- 与当前业务数据库无关的材料知识、日常问题和普通讨论仍使用 general_conversation。

高级 Dataset/ML/BO/V0.2/V0.3 请求在进入本路由器之前通常由上层确定性路由处理。
- 普通材料知识、研发方法、概念解释、建议讨论、寒暄或其它不需要当前数据库/附件/RAG事实的问题，使用 general_conversation，tool_name=null，由 DeepSeek 通用回答层直接回答。
- 要求当前数据库/样品/附件中的事实但缺少必要对象时，优先 needs_clarification=true，不要降级为没有证据的通用知识回答。
- 明确要求尚未接入的真实设备控制、写数据库或其它执行动作时，使用 unsupported_future_feature；后端会让 DeepSeek解释能力边界，但不会伪装已经执行。

【对话智能规则】
- 你可以参考 conversation_history 和 backend_context_hints 解析“这个样品、它、刚才那个、只看配方、继续、不是3811是3812”等表达。
- 对“继续/只看/只查看/我想看/我可以只查看/换成/不是A是B”等 follow-up，不要把 primary_intent 写成 continue_previous/user_correction。primary_intent 必须仍是最终要执行的业务意图；把对话动作写入 context_reference.action，并把最终参数直接修正好。
- “我可以只查看性能么 / 我只想看看配方 / 那只看工艺”若本轮没有明确新样品，必须复用当前 active sample；绝对不要把“看性能么/看看配方/工艺呢”等自然语言片段当成 identifier。
- “这两个样品/两个样品/两者/它们”必须优先复用 backend_context_hints.active_comparison_identifiers；例如上一轮比较3811和3809配方，本轮问“这两个样品的工艺有什么区别”，必须输出 process_difference + compare_samples(3811, 3809)。没有可恢复的两个样品时必须请求澄清，不能转成单位/海科数据概览。
- “Project 115呢？/那全部项目呢？”如果上一轮正在做历史检索，只表示修改历史检索 scope，必须继承上一轮历史任务和检索主题，不得把问题重置成“Project 115 是什么”。
- “历史上有没有和这个类似的冲击强度异常？”若 history 能唯一确定“这个”是哪一个样品，应使用 sample_historical_similarity，并复用该样品。
- 如果历史中能唯一确定样品，可以复用；若有多个候选且无法确定，needs_clarification=true。
- 不得编造样品 ID、项目 ID、性能指标。
- “找/筛选/列出冲击强度大于40且成本低于30的样品”“项目115里PC含量大于50%的样品”属于 find_samples_multi_condition，不要降级成 find_samples(keyword)，也不要生成 SQL。
- “找和3811最像的5个样品/找配方与3811相似的样品”属于 similar_samples，不要降级成 find_samples(keyword)，也不要交给 Database Explorer。
- 一句话包含多个需求时：primary_intent 表示当前首先要执行/最终落地的主业务意图；其它语义放 secondary_intents。现阶段只有后端已验证的 Tool/Skill 会真正执行。
- 同一句话明确要求“数据库样品事实 + 历史资料”时，必须用 joint_mysql_knowledge_analysis，不能降级成单一数据源。
- 有附件且用户明确问“这份/附件/报告/表格/Excel”的内容时，优先走 current attachment。
- 信息不足时不要猜。设置 needs_clarification=true，并给出 clarification_question。

【primary_intent 白名单】
get_sample_context, get_formula, get_process, get_performance, compare_samples, find_samples,
sample_full_profile, formula_difference, process_difference, comparability_check,
performance_rank, performance_statistics, experiment_series_analysis, data_quality_check, find_samples_multi_condition, similar_samples, historical_similar_case,
analyze_cause, analyze_performance_difference,
analyze_current_attachment, ask_current_attachment,
search_historical_knowledge, sample_historical_similarity, joint_mysql_knowledge_analysis,
database_explorer, general_conversation, unsupported_future_feature, clarification_required

【tool_name 白名单】
get_sample_context, get_formula, get_process, get_performance, compare_samples, find_samples, list_samples_for_analysis，或 null。
数据库 primary_intent 必须与对应 tool 一致；不要发明 query_sample_data 之类的新名字。

只输出一个 JSON 对象，推荐 V2 格式：
{
  "domain":"retrieve|compare|analyze|diagnosis|attachment|knowledge|validate|predict_optimize|plan_execute|conversation|system",
  "primary_intent":"...",
  "secondary_intents":["..."],
  "entities":{},
  "scope":{"company":"current","projects":"all_authorized或明确项目"},
  "constraints":{},
  "context_reference":{
    "action":"new_request|follow_up_reference|continue_previous|refine_previous|user_correction",
    "use_previous_sample":false,
    "use_previous_scope":false,
    "use_previous_task":false,
    "use_current_attachment":false
  },
  "tool_name":"...或null",
  "tool_args":{},
  "tool_plan":[],
  "needs_clarification":false,
  "clarification_question":"",
  "reasoning_summary":"一句简短可展示说明"
}

为了兼容旧版本，也允许你输出旧字段 intent 代替 primary_intent，但优先使用 V2。
""".strip()

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.I | re.S)
        if fence:
            raw = fence.group(1)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek 意图结果不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("DeepSeek 意图结果必须是 JSON 对象")
        return data

    def _normalize_decision(
        self,
        *,
        data: dict[str, Any],
        message: str,
        history: list[dict[str, str]],
        attachments: list[dict[str, Any]],
        hints: ConversationHints,
        field_catalog: dict[str, Any] | None = None,
        database_explorer_enabled: bool = False,
    ) -> DeepSeekIntentDecision:
        raw_intent = str(
            data.get("primary_intent")
            or data.get("intent")
            or ""
        ).strip()
        intent = _INTENT_ALIASES.get(raw_intent, raw_intent)

        raw_tool = data.get("tool_name")
        tool_name = None if raw_tool is None else str(raw_tool).strip()
        if tool_name:
            tool_name = _TOOL_ALIASES.get(tool_name, tool_name)
        else:
            tool_name = None

        # V2.1 deterministic task continuity. A scope-only follow-up must keep
        # the previous historical task. A historical request that explicitly
        # names one sample (or refers to the active sample as “这个/它”) is a
        # sample+history analysis, even if the LLM tries to reduce it to a
        # generic RAG search.
        forced_history_intent = None
        if (hints.scope_only_followup or hints.scope_reset_followup) and hints.active_history_task:
            forced_history_intent = hints.active_history_task
        elif hints.current_history_request:
            current_samples = list(hints.current_sample_identifiers)
            if len(current_samples) == 1:
                forced_history_intent = "sample_historical_similarity"
            elif (
                not current_samples
                and hints.current_history_referential
                and hints.active_sample_identifier
            ):
                forced_history_intent = "sample_historical_similarity"
        if forced_history_intent:
            intent = forced_history_intent
            tool_name = None
        elif hints.task_refinement_intent:
            # V2.1.1: low-risk single-sample view requests are deterministic.
            # Do not let the model turn text fragments such as “看性能么” into
            # a sample identifier or choose a broader sample profile intent.
            intent = hints.task_refinement_intent
            tool_name = _EXPECTED_TOOL_BY_INTENT[intent]
        else:
            specialized = self._deterministic_material_intent(message, hints)
            if specialized:
                intent = specialized
                tool_name = _EXPECTED_TOOL_BY_INTENT[intent]

        args = data.get("tool_args") or data.get("arguments") or {}
        if not isinstance(args, dict):
            raise ValueError("tool_args 必须是对象")
        args = dict(args)

        # Unknown intent + valid tool: use the executable tool intent rather than
        # failing on a harmless LLM synonym. This is the production guard that
        # prevents query_sample_data-like inventions from breaking the chat.
        if intent not in _ALLOWED_INTENTS and tool_name in _ALLOWED_TOOLS:
            intent = tool_name

        if intent not in _ALLOWED_INTENTS:
            raise ValueError(f"未允许的 intent: {raw_intent or intent}")
        if tool_name is not None and tool_name not in _ALLOWED_TOOLS:
            raise ValueError(f"未允许的 tool_name: {tool_name}")

        # The router may know this future intent, but it must never activate it
        # unless the backend explicitly enabled the guarded explorer framework.
        if intent == "database_explorer" and not database_explorer_enabled:
            intent = "general_conversation"
            tool_name = None

        if intent in _SPECIAL_NO_TOOL_INTENTS:
            tool_name = None
        else:
            expected_tool = _EXPECTED_TOOL_BY_INTENT.get(intent)
            if expected_tool is None:
                raise ValueError(f"intent={intent} 没有定义可执行 Tool")
            if tool_name is None:
                tool_name = expected_tool
            elif tool_name != expected_tool:
                raise ValueError(
                    f"intent/tool 冲突：intent={intent} 只能绑定 {expected_tool}，"
                    f"不能绑定 {tool_name}"
                )

        context_reference = self._normalize_context_reference(
            data.get("context_reference"),
            hints,
            bool(attachments),
        )
        args = self._apply_context_to_args(
            intent=intent,
            args=args,
            hints=hints,
            context_reference=context_reference,
            message=message,
        )
        args = self._sanitize_project_scope_args(
            intent=intent,
            args=args,
            hints=hints,
            context_reference=context_reference,
        )
        args = self._sanitize_tool_args(intent=intent, args=args)
        filter_validation_errors: list[str] = []
        filter_bindings: list[dict[str, Any]] = []
        if intent == "find_samples_multi_condition":
            (
                args,
                filter_validation_errors,
                filter_bindings,
            ) = self._normalize_multi_condition_args(
                args,
                message,
                field_catalog,
            )

        secondary_intents = self._normalize_secondary_intents(
            data.get("secondary_intents"),
            hints.action_hint,
        )
        entities = self._normalize_mapping(data.get("entities"))
        entities = self._enrich_entities(entities, args, hints)
        scope = self._normalize_scope(
            data.get("scope"),
            args,
            hints,
            context_reference,
        )
        constraints = self._normalize_mapping(data.get("constraints"))
        if intent == "find_samples_multi_condition":
            scope["data_source"] = "business_mysql"
            constraints.update({
                "read_only": True,
                "deterministic_filtering": True,
                "arbitrary_sql": False,
                "unit_conversion": False,
            })
            if filter_validation_errors:
                constraints["filter_validation_errors"] = filter_validation_errors
            if filter_bindings:
                constraints["field_bindings"] = filter_bindings
            if field_catalog and field_catalog.get("status") == "ok":
                constraints["field_catalog"] = {
                    "schema_version": field_catalog.get("schema_version"),
                    "total_field_count": field_catalog.get("total_field_count", 0),
                    "source_sample_count": field_catalog.get("source_sample_count", 0),
                    "contains_values": False,
                }
        elif intent == "similar_samples":
            scope["data_source"] = "business_mysql"
            constraints.update({
                "read_only": True,
                "deterministic_similarity": True,
                "exact_field_and_unit_alignment": True,
                "arbitrary_sql": False,
            })
        elif intent == "database_explorer":
            scope["data_source"] = "business_mysql"
            constraints.update({
                "read_only": True,
                "authorized_virtual_sources_only": True,
                "company_project_scope_enforced_by_backend": True,
                "bounded_sql_retry": True,
                "arbitrary_physical_table_access": False,
            })

        model_needs_clarification = bool(data.get("needs_clarification", False))
        missing = self._missing_required_args(intent, args)
        needs_clarification = (
            model_needs_clarification
            or bool(missing)
            or bool(filter_validation_errors)
        )
        clarification_question = str(data.get("clarification_question") or "").strip()
        if filter_validation_errors:
            clarification_question = (
                "我没能安全解析完整的筛选条件。请明确写出字段、比较关系和值，"
                "例如“冲击强度大于40，并且成本低于30”；需要限定单位时也请写出单位。"
            )
        elif needs_clarification and not clarification_question:
            clarification_question = self._clarification_question(intent, missing)

        domain = str(data.get("domain") or "").strip()
        if intent in {
            "sample_full_profile",
            "formula_difference",
            "process_difference",
            "comparability_check",
            "performance_rank",
            "performance_statistics",
            "experiment_series_analysis",
            "data_quality_check",
            "find_samples_multi_condition",
            "similar_samples",
            "database_explorer",
        }:
            domain = _DOMAIN_BY_INTENT[intent]
        elif domain not in _ALLOWED_DOMAINS:
            domain = _DOMAIN_BY_INTENT.get(intent, "conversation")

        plan = self._derive_safe_tool_plan(intent, tool_name, args)
        reasoning_summary = str(data.get("reasoning_summary") or "").strip()

        return DeepSeekIntentDecision(
            domain=domain,
            primary_intent=intent,
            tool_name=tool_name,
            tool_args=args,
            secondary_intents=secondary_intents,
            entities=entities,
            scope=scope,
            constraints=constraints,
            context_reference=context_reference,
            tool_plan=plan,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            reasoning_summary=reasoning_summary,
            router_version=(
                "DBE-0.1"
                if intent == "database_explorer"
                else "2B-1.1"
                if intent == "find_samples_multi_condition"
                else "2B-2.1"
                if intent == "similar_samples"
                else "2.1"
            ),
        )

    @staticmethod
    def _deterministic_material_intent(
        message: str, hints: ConversationHints
    ) -> str | None:
        """Specialize only very explicit Round 2A phrases.

        The rule does not replace semantic routing; it prevents a correct but
        overly broad ``compare_samples`` classification from hiding a clear
        user request for formula/process differences or comparability.
        """
        text = str(message or "").strip()
        samples = list(hints.current_sample_identifiers)
        if (
            not samples
            and hints.current_pair_referential
            and len(hints.active_comparison_identifiers) >= 2
        ):
            samples = list(hints.active_comparison_identifiers[:2])

        if RuleIntentRouter._route_similar_samples(text) is not None:
            return "similar_samples"

        if looks_like_multi_condition_request(text):
            return "find_samples_multi_condition"

        if "样品" in text and any(marker in text.lower() for marker in (
            "最好", "最高", "最低", "排序", "排名", "前几", "top",
        )):
            return "performance_rank"
        if RuleIntentRouter._route_performance_statistics(text) is not None:
            return "performance_statistics"
        if any(marker in text for marker in (
            "这一组在研究什么", "这组在研究什么",
            "这一组实验在研究什么", "这组实验在研究什么",
            "实验系列", "这一系列",
        )):
            return "experiment_series_analysis"
        if "数据" in text and any(marker in text for marker in (
            "有没有问题", "质量检查", "缺失", "异常", "重复",
        )):
            return "data_quality_check"

        if len(samples) >= 2:
            if any(marker in text for marker in (
                "能直接比较", "可以直接比较", "能不能比较", "可以比较吗",
                "是否可比", "可比吗", "可比性", "适合比较",
            )):
                return "comparability_check"

            diff_markers = (
                "差在哪", "差异", "区别", "不同", "改了什么", "变化",
                "变了什么", "哪里不一样",
            )
            if "配方" in text and any(marker in text for marker in diff_markers):
                return "formula_difference"
            if any(marker in text for marker in ("工艺", "流程", "加工")) and any(
                marker in text for marker in diff_markers
            ):
                return "process_difference"

        if len(samples) == 1 and any(marker in text for marker in (
            "完整信息", "完整资料", "完整看看", "全面看看", "全部信息",
            "所有信息", "具体信息", "详细信息", "具体资料", "详细资料", "完整研发上下文",
        )):
            return "sample_full_profile"
        return None

    @classmethod
    def deterministic_material_followup_decision(
        cls,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> IntentDecision | None:
        """Resolve explicit pair follow-ups before unrelated dataset routers.

        This deliberately handles only a referential pair whose identifiers are
        deterministically recoverable from user history. Other semantic routing
        remains the DeepSeek router's responsibility.
        """
        hints = build_conversation_hints(message, history or [])
        if (
            not hints.current_pair_referential
            or len(hints.active_comparison_identifiers) < 2
        ):
            return None
        intent = cls._deterministic_material_intent(message, hints)
        if intent not in {
            "formula_difference",
            "process_difference",
            "comparability_check",
        }:
            return None
        args: dict[str, Any] = {
            "left_identifier": hints.active_comparison_identifiers[0],
            "right_identifier": hints.active_comparison_identifiers[1],
        }
        if intent == "comparability_check" and hints.current_metrics:
            args["target_metric"] = hints.current_metrics[-1]
        return IntentDecision(intent, "compare_samples", args)

    @staticmethod
    def _normalize_mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _normalize_context_reference(
        raw: Any,
        hints: ConversationHints,
        has_attachments: bool,
    ) -> dict[str, Any]:
        result = dict(raw) if isinstance(raw, dict) else {}
        action = str(result.get("action") or "").strip()
        if not action or action == "new_request":
            action = hints.action_hint
        result["action"] = action
        result["use_previous_sample"] = bool(result.get("use_previous_sample", False))
        result["use_previous_scope"] = bool(result.get("use_previous_scope", False))
        result["use_previous_task"] = bool(result.get("use_previous_task", False))
        result["use_current_attachment"] = bool(
            result.get("use_current_attachment", False)
        )
        if hints.scope_only_followup or hints.scope_reset_followup:
            result["use_previous_task"] = bool(hints.active_history_task)
            result["use_previous_scope"] = True
        if action in {"follow_up_reference", "continue_previous", "refine_previous", "user_correction"}:
            if not hints.current_sample_identifiers and hints.active_sample_identifier:
                result["use_previous_sample"] = True
        if has_attachments and action == "follow_up_reference":
            result["use_current_attachment"] = True
        return result

    @staticmethod
    def _apply_context_to_args(
        *,
        intent: str,
        args: dict[str, Any],
        hints: ConversationHints,
        context_reference: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        result = dict(args)

        # A direct correction is stronger than history and must replace the old
        # referent if it appears in the argument set.
        correction = hints.correction
        if correction:
            old = correction["from"]
            new = correction["to"]
            for key in ("identifier", "left_identifier", "right_identifier"):
                if str(result.get(key, "")) == old:
                    result[key] = new
            if intent in {
                "get_sample_context",
                "get_formula",
                "get_process",
                "get_performance",
                "sample_full_profile",
                "analyze_cause",
                "sample_historical_similarity",
                "similar_samples",
            }:
                result["identifier"] = new

        current_samples = list(hints.current_sample_identifiers)
        recent_samples = list(hints.recent_sample_identifiers)

        if intent in {
            "get_sample_context",
            "get_formula",
            "get_process",
            "get_performance",
            "sample_full_profile",
            "analyze_cause",
            "similar_samples",
        }:
            # Explicit user sample is authoritative over any model extraction.
            if len(current_samples) == 1:
                explicit_identifier = current_samples[0]
                existing_identifier = result.get("identifier")
                # Preserve an already-correct model value (including its type,
                # e.g. integer 3073) while overriding any conflicting value.
                if (
                    existing_identifier is None
                    or str(existing_identifier).strip() != explicit_identifier
                ):
                    result["identifier"] = explicit_identifier
            elif hints.task_refinement_intent == intent:
                # For a pure view refinement with no explicit sample, discard
                # any LLM-invented text fragment (observed: “看性能么”).
                result.pop("identifier", None)
                if hints.active_sample_identifier:
                    result["identifier"] = hints.active_sample_identifier
            elif not str(result.get("identifier") or "").strip():
                if context_reference.get("use_previous_sample") and hints.active_sample_identifier:
                    result["identifier"] = hints.active_sample_identifier

        if intent == "sample_historical_similarity":
            if not str(result.get("identifier") or "").strip():
                if len(current_samples) == 1:
                    result["identifier"] = current_samples[0]
                elif (hints.scope_only_followup or hints.scope_reset_followup) and len(hints.active_history_sample_identifiers) == 1:
                    result["identifier"] = hints.active_history_sample_identifiers[0]
                elif hints.current_history_referential and hints.active_sample_identifier:
                    result["identifier"] = hints.active_sample_identifier
                elif context_reference.get("use_previous_task") and len(hints.active_history_sample_identifiers) == 1:
                    result["identifier"] = hints.active_history_sample_identifiers[0]
            if not str(result.get("target_metric") or "").strip():
                metric = (
                    (hints.current_metrics[-1] if hints.current_metrics else None)
                    or hints.active_history_metric
                    or hints.active_metric
                )
                if metric:
                    result["target_metric"] = metric

        if intent in {
            "compare_samples",
            "formula_difference",
            "process_difference",
            "comparability_check",
            "analyze_performance_difference",
            "joint_mysql_knowledge_analysis",
        }:
            if len(current_samples) >= 2:
                for key, explicit_identifier in (
                    ("left_identifier", current_samples[0]),
                    ("right_identifier", current_samples[1]),
                ):
                    existing_identifier = result.get(key)
                    if (
                        existing_identifier is None
                        or str(existing_identifier).strip() != explicit_identifier
                    ):
                        result[key] = explicit_identifier
            elif (
                hints.current_pair_referential
                and len(hints.active_comparison_identifiers) >= 2
            ):
                # The latest explicit user pair is authoritative over model
                # guesses for “这两个样品/两者/它们”.
                result["left_identifier"] = hints.active_comparison_identifiers[0]
                result["right_identifier"] = hints.active_comparison_identifiers[1]
            elif (hints.scope_only_followup or hints.scope_reset_followup) and len(hints.active_history_sample_identifiers) >= 2:
                result.setdefault("left_identifier", hints.active_history_sample_identifiers[0])
                result.setdefault("right_identifier", hints.active_history_sample_identifiers[1])
            elif context_reference.get("use_previous_sample"):
                combined = []
                for item in current_samples + recent_samples:
                    if item not in combined:
                        combined.append(item)
                if not str(result.get("left_identifier") or "").strip() and combined:
                    result["left_identifier"] = combined[0]
                if not str(result.get("right_identifier") or "").strip() and len(combined) >= 2:
                    result["right_identifier"] = combined[1]

        if intent in {
            "comparability_check",
            "analyze_performance_difference",
            "joint_mysql_knowledge_analysis",
        } and not str(result.get("target_metric") or "").strip():
            current_metrics = list(hints.current_metrics)
            recent_metrics = list(hints.recent_metrics)
            if len(current_metrics) == 1:
                result["target_metric"] = current_metrics[0]
            elif (hints.scope_only_followup or hints.scope_reset_followup) and hints.active_history_metric:
                result["target_metric"] = hints.active_history_metric
            elif context_reference.get("use_previous_sample") and hints.active_metric:
                result["target_metric"] = hints.active_metric

        if intent in {
            "search_historical_knowledge",
            "sample_historical_similarity",
            "historical_similar_case",
        } and hints.effective_history_query:
            result["history_query"] = hints.effective_history_query

        if intent == "performance_rank":
            text = str(message or "")
            if not str(result.get("target_metric") or "").strip():
                metric = RuleIntentRouter.extract_rank_metric(text)
                if metric:
                    result["target_metric"] = metric
            top_match = re.search(r"(?:前|top)\s*(\d+)", text, re.I)
            result.setdefault("top_n", int(top_match.group(1)) if top_match else 10)
            result.setdefault("order", "asc" if "最低" in text else "desc")
            result.setdefault("keyword", "")

        if intent == "performance_statistics":
            text = str(message or "")
            deterministic = RuleIntentRouter._route_performance_statistics(text)
            if deterministic is not None:
                result.update(deterministic.tool_args)
            else:
                result.setdefault("requested_statistics", ["mean"])
                result.setdefault("keyword", "")

        if intent == "similar_samples":
            text = str(message or "")
            deterministic = RuleIntentRouter._route_similar_samples(text)
            if deterministic is not None:
                # Explicit user identifier and requested similarity scope are
                # authoritative over model guesses.
                result.update(deterministic.tool_args)
            else:
                result.setdefault("similarity_scope", "combined")
                result.setdefault("top_n", 5)
                result.setdefault("keyword", "")

        if intent == "experiment_series_analysis" and not str(result.get("keyword") or "").strip():
            text = str(message or "")
            series = re.search(r"([A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*)", text)
            if series:
                result["keyword"] = series.group(1)

        if intent == "data_quality_check":
            result.setdefault("keyword", "")

        return result

    @staticmethod
    def _normalize_multi_condition_args(
        args: dict[str, Any],
        message: str,
        field_catalog: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
        normalized, errors = normalize_filters(args.get("filters"))
        result: dict[str, Any] = {
            "filters": [] if errors else normalized,
            "logic": normalize_logic(args.get("logic")),
        }
        if not errors:
            for spec in result["filters"]:
                if spec.get("unit") and not unit_is_explicit_in_text(
                    spec["unit"], message
                ):
                    # The model is not allowed to infer a unit from material
                    # knowledge. An omitted user unit remains omitted so the
                    # deterministic executor can detect mixed-unit ambiguity.
                    spec.pop("unit", None)

                if spec["section"] == "sample" and spec["field"] == "project_id":
                    expected_values = (
                        spec.get("values")
                        if spec.get("operator") in {"between", "in"}
                        else [spec.get("value")]
                    )
                    for expected in expected_values or []:
                        pattern = (
                            rf"(?:project|项目)\s*(?:id|编号|号)?\s*"
                            rf"(?:为|是|=|：|:|#)?\s*{re.escape(str(expected))}(?!\d)"
                        )
                        if not re.search(pattern, message, re.I):
                            errors.append("项目筛选值必须由用户在当前问题中明确给出")
                            break

        bindings: list[dict[str, Any]] = []
        if not errors and field_catalog:
            bound_filters, bindings, binding_errors = bind_filters_to_catalog(
                result["filters"],
                field_catalog,
            )
            if binding_errors:
                for item in binding_errors:
                    candidates = "、".join(
                        f"{candidate.get('section')}.{candidate.get('field')}"
                        for candidate in item.get("candidates") or []
                    )
                    if item.get("code") == "ambiguous_field":
                        errors.append(
                            f"字段“{item.get('requested_field')}”存在多个候选"
                            + (f"：{candidates}" if candidates else "")
                        )
                    else:
                        errors.append(
                            f"授权字段目录中不存在“{item.get('requested_field')}”"
                        )
            else:
                result["filters"] = bound_filters

        if errors:
            result["filters"] = []

        keyword = str(args.get("keyword") or "").strip()[:120]
        if (
            keyword
            and not keyword.isdigit()
            and keyword.casefold() in str(message or "").casefold()
        ):
            result["keyword"] = keyword
        else:
            result["keyword"] = ""
        for key, default, upper in (
            ("result_limit", 50, 100),
            ("scan_limit", 500, 500),
        ):
            try:
                value = int(args.get(key, default))
            except (TypeError, ValueError):
                value = default
            result[key] = max(1, min(value, upper))
        return result, errors, bindings

    @staticmethod
    def _sanitize_tool_args(
        *,
        intent: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Strip LLM-only metadata before it can reach executable tools.

        DeepSeek may harmlessly attach fields such as ``target_metric`` to a
        generic compare request.  The database tools intentionally use strict
        Python signatures, so forwarding those extras would raise TypeError.
        This is a second safety boundary in addition to Skill-level filtering.
        """
        allowed_by_intent: dict[str, set[str]] = {
            "get_sample_context": {"identifier"},
            "get_formula": {"identifier"},
            "get_process": {"identifier"},
            "get_performance": {"identifier"},
            "find_samples": {"keyword", "limit"},
            "compare_samples": {"left_identifier", "right_identifier"},
            "sample_full_profile": {"identifier"},
            "formula_difference": {"left_identifier", "right_identifier"},
            "process_difference": {"left_identifier", "right_identifier"},
            "comparability_check": {
                "left_identifier",
                "right_identifier",
                "target_metric",
            },
            "performance_rank": {"target_metric", "keyword", "top_n", "order", "scan_limit"},
            "performance_statistics": {
                "target_metric",
                "requested_statistics",
                "keyword",
                "scan_limit",
            },
            "experiment_series_analysis": {"keyword", "scan_limit"},
            "data_quality_check": {"keyword", "scan_limit"},
            "similar_samples": {
                "identifier",
                "similarity_scope",
                "top_n",
                "keyword",
                "scan_limit",
            },
            "find_samples_multi_condition": {
                "filters",
                "logic",
                "keyword",
                "result_limit",
                "scan_limit",
            },
            "analyze_cause": {"identifier"},
            "analyze_performance_difference": {
                "left_identifier",
                "right_identifier",
                "target_metric",
                "direction",
                "direction_claim",
            },
            "search_historical_knowledge": {
                "project_id",
                "history_query",
            },
            "historical_similar_case": {"project_id", "history_query"},
            "sample_historical_similarity": {
                "identifier",
                "target_metric",
                "project_id",
                "history_query",
            },
            "joint_mysql_knowledge_analysis": {
                "left_identifier",
                "right_identifier",
                "target_metric",
                "direction",
                "direction_claim",
                "project_id",
            },
            "database_explorer": set(),
            "general_conversation": set(),
            "unsupported_future_feature": set(),
        }
        allowed = allowed_by_intent.get(intent)
        if allowed is None:
            return dict(args)
        return {key: value for key, value in args.items() if key in allowed}

    @staticmethod
    def _sanitize_project_scope_args(
        *,
        intent: str,
        args: dict[str, Any],
        hints: ConversationHints,
        context_reference: dict[str, Any],
    ) -> dict[str, Any]:
        if intent not in {
            "search_historical_knowledge",
            "historical_similar_case",
            "sample_historical_similarity",
            "joint_mysql_knowledge_analysis",
        }:
            return args

        result = dict(args)
        current_projects = list(hints.current_project_ids)
        recent_projects = list(hints.recent_project_ids)

        if hints.scope_reset_followup:
            result.pop("project_id", None)
            return result

        if current_projects:
            result["project_id"] = current_projects[-1]
            return result

        raw = result.get("project_id")
        if raw is None or str(raw).strip() == "":
            result.pop("project_id", None)
            return result

        # Allow reuse only when the turn explicitly says to keep previous scope.
        validated_previous_projects = list(hints.active_history_project_ids) or recent_projects
        if context_reference.get("use_previous_scope") and validated_previous_projects:
            try:
                candidate = int(raw)
            except (TypeError, ValueError):
                result.pop("project_id", None)
                return result
            if candidate in validated_previous_projects:
                result["project_id"] = candidate
                return result

        # Otherwise an unmentioned project_id is considered an LLM invention.
        result.pop("project_id", None)
        return result

    @staticmethod
    def _normalize_secondary_intents(raw: Any, action_hint: str) -> tuple[str, ...]:
        values: list[str] = []
        if isinstance(raw, list):
            for item in raw[:8]:
                text = str(item or "").strip()
                if text and text not in values:
                    values.append(text[:80])
        if action_hint != "new_request" and action_hint not in values:
            values.append(action_hint)
        return tuple(values)

    @staticmethod
    def _enrich_entities(
        entities: dict[str, Any],
        args: dict[str, Any],
        hints: ConversationHints,
    ) -> dict[str, Any]:
        result = dict(entities)
        sample_ids = []
        for key in ("identifier", "left_identifier", "right_identifier"):
            value = args.get(key)
            if value is None or str(value).strip() == "":
                continue
            if value not in sample_ids:
                sample_ids.append(value)
        if sample_ids:
            result["sample_identifiers"] = sample_ids
        metric = args.get("target_metric")
        if metric is not None and str(metric).strip():
            result["metrics"] = [str(metric).strip()]
        elif hints.current_metrics:
            result.setdefault("metrics", list(hints.current_metrics))
        if args.get("project_id") is not None:
            result["project_ids"] = [args["project_id"]]
        return result

    @staticmethod
    def _normalize_scope(
        raw: Any,
        args: dict[str, Any],
        hints: ConversationHints,
        context_reference: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(raw) if isinstance(raw, dict) else {}
        # Router scope is descriptive only. Server-side UserContext remains the
        # authority and will validate every explicit project.
        result["company"] = "current"
        if args.get("project_id") is not None:
            result["projects"] = [args["project_id"]]
            result["source"] = "explicit_or_validated_previous_scope"
        else:
            result["projects"] = "all_authorized"
            result["source"] = "authorized_default"
        return result

    @staticmethod
    def _missing_required_args(intent: str, args: dict[str, Any]) -> list[str]:
        required = _REQUIRED_ARGS.get(intent, ())
        missing = []
        for key in required:
            value = args.get(key)
            if value is None or str(value).strip() == "":
                missing.append(key)
            elif isinstance(value, (list, tuple, dict, set)) and not value:
                missing.append(key)
        return missing

    @staticmethod
    def _clarification_question(intent: str, missing: list[str]) -> str:
        if intent in {
            "get_sample_context",
            "get_formula",
            "get_process",
            "get_performance",
            "sample_full_profile",
            "analyze_cause",
            "sample_historical_similarity",
            "similar_samples",
        }:
            return "你想分析哪个样品？请提供样品 ID 或样品名称。"
        if intent in {
            "compare_samples",
            "formula_difference",
            "process_difference",
            "comparability_check",
            "analyze_performance_difference",
            "joint_mysql_knowledge_analysis",
        }:
            if "target_metric" in missing and not {
                "left_identifier",
                "right_identifier",
            }.intersection(missing):
                return "你想比较哪个性能指标？例如冲击强度、MFR、拉伸强度等。"
            return "请告诉我需要比较的两个样品（样品 ID 或名称）；如果是性能原因分析，也请说明目标性能指标。"
        if intent == "find_samples":
            return "你想按什么条件找样品？可以给材料、样品名关键词、配方或性能条件。"
        if intent == "find_samples_multi_condition":
            return (
                "请明确筛选字段、比较关系和值，例如“冲击强度大于40，"
                "并且成本低于30”；如果单位会影响判断，也请写出单位。"
            )
        if intent == "similar_samples":
            return "你想以哪个样品作为参照？请提供样品 ID 或样品名称。"
        return "当前信息还不足以确定要执行的材料研发操作，请补充样品、指标或分析范围。"

    @staticmethod
    def _derive_safe_tool_plan(
        intent: str,
        tool_name: str | None,
        args: dict[str, Any],
    ) -> tuple[IntentToolPlanStep, ...]:
        if intent in _EXPECTED_TOOL_BY_INTENT and tool_name:
            return (
                IntentToolPlanStep(
                    kind="tool",
                    name=tool_name,
                    args=dict(args),
                    purpose="执行当前只读材料数据库操作",
                ),
            )
        if intent in {"analyze_current_attachment", "ask_current_attachment"}:
            return (
                IntentToolPlanStep(
                    kind="skill",
                    name="current_attachment",
                    args={},
                    purpose="分析当前 Chat 临时附件",
                ),
            )
        if intent == "search_historical_knowledge":
            return (
                IntentToolPlanStep(
                    kind="skill",
                    name="historical_knowledge",
                    args={
                        "project_id": args.get("project_id")
                    } if args.get("project_id") is not None else {},
                    purpose="检索当前 Company 授权范围内的长期历史知识",
                ),
            )
        if intent == "historical_similar_case":
            return (
                IntentToolPlanStep(
                    kind="skill",
                    name="historical_knowledge",
                    args={
                        "project_id": args.get("project_id")
                    } if args.get("project_id") is not None else {},
                    purpose="检索当前 Company 授权范围内的相似历史案例",
                ),
            )
        if intent == "database_explorer":
            return (
                IntentToolPlanStep(
                    kind="skill",
                    name="database_explorer",
                    args={},
                    purpose=(
                        "对未匹配高精度意图的业务数据库问题执行受控只读探索"
                    ),
                ),
            )
        if intent == "sample_historical_similarity":
            history_args = (
                {"project_id": args["project_id"]}
                if args.get("project_id") is not None
                else {}
            )
            return (
                IntentToolPlanStep(
                    kind="tool",
                    name="get_sample_context",
                    args={"identifier": args.get("identifier")},
                    purpose="取得当前样品数据库事实",
                ),
                IntentToolPlanStep(
                    kind="skill",
                    name="historical_knowledge",
                    args=history_args,
                    purpose="在当前 Company 授权范围检索相似历史案例",
                ),
                IntentToolPlanStep(
                    kind="skill",
                    name="sample_historical_similarity",
                    args=dict(args),
                    purpose="联合当前样品事实与历史证据判断相似点",
                ),
            )
        if intent == "joint_mysql_knowledge_analysis":
            compare_args = {
                key: args[key]
                for key in ("left_identifier", "right_identifier")
                if key in args
            }
            history_args = (
                {"project_id": args["project_id"]}
                if args.get("project_id") is not None
                else {}
            )
            return (
                IntentToolPlanStep(
                    kind="tool",
                    name="compare_samples",
                    args=compare_args,
                    purpose="取得两个样品的数据库事实",
                ),
                IntentToolPlanStep(
                    kind="skill",
                    name="historical_knowledge",
                    args=history_args,
                    purpose="检索相似历史案例",
                ),
                IntentToolPlanStep(
                    kind="skill",
                    name="joint_mysql_knowledge",
                    args=dict(args),
                    purpose="联合数据库事实与历史证据",
                ),
            )
        return ()
