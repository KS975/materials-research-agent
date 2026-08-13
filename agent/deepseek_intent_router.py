from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm.base import LLMProvider


@dataclass(frozen=True, slots=True)
class DeepSeekIntentDecision:
    intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    reasoning_summary: str = ""


_ALLOWED_TOOLS = {
    "get_sample_context", "get_formula", "get_process",
    "get_performance", "compare_samples", "find_samples",
}
_ALLOWED_INTENTS = _ALLOWED_TOOLS | {
    "analyze_performance_difference",
    "analyze_current_attachment",
    "ask_current_attachment",
    "search_historical_knowledge",
    "joint_mysql_knowledge_analysis",
    "unsupported_future_feature",
}


class DeepSeekIntentRouter:
    """DeepSeek only extracts intent/arguments. It never generates SQL."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def route(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> DeepSeekIntentDecision:
        system_prompt = """
你是“材数智能体”的意图路由器，只做意图分类和参数提取，不回答业务问题，不生成 SQL。

V0.1.1 数据库 Tool：
- get_sample_context(identifier)
- get_formula(identifier)
- get_process(identifier)
- get_performance(identifier)
- compare_samples(left_identifier, right_identifier)
- find_samples(keyword)

性能差异原因：
intent=analyze_performance_difference，tool_name=compare_samples，参数包含
left_identifier、right_identifier、target_metric、direction_claim。

V0.1.2-A 当前已开放“当前 Chat 临时附件”：
- 用户已经上传附件，要求“分析/总结这份报告、这个文件、附件”时：
  intent=analyze_current_attachment，tool_name=null。
- 用户已经上传附件，针对附件问具体问题时：
  intent=ask_current_attachment，tool_name=null。
- 当前附件不会写 Qdrant，不属于长期知识库。

V0.1.2-B 当前已开放“历史 Knowledge Index / Qdrant RAG”：
- 用户询问“历史有没有类似问题”“以前有没有类似情况”“历史报告里有没有...”
  “项目历史资料里...”等只需要检索长期历史资料的问题：
  intent=search_historical_knowledge，tool_name=null。
- 如果用户明确写出项目号，可在 tool_args 中给出 {"project_id": 115}；
  未明确项目时不要编造 project_id，返回空 tool_args 即可。
- 历史 Knowledge Index 与“当前 Chat 临时附件”是两个不同证据源。

V0.1.2 T07 当前已开放“MySQL + 历史报告联合分析”：
- 当用户明确要求同时结合数据库样品事实与历史资料/历史报告进行分析，例如：
  “3811 的冲击强度比 3809 低很多，历史上有没有类似问题？结合数据库和历史报告分析一下。”
  使用：
  intent=joint_mysql_knowledge_analysis，tool_name=null。
- tool_args 必须提取：
  left_identifier、right_identifier、target_metric、direction_claim；
  用户明确项目号时再提取 project_id。
- 示例：
  {"left_identifier":3811,"right_identifier":3809,
   "target_metric":"冲击强度","direction_claim":"更低","project_id":115}
- T07 仍然不允许 LLM 生成 SQL；数据库事实由后端既有 compare_samples Tool/Repository 获取。

仍未开放：
- V0.1.3 Dataset/ML
- V0.1.4 Optimization/BO
这些请求：intent=unsupported_future_feature，tool_name=null。

规则：
- 有附件时，若用户明确在问“这份/这个/附件/报告/文档”的内容，优先走 current_attachment intent。
- “历史/以前/过去/历史报告/历史资料/有没有类似”且只要求长期资料检索时，走 search_historical_knowledge。
- 同一句话明确同时要求“数据库/样品对比”与“历史资料/历史报告”联合分析时，
  必须走 joint_mysql_knowledge_analysis，不能只选其中一个数据源。
- 明确询问样品数据库字段，例如“3811 的冲击强度是多少”，仍走 V0.1.1 数据库 Tool。
- 无附件时，不得假装已经有附件。
- 允许参考 conversation_history 解析“这个样品/另一个/它”等指代。
- 不确定样品时不要编造 ID，可以使用 find_samples。
- Tool 只能来自上面的 V0.1.1 白名单。

只输出 JSON：
{"intent":"...","tool_name":"...或null","tool_args":{},"reasoning_summary":"一句简短可展示说明"}
""".strip()
        payload = {
            "conversation_history": (history or [])[-12:],
            "current_message": message,
            "current_chat_attachments": attachments or [],
        }
        raw = self.llm.complete(system_prompt, json.dumps(payload, ensure_ascii=False)).strip()
        fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.I | re.S)
        if fence:
            raw = fence.group(1)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek 意图结果不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("DeepSeek 意图结果必须是 JSON 对象")

        intent = str(data.get("intent") or "").strip()
        tool_name = data.get("tool_name")
        tool_name = None if tool_name is None else str(tool_name).strip()
        args = data.get("tool_args") or {}

        if intent not in _ALLOWED_INTENTS:
            raise ValueError(f"未允许的 intent: {intent}")
        if tool_name is not None and tool_name not in _ALLOWED_TOOLS:
            raise ValueError(f"未允许的 tool_name: {tool_name}")
        if not isinstance(args, dict):
            raise ValueError("tool_args 必须是对象")
        if intent in {
            "analyze_current_attachment",
            "ask_current_attachment",
            "search_historical_knowledge",
            "joint_mysql_knowledge_analysis",
            "unsupported_future_feature",
        }:
            if tool_name is not None:
                raise ValueError(f"{intent} 不允许绑定数据库 Tool")
        elif tool_name is None:
            raise ValueError("可执行数据库意图必须绑定 Tool")

        return DeepSeekIntentDecision(
            intent=intent,
            tool_name=tool_name,
            tool_args=args,
            reasoning_summary=str(data.get("reasoning_summary") or "").strip(),
        )
