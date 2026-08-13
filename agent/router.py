from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm.base import LLMProvider


@dataclass(frozen=True, slots=True)
class IntentDecision:
    intent: str
    tool_name: str
    tool_args: dict[str, Any]


class RuleIntentRouter:
    _compare = re.compile(
        r"(?:比较|对比)\s*(?P<a>.+?)\s*(?:和|与|跟|、)\s*(?P<b>.+?)(?:\s*(?:有什么)?(?:差异|区别|不同))?[？?。]?$"
    )
    _why_compare_ids = re.compile(
        r"为什么\s*(?P<a>\d+)\s*(?:的)?\s*(?P<metric>.+?)\s*比\s*(?P<b>\d+)\s*"
        r"(?P<direction>更低|更高|低|高|下降|上升|差|好)\s*[？?。]?$"
    )
    _why = re.compile(r"为什么\s*(?P<sample>[^\s，,。？?]+)")
    _query = re.compile(r"(?:查|查询|查看)\s*(?P<sample>.+?)(?:的)?(?:完整研发上下文|完整信息|研发上下文)?[？?。]?$")

    def route(self, message: str) -> IntentDecision | None:
        text = message.strip()

        match = self._why_compare_ids.search(text)
        if match:
            return IntentDecision(
                "analyze_performance_difference",
                "compare_samples",
                {
                    "left_identifier": match.group("a").strip(),
                    "right_identifier": match.group("b").strip(),
                    "target_metric": match.group("metric").strip(),
                    "direction": match.group("direction").strip(),
                },
            )

        match = self._compare.search(text)
        if match:
            return IntentDecision(
                "compare_samples",
                "compare_samples",
                {
                    "left_identifier": match.group("a").strip(),
                    "right_identifier": match.group("b").strip(),
                },
            )

        match = self._why.search(text)
        if match:
            return IntentDecision(
                "analyze_cause",
                "get_sample_context",
                {"identifier": match.group("sample").strip()},
            )

        if "配方" in text:
            sample = self._extract_after_query_verb(text) or self._extract_before_keyword(text, "配方")
            if sample:
                return IntentDecision("get_formula", "get_formula", {"identifier": sample})

        if "工艺" in text or "流程" in text:
            sample = self._extract_after_query_verb(text) or self._extract_before_keyword(text, "工艺")
            if sample:
                return IntentDecision("get_process", "get_process", {"identifier": sample})

        if "性能" in text:
            sample = self._extract_after_query_verb(text) or self._extract_before_keyword(text, "性能")
            if sample:
                return IntentDecision(
                    "get_performance",
                    "get_performance",
                    {"identifier": sample},
                )

        match = self._query.search(text)
        if match:
            sample = match.group("sample").strip()
            sample = re.sub(r"(?:的)?(?:完整研发上下文|完整信息|研发上下文)$", "", sample).strip()
            return IntentDecision(
                "get_sample_context",
                "get_sample_context",
                {"identifier": sample},
            )

        if text.startswith(("找", "搜索", "查找")):
            keyword = re.sub(r"^(找|搜索|查找)\s*", "", text).strip("？?。 ")
            return IntentDecision(
                "find_samples",
                "find_samples",
                {"keyword": keyword or ""},
            )

        return None

    @staticmethod
    def _extract_after_query_verb(text: str) -> str | None:
        match = re.search(r"(?:查|查询|查看)\s*([^\s，,。？?]+)", text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_before_keyword(text: str, keyword: str) -> str | None:
        before = text.split(keyword, 1)[0].strip()
        before = re.sub(r"^(查|查询|查看)\s*", "", before).strip()
        return before or None


class LLMIntentRouter:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def route(self, message: str) -> IntentDecision:
        system = """你是材料研发智能体的意图路由器。
只输出 JSON，不要输出 SQL。
允许的 tool_name 只有：
get_sample_context, get_formula, get_process, get_performance, compare_samples, find_samples。
注意：性能原因分析仍然使用 compare_samples，不新增生产数据库 Tool。
JSON 格式：
{"intent":"...", "tool_name":"...", "tool_args":{...}}
比较需要 left_identifier/right_identifier；性能差异分析还可包含 target_metric/direction；
其他样品查询需要 identifier；搜索需要 keyword。
"""
        raw = self.llm.complete(system, message)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM 路由结果不是合法 JSON") from exc
        return IntentDecision(
            intent=str(payload["intent"]),
            tool_name=str(payload["tool_name"]),
            tool_args=dict(payload.get("tool_args") or {}),
        )
