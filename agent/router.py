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
    _explicit_numeric_sample = re.compile(r"(?<![A-Za-z0-9_.-])(?P<id>\d{3,})(?![A-Za-z0-9_.-])")
    _explicit_named_sample = re.compile(r"(?<![A-Za-z0-9_.-])(?P<name>[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*)(?![A-Za-z0-9_.-])")
    _diff_markers = (
        "差在哪", "差异", "区别", "不同", "改了什么", "变化",
        "变了什么", "哪里不一样", "有何不同", "有什么不同",
    )
    _comparability_markers = (
        "能直接比较", "可以直接比较", "能不能直接比较", "能不能比较",
        "可以比较吗", "是否可比", "可比吗", "可比性", "适合比较",
    )
    _pair_referential_markers = (
        "这两个样品", "这两个样本", "这两个", "两个样品", "两个样本",
        "两者", "二者", "它们", "刚才比较的两个", "前面比较的两个",
    )
    _rank_markers = ("最好", "最高", "最低", "排序", "排名", "前几", "top")
    _rank_request_prefix = re.compile(
        r"^(?:(?:请|麻烦)\s*)?"
        r"(?:给我|请给我|帮我找|帮我查|帮我|查一下|查询一下|"
        r"看看|看一下|列出|列一下|统计一下)\s*"
    )

    def route(self, message: str) -> IntentDecision | None:
        text = message.strip()

        if "样品" in text and any(marker in text.lower() for marker in self._rank_markers):
            metric = self.extract_rank_metric(text)
            top_match = re.search(r"(?:前|top)\s*(\d+)", text, re.I)
            if metric:
                return IntentDecision(
                    "performance_rank",
                    "list_samples_for_analysis",
                    {
                        "target_metric": metric,
                        "top_n": int(top_match.group(1)) if top_match else 10,
                        "order": "asc" if "最低" in text else "desc",
                        "keyword": "",
                    },
                )

        if any(marker in text for marker in (
            "这一组在研究什么", "这组在研究什么",
            "这一组实验在研究什么", "这组实验在研究什么",
            "实验系列", "这一系列",
        )):
            series = re.search(r"([A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*)", text)
            if series:
                return IntentDecision(
                    "experiment_series_analysis",
                    "list_samples_for_analysis",
                    {"keyword": series.group(1)},
                )

        if "数据" in text and any(marker in text for marker in (
            "有没有问题", "质量检查", "缺失", "异常", "重复",
        )):
            return IntentDecision(
                "data_quality_check",
                "list_samples_for_analysis",
                {"keyword": ""},
            )

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

        # Two-sample material questions must be specialized before the
        # legacy single-field rules below. Otherwise text such as
        # “3811和3809的配方差在哪” is incorrectly collapsed into one sample
        # identifier (“3811和3809的”).
        pair_decision = self._route_explicit_pair_question(text)
        if pair_decision is not None:
            return pair_decision

        # A stateless fallback router cannot resolve “这两个样品” safely.
        # Return no decision so the context-aware router can restore the latest
        # explicit pair, or ask for clarification when no pair exists. Never
        # query a literal sample identifier such as “这两个样品的”.
        if any(marker in text for marker in self._pair_referential_markers):
            return None

        match = self._compare.search(text)
        if match:
            pair = self._extract_explicit_sample_pair(text)
            left = pair[0] if pair else match.group("a").strip()
            right = pair[1] if pair else match.group("b").strip()
            return IntentDecision(
                "compare_samples",
                "compare_samples",
                {
                    "left_identifier": left,
                    "right_identifier": right,
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
            # Rule routing is only a fallback. Never send a whole natural-language
            # phrase such as “看样品3811具体信息” to the database as a sample name.
            # Prefer an explicit numeric/sample token from the user's text.
            sample = self._extract_explicit_sample_identifier(text)
            if not sample:
                sample = match.group("sample").strip()
                sample = re.sub(
                    r"^(?:样品|样本)\s*", "", sample
                ).strip()
                sample = re.sub(
                    r"(?:的)?(?:完整研发上下文|完整信息|具体信息|详细信息|研发上下文|信息|资料)$",
                    "",
                    sample,
                ).strip()
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

    @classmethod
    def extract_rank_metric(cls, text: str) -> str:
        """Extract only the performance name from a natural rank request."""
        value = str(text or "").strip()
        previous = None
        while value != previous:
            previous = value
            value = cls._rank_request_prefix.sub("", value).strip()
        value = re.sub(
            r"^(?:哪些|哪几个|所有|把|请把)?\s*(?:样品|样本)(?:的)?\s*",
            "",
            value,
        )
        value = re.sub(
            r"(?:最好|最高|最低|排序|排名|前\s*\d+|top\s*\d+).*$",
            "",
            value,
            flags=re.I,
        )
        return value.strip("的 ，,。？?")

    @classmethod
    def _extract_explicit_sample_identifiers(cls, text: str) -> list[str]:
        value = str(text or "")
        found: list[tuple[int, str]] = []
        for match in cls._explicit_numeric_sample.finditer(value):
            found.append((match.start(), match.group("id")))
        for match in cls._explicit_named_sample.finditer(value):
            found.append((match.start(), match.group("name")))
        found.sort(key=lambda item: item[0])

        result: list[str] = []
        for _, identifier in found:
            if identifier not in result:
                result.append(identifier)
        return result

    @classmethod
    def _extract_explicit_sample_identifier(cls, text: str) -> str | None:
        identifiers = cls._extract_explicit_sample_identifiers(text)
        return identifiers[0] if len(identifiers) == 1 else None

    @classmethod
    def _extract_explicit_sample_pair(cls, text: str) -> tuple[str, str] | None:
        identifiers = cls._extract_explicit_sample_identifiers(text)
        if len(identifiers) == 2:
            return identifiers[0], identifiers[1]
        return None

    @classmethod
    def _route_explicit_pair_question(cls, text: str) -> IntentDecision | None:
        pair = cls._extract_explicit_sample_pair(text)
        if pair is None:
            return None
        left, right = pair

        if any(marker in text for marker in cls._comparability_markers):
            args: dict[str, Any] = {
                "left_identifier": left,
                "right_identifier": right,
            }
            metric = cls._extract_pair_target_metric(text, left, right)
            if metric:
                args["target_metric"] = metric
            return IntentDecision(
                "comparability_check",
                "compare_samples",
                args,
            )

        # If two explicit samples are named together with a formula/process
        # field, the user is asking about the pair, not a sample literally
        # named “3811和3809”.  Difference wording is common but not required.
        if "配方" in text:
            return IntentDecision(
                "formula_difference",
                "compare_samples",
                {"left_identifier": left, "right_identifier": right},
            )

        if any(marker in text for marker in ("工艺", "流程", "加工")):
            return IntentDecision(
                "process_difference",
                "compare_samples",
                {"left_identifier": left, "right_identifier": right},
            )

        if any(marker in text for marker in cls._diff_markers):
            return IntentDecision(
                "compare_samples",
                "compare_samples",
                {"left_identifier": left, "right_identifier": right},
            )
        return None

    @classmethod
    def _extract_pair_target_metric(
        cls, text: str, left: str, right: str
    ) -> str | None:
        # Best-effort only. comparability_check can operate without a target
        # metric, so never invent one when the phrase cannot be cleanly parsed.
        pattern = re.compile(
            rf"{re.escape(left)}\s*(?:和|与|跟|、)\s*{re.escape(right)}"
            rf"\s*(?:的)?\s*(?P<metric>.*?)\s*"
            rf"(?:能直接比较|可以直接比较|能不能直接比较|能不能比较|"
            rf"可以比较吗|是否可比|可比吗|可比性|适合比较)"
        )
        match = pattern.search(text)
        if not match:
            return None
        metric = match.group("metric").strip(" 的，,。？?：: ")
        return metric or None

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
