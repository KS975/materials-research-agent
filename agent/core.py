from __future__ import annotations

import json
from typing import Any

from agent.router import LLMIntentRouter, RuleIntentRouter
from agent.tool_registry import ToolRegistry
from llm.base import LLMProvider
from schemas.user_context import UserContext
from skills.analysis import AnalysisSkill
from skills.comparison import ComparisonSkill
from skills.data_query import DataQuerySkill


class AgentCore:
    def __init__(
        self,
        registry: ToolRegistry,
        llm: LLMProvider,
        llm_enabled: bool,
    ):
        self.registry = registry
        self.llm = llm
        self.llm_enabled = llm_enabled
        self.rule_router = RuleIntentRouter()
        self.llm_router = LLMIntentRouter(llm)
        self.skills = [
            DataQuerySkill(registry),
            ComparisonSkill(registry),
            AnalysisSkill(registry),
        ]

    def route(self, message: str):
        decision = self.rule_router.route(message)
        if decision is not None:
            return decision
        if self.llm_enabled:
            return self.llm_router.route(message)
        raise ValueError(
            "无法用 V0.1.1 规则识别该问题；请启用 LLM 或使用样品查询/比较类表达"
        )

    def execute(
        self,
        intent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
    ):
        for skill in self.skills:
            if skill.can_handle(intent):
                return skill.execute(tool_name, tool_args, ctx)
        raise ValueError(f"没有 Skill 可以处理 intent={intent}")

    def answer(
        self,
        message: str,
        intent: str,
        tool_result: Any,
    ) -> str:
        if not self.llm_enabled:
            return self._deterministic_answer(intent, tool_result)

        system = """你是材料研发智能体 V0.1.1。
你只能根据给定 Tool Result 回答。
必须明确区分：
1. 数据库事实
2. 工程推断/假设
3. 证据不足
禁止把相关性写成因果。
禁止补全不存在的字段、单位或测试条件。
如果测试条件 status=missing_both，不得写成“测试条件相同”，只能写“双方测试条件均未记录”。
如果动态字段 resolved=false，保留 raw_key 并说明未解析。
对于 performance_difference，优先按照 facts / hypotheses / evidence_gaps / conclusion_limit 组织答案。
回答使用中文。
"""
        payload = json.dumps(tool_result, ensure_ascii=False, default=str)
        user = f"用户问题：{message}\n\nTool Result（真实数据库/工具证据）：\n{payload}"
        return self.llm.complete(system, user)

    @staticmethod
    def _deterministic_answer(intent: str, result: Any) -> str:
        if not isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, default=str, indent=2)

        status = result.get("status")
        if status == "not_found":
            return f"未在当前公司/项目权限范围内找到样品：{result.get('identifier')}"
        if status == "ambiguous":
            candidates = result.get("candidates", [])
            return "样品名称存在多条记录，请指定样品 ID：\n" + "\n".join(
                f"- id={x.get('id')}，name={x.get('name')}，project_id={x.get('project_id')}"
                for x in candidates
            )
        if status == "target_metric_not_found":
            available = "、".join(result.get("available_changed_performance", [])) or "无"
            return (
                f"没有在两个样品的性能差异中找到指标“{result.get('target_metric')}”。"
                f"可用的变化性能指标：{available}。"
            )

        if intent == "analyze_performance_difference" and status == "ok":
            facts = result.get("facts", {})
            target = facts.get("target_performance", {})
            numeric = facts.get("numeric_difference") or {}
            left_sample = facts.get("left_sample") or {}
            right_sample = facts.get("right_sample") or {}
            condition_info = facts.get("test_conditions") or {}

            lines = [
                "【数据库事实】",
                (
                    f"{left_sample.get('id')}（{left_sample.get('name')}）的"
                    f"{result.get('target_metric')}为 {target.get('left')} {target.get('unit') or ''}；"
                    f"{right_sample.get('id')}（{right_sample.get('name')}）为 "
                    f"{target.get('right')} {target.get('unit') or ''}。"
                ).strip(),
            ]
            if numeric:
                lines.append(
                    f"左样品减右样品 = {numeric.get('left_minus_right')}；"
                    f"相对右样品变化 = {numeric.get('relative_to_right_percent')}%。"
                )
            lines.append(
                f"检测到 {len(facts.get('formula_changes', []))} 个配方字段变化、"
                f"{len(facts.get('process_changes', []))} 个工艺字段变化。"
            )

            if condition_info.get("status") == "missing_both":
                lines.append("双方测试条件均未记录，因此不能确认测试条件一致。")
            elif condition_info.get("same") is True:
                lines.append("数据库中记录的测试条件一致。")
            elif condition_info.get("same") is False:
                lines.append("数据库中记录的测试条件存在差异。")

            lines.append("\n【工程推断 / 假设】")
            hypotheses = result.get("hypotheses", [])
            if hypotheses:
                for item in hypotheses:
                    lines.append(f"- {item.get('statement')} 依据：{item.get('basis')}")
            else:
                lines.append("- 当前数据不足以形成有依据的工程假设。")

            lines.append("\n【证据缺口】")
            for gap in result.get("evidence_gaps", []):
                lines.append(f"- {gap}")

            lines.append("\n【结论边界】")
            lines.append(str(result.get("conclusion_limit") or "当前证据不能确认因果关系。"))
            return "\n".join(lines)

        if intent == "analyze_cause":
            return (
                "已取得该样品的数据库事实，但当前不能仅凭单一样品把性能变化认定为因果。"
                "请提供一个可比样品，或使用“为什么 3811 的冲击强度比 3809 低？”这类表达。"
                "\n\n数据库事实：\n"
                + json.dumps(result, ensure_ascii=False, default=str, indent=2)
            )

        return json.dumps(result, ensure_ascii=False, default=str, indent=2)
