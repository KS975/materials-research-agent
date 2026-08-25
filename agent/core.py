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
from skills.material_intelligence import MaterialIntelligenceSkill


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
        self.material_intelligence_skill = MaterialIntelligenceSkill(registry)
        self.skills = [
            DataQuerySkill(registry),
            ComparisonSkill(registry),
            AnalysisSkill(registry),
            self.material_intelligence_skill,
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
                if isinstance(skill, MaterialIntelligenceSkill):
                    return skill.execute_intent(intent, tool_name, dict(tool_args), ctx)
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
对于 sample_full_profile / formula_difference / process_difference / comparability_check：
- 数值差值、相对变化、配方算术和只能直接引用 Tool Result 中的确定性计算，不得自行重新计算。
- comparability_check 的缺失测试条件必须表述为“无法确认一致”，不得当作“相同”。
- 只解释 Tool Result 已支持的事实，不根据常识补齐材料体系或测试标准。
对于 performance_rank / experiment_series_analysis / data_quality_check：
- 排序、计数、比例、缺失和异常标记只能引用 Tool Result，不得重新计算。
- experiment_series_analysis 必须严格区分 constant_fields、variable_fields 与 missing_fields；value 为 None/空值的字段不得写成常量。
- experiment_series_analysis 如果提供 purpose_inference，必须先说明数据库未显式记录目的，再引用其 summary 给出“实验设计推断”；不得因为缺少目的字段而完全拒绝回答，也不得把该推断写成已证实因果。
- 命中多个 project_id 时必须按 project_groups 分组说明，并引用 cross_project_assessment；不得把不同项目直接合并成重复实验。
- missing_fields 是证据缺口，只概括数量和关键字段，不要把大量 None 逐项列入“常量信息”。
- 缺失字段数量只能引用 field_summary.missing_field_count 或 project_groups[*].field_summary.missing_field_count，不得由模型自行数数组后写“约多少项”。
- 若结构化 formula 为空但工艺文本包含配比，只能写“未提供可解析的结构化配方，工艺文本含配比描述”，不能笼统写“未提供配方”。
- performance_rank 必须引用 scanned_sample_count / total_matching_sample_count；正常情况通过分页读取全部匹配记录，scan_truncated=true 时不得声称是全部授权数据的最终排名。
- 排名按数据库记录进行；同名不同 ID 是不同记录，不得擅自去重。
- 实验系列 sample_count=0 时只能报告未命中和 Tool Result 给出的 similar_names，不得编造系列目的。
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

        if intent in {"formula_difference", "process_difference"} and status == "ok":
            title = "配方差异" if intent == "formula_difference" else "工艺差异"
            lines = [f"【{title}】"]
            for item in result.get("changed_fields", []):
                delta = item.get("numeric_delta") or {}
                text = (
                    f"- {item.get('field')}：{item.get('left')} {item.get('left_unit') or ''}"
                    f" → {item.get('right')} {item.get('right_unit') or ''}"
                ).strip()
                if delta:
                    text += f"；左-右={delta.get('left_minus_right')}"
                    if delta.get("relative_to_right_percent") is not None:
                        text += f"；相对右样品={delta.get('relative_to_right_percent')}%"
                lines.append(text)
            lines.append(
                f"共 {result.get('summary', {}).get('changed_count', 0)} 个变化字段，"
                f"{result.get('summary', {}).get('same_count', 0)} 个相同字段。"
            )
            if intent == "formula_difference" and result.get("raw_numeric_totals"):
                totals = result["raw_numeric_totals"]
                lines.append(
                    "记录值算术和（不等于已归一化质量百分比）："
                    f"左={totals.get('left_raw_numeric_sum')}，"
                    f"右={totals.get('right_raw_numeric_sum')}。"
                )
            return "\n".join(lines)

        if intent == "comparability_check" and status == "ok":
            assessment = result.get("assessment") or {}
            grade_map = {
                "COMPARABLE_ON_RECORDED_EVIDENCE": "按已记录证据可比",
                "PARTIALLY_COMPARABLE": "部分可比，但证据不完整",
                "NOT_DIRECTLY_COMPARABLE": "不建议直接比较",
            }
            lines = [
                "【可比性判断】",
                grade_map.get(assessment.get("grade"), str(assessment.get("grade") or "未知")),
            ]
            for label, key in (("支持证据", "supports"), ("阻断因素", "blockers"), ("证据缺口", "evidence_gaps")):
                values = assessment.get(key) or []
                if values:
                    lines.append(f"{label}：")
                    lines.extend(f"- {x}" for x in values)
            lines.append(str(assessment.get("interpretation") or ""))
            return "\n".join(x for x in lines if x)

        if intent == "performance_rank":
            if status == "unit_mismatch":
                return "目标性能存在多个单位，已停止混合排序：" + "、".join(result.get("observed_units", []))
            lines = [f"【{result.get('target_metric')}排序】"]
            for index, row in enumerate(result.get("ranking", []), 1):
                sample = row.get("sample") or {}
                lines.append(f"{index}. {sample.get('id')}（{sample.get('name')}）：{row.get('value')} {row.get('unit') or ''}".strip())
            lines.append(str(result.get("ranking_basis") or ""))
            lines.append(
                f"扫描 {result.get('scanned_sample_count', 0)} / "
                f"{result.get('total_matching_sample_count', result.get('scanned_sample_count', 0))} 个匹配样品；"
                f"排除 {len(result.get('excluded_samples', []))} 个缺失或非数值记录。"
            )
            if result.get("scan_truncated"):
                lines.append("分页读取未完整结束；当前结果只代表已读取记录，不代表全部授权数据的最终排名。")
            return "\n".join(lines)

        if intent == "experiment_series_analysis" and status == "ok":
            lines = ["【实验系列分析】", f"共 {result.get('sample_count', 0)} 个样品。"]
            if result.get("sample_count", 0) == 0:
                suggestions = result.get("similar_names") or []
                if suggestions:
                    lines.append(
                        "当前关键词未命中；相近名称："
                        + "、".join(
                            f"{item.get('name')}（ID {item.get('id')}）"
                            for item in suggestions
                        )
                    )
                else:
                    lines.append("当前权限范围的样品名称中未找到该系列，也没有可验证的相近名称。")
            project_assessment = result.get("cross_project_assessment") or {}
            if project_assessment.get("project_count", 0):
                lines.append(
                    "项目分组："
                    + "、".join(str(x) for x in project_assessment.get("project_ids", []))
                )
                if project_assessment.get("requires_separate_analysis"):
                    lines.append(str(project_assessment.get("conclusion") or ""))
            inference = result.get("purpose_inference") or {}
            factors = inference.get("candidate_independent_factors") or []
            responses = inference.get("response_metrics") or []
            lines.append("主要变化因素：" + ("、".join(x.get("field", "") for x in factors[:8]) or "未识别到"))
            lines.append("响应指标：" + ("、".join(x.get("field", "") for x in responses[:12]) or "未识别到"))
            missing_count = int(
                (result.get("field_summary") or {}).get("missing_field_count", 0)
                or 0
            )
            if missing_count:
                lines.append(f"全组缺失字段：{missing_count} 项（不计为常量）。")
            if inference.get("summary"):
                lines.append(str(inference.get("summary")))
            if inference.get("causality_limit"):
                lines.append(str(inference.get("causality_limit")))
            lines.append(str(result.get("interpretation_limit") or ""))
            return "\n".join(lines)

        if intent == "data_quality_check" and status == "ok":
            summary = result.get("summary") or {}
            return "\n".join([
                "【数据质量检查】",
                f"样品数：{result.get('sample_count', 0)}",
                f"重复名称：{summary.get('duplicate_name_count', 0)}；缺失测试条件：{summary.get('missing_condition_count', 0)}（{summary.get('missing_condition_percent', '0.00')}%）",
                f"空数据区段：{summary.get('empty_section_count', 0)}；非数值性能：{summary.get('non_numeric_performance_count', 0)}；配方算术和提示：{summary.get('formula_total_warning_count', 0)}；未解析动态字段样品：{summary.get('unresolved_sample_count', 0)}。",
            ])

        return json.dumps(result, ensure_ascii=False, default=str, indent=2)
