from __future__ import annotations

import json
from typing import Any

from agent.router import LLMIntentRouter, RuleIntentRouter
from agent.scenario_composer import ScenarioWorkflowComposer
from agent.skill_registry import SkillRegistry
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
        skill_registry: SkillRegistry | None = None,
        scenario_composer: ScenarioWorkflowComposer | None = None,
    ):
        self.registry = registry
        self.llm = llm
        self.llm_enabled = llm_enabled
        if skill_registry is None:
            from skills.catalog import build_default_skill_registry

            skill_registry = build_default_skill_registry()
        self.skill_registry = skill_registry
        self.scenario_composer = (
            scenario_composer or ScenarioWorkflowComposer(skill_registry)
        )
        self.rule_router = RuleIntentRouter()
        self.llm_router = LLMIntentRouter(llm)
        self.material_intelligence_skill = MaterialIntelligenceSkill(registry)
        data_query = DataQuerySkill(registry)
        comparison = ComparisonSkill(registry)
        analysis = AnalysisSkill(registry)
        # Runtime handlers remain small and deterministic.  Selection is now
        # constrained by the declarative Skill Registry instead of iterating
        # every handler globally for a matching legacy intent.
        self.skill_handlers = {
            "knowledge_qa": [
                data_query,
                comparison,
                analysis,
                self.material_intelligence_skill,
            ],
            "data_governance": [data_query, self.material_intelligence_skill],
        }

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
        plan = self.scenario_composer.compose(
            intent=intent,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        handlers = self.skill_handlers.get(plan.primary_skill, [])
        for handler in handlers:
            if handler.can_handle(intent):
                if isinstance(handler, MaterialIntelligenceSkill):
                    result = handler.execute_intent(
                        intent,
                        tool_name,
                        dict(tool_args),
                        ctx,
                    )
                else:
                    result = handler.execute(tool_name, tool_args, ctx)
                self.skill_registry.get(plan.primary_skill).validate_output(result)
                return result
        raise ValueError(
            "Skill 已注册但没有运行处理器："
            f"skill={plan.primary_skill}, operation={intent}"
        )

    def answer(
        self,
        message: str,
        intent: str,
        tool_result: Any,
    ) -> str:
        # A simple aggregate should not wait for a second LLM call or give the
        # model an opportunity to recompute the database result.  The value and
        # its coverage statement are rendered directly from deterministic data.
        if intent == "performance_statistics":
            return self._deterministic_answer(intent, tool_result)
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
【面向用户的表达要求】
- 第一段直接回答用户问题，再给必要证据；不要先复述系统限制或工具结构。
- 正文不要暴露 Tool Result、purpose_inference、cross_project_assessment、field_summary、resolved、raw_key、source、conditions、recipe_batches、craft_detail、data_column、sample_materials 等内部键名或表名；应转换成自然中文，例如“数据库未记录测试条件”“该字段未完成结构化解析”。只有用户明确要求执行详情时才说明技术字段。
- null/None/空对象统一表述为“未记录”，不要直接展示 null、None 或空数组。
- 同一个限制或缺失数量只说明一次；证据缺口放在结尾，不要重复压过主要结论。
对于 performance_difference，优先按照 facts / hypotheses / evidence_gaps / conclusion_limit 组织答案。
对于 sample_full_profile / formula_difference / process_difference / comparability_check：
- 数值差值、相对变化、配方算术和只能直接引用 Tool Result 中的确定性计算，不得自行重新计算。
- comparability_check 的缺失测试条件必须表述为“无法确认一致”，不得当作“相同”。
- 只解释 Tool Result 已支持的事实，不根据常识补齐材料体系或测试标准。
对于 performance_rank / performance_statistics / experiment_series_analysis / data_quality_check / find_samples_multi_condition / similar_samples：
- 排序、计数、比例、缺失和异常标记只能引用 Tool Result，不得重新计算。
- find_samples_multi_condition 只能引用后端返回的 filters、matched_sample_count、matched_samples 和 filter_diagnostics；不要自行放宽/增加筛选条件，也不要把未命中写成数据库没有该样品。
- find_samples_multi_condition 遇到 field_not_found 或 unit_ambiguity 时必须明确停止原因；不得猜字段、猜单位或自行换算后继续筛选。
- find_samples_multi_condition 的 field_bindings 是后端依据授权字段目录完成的确定性绑定，例如“PC含量”→配方“PC”或错误类别的“成本”→性能“成本”；回答必须使用绑定后的 canonical 字段，不得恢复成模型原先的错误类别。
- find_samples_multi_condition 遇到 field_ambiguity 时列出候选并请用户明确，不能任选一个。
- find_samples_multi_condition 的 scanned_sample_count/total_matching_sample_count/scan_truncated 决定检索覆盖范围；同名不同 ID 不自动合并。
- experiment_series_analysis 必须严格区分 constant_fields、variable_fields 与 missing_fields；value 为 None/空值的字段不得写成常量。
- experiment_series_analysis 如果提供 purpose_inference，必须先说明数据库未显式记录目的，再引用其 summary 给出“实验设计推断”；不得因为缺少目的字段而完全拒绝回答，也不得把该推断写成已证实因果。
- 命中多个 project_id 时必须按 project_groups 分组说明，并引用 cross_project_assessment；不得把不同项目直接合并成重复实验。
- missing_fields 是证据缺口，只概括数量和关键字段，不要把大量 None 逐项列入“常量信息”。
- 缺失字段数量只能引用 field_summary.missing_field_count 或 project_groups[*].field_summary.missing_field_count，不得由模型自行数数组后写“约多少项”。
- 若结构化 formula 为空但工艺文本包含配比，只能写“未提供可解析的结构化配方，工艺文本含配比描述”，不能笼统写“未提供配方”。
- performance_rank 必须引用 scanned_sample_count / total_matching_sample_count；正常情况通过分页读取全部匹配记录，scan_truncated=true 时不得声称是全部授权数据的最终排名。
- performance_rank 虽保留兼容名称，但目标可以属于配方、工艺或性能；必须使用后端绑定后的 target_section/target_metric，并分别引用 numeric_sample_count、field_absent_sample_count、empty_value_sample_count、non_numeric_sample_count、ambiguous_sample_count；不得把字段出现数说成有效数值数，也不得把所有排除原因笼统写成“缺失”。
- performance_statistics 虽保留兼容名称，但目标可以属于配方、工艺或性能；平均值、字段类别、有效数值数、缺失数、非数值数和单位只能引用后端返回结果，不得自行重算。scan_truncated=true 时只能称为“已读取记录的平均值”，不得称为全部授权样品平均值。
- 排名按数据库记录进行；同名不同 ID 是不同记录，不得擅自去重。
- similar_samples 的相似度、字段覆盖率、归一化距离和排名只能引用 Tool Result，不得由模型重算。
- similar_samples 必须说明所用范围是配方、工艺或综合，并说明这是结构化数值接近度，不代表机理相同、性能等价或因果关系。
- similar_samples 不得隐藏低字段覆盖率；引用 compared_field_count/reference_field_count，综合模式分别说明配方和工艺分数。
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

        if intent == "find_samples_multi_condition":
            if status == "invalid_filters":
                return "筛选条件未通过安全校验，请明确字段、比较关系和值后重试。"
            if status == "field_not_found":
                fields = "、".join(
                    str(item.get("field") or item.get("requested_field") or "")
                    for item in result.get("unknown_filter_fields", [])
                ) or "未知字段"
                return f"当前读取范围内没有找到筛选字段：{fields}。请确认字段名称或样品范围。"
            if status == "field_ambiguity":
                descriptions = []
                for item in result.get("ambiguous_filter_fields", []):
                    candidates = "、".join(
                        f"{candidate.get('section')}.{candidate.get('field')}"
                        for candidate in item.get("candidates") or []
                    )
                    descriptions.append(
                        f"{item.get('requested_field')}"
                        + (f"（候选：{candidates}）" if candidates else "")
                    )
                return (
                    "筛选字段存在歧义，已停止执行。请明确字段类别："
                    + "；".join(descriptions)
                )
            if status == "unit_ambiguity":
                descriptions = []
                for item in result.get("unit_ambiguities", []):
                    units = "、".join(item.get("observed_units") or [])
                    descriptions.append(f"{item.get('field')}（{units}）")
                return (
                    "筛选已停止：同一字段存在多种或缺失单位，不能安全混合比较。"
                    + ("请明确单位：" + "；".join(descriptions) if descriptions else "")
                )
            if status == "ok":
                lines = [
                    f"共找到 {result.get('matched_sample_count', 0)} 个符合条件的样品。"
                ]
                for index, row in enumerate(result.get("matched_samples", []), 1):
                    sample = row.get("sample") or {}
                    lines.append(
                        f"{index}. {sample.get('id')}（{sample.get('name')}），"
                        f"project_id={sample.get('project_id')}"
                    )
                if result.get("results_truncated"):
                    lines.append(
                        f"当前只展示前 {result.get('returned_sample_count', 0)} 个匹配记录。"
                    )
                lines.append(
                    f"已扫描 {result.get('scanned_sample_count', 0)} / "
                    f"{result.get('total_matching_sample_count', result.get('scanned_sample_count', 0))} 条授权范围记录。"
                )
                if result.get("scan_truncated"):
                    lines.append("数据库分页读取未完整结束，当前结果不代表全部授权记录。")
                return "\n".join(lines)

        if intent == "similar_samples":
            if status == "insufficient_reference_fields":
                return "参照样品在所选范围内没有足够的唯一数值字段，暂时无法计算相似度。"
            if status == "no_comparable_candidates":
                return "当前授权样品中没有与参照样品共享同名、同单位数值字段的可比候选。"
            if status == "ok":
                reference = result.get("reference_sample") or {}
                scope_label = {
                    "formula": "配方",
                    "process": "工艺",
                    "combined": "配方与工艺综合",
                }.get(result.get("similarity_scope"), "综合")
                lines = [
                    f"按{scope_label}结构化数值计算，与 {reference.get('id')}（{reference.get('name')}）最相似的样品为："
                ]
                for index, row in enumerate(result.get("ranking") or [], 1):
                    sample = row.get("sample") or {}
                    lines.append(
                        f"{index}. {sample.get('id')}（{sample.get('name')}）："
                        f"{row.get('similarity_percent')}%；"
                        f"共同字段 {row.get('compared_field_count')}/"
                        f"{row.get('reference_field_count')}"
                    )
                lines.append(str(result.get("interpretation_limit") or ""))
                return "\n".join(line for line in lines if line)

        if intent == "performance_rank":
            metric = str(result.get("target_metric") or result.get("requested_target_metric") or "目标字段")
            section_label = str(result.get("target_section_label") or "字段")
            available_fields = result.get("available_fields") or []
            available = "、".join(
                f"{item.get('section_label')}.{item.get('name')}"
                for item in available_fields[:20]
            )
            if status == "field_not_found":
                suffix = f" 当前可见字段示例：{available}。" if available else ""
                return f"没有在当前授权字段目录中找到配方、工艺或性能字段“{metric}”。" + suffix
            if status == "ambiguous_field":
                candidates = "、".join((result.get("field_binding") or {}).get("candidates") or [])
                return f"字段“{metric}”对应多个类别（{candidates or '候选未明确'}），请明确是配方、工艺还是性能。"
            if status == "unit_mismatch":
                return f"目标{section_label}字段存在多个或缺失单位，已停止混合排序：" + "、".join(result.get("observed_units", []))
            if status == "no_numeric_values":
                return (
                    f"已找到{section_label}字段“{metric}”，但没有可用于排序的唯一有效数值。"
                    f"共扫描 {result.get('scanned_sample_count', 0)} 条："
                    f"字段未记录 {result.get('field_absent_sample_count', 0)} 条，"
                    f"空值 {result.get('empty_value_sample_count', 0)} 条，"
                    f"非数值 {result.get('non_numeric_sample_count', 0)} 条，"
                    f"重复字段 {result.get('ambiguous_sample_count', 0)} 条。"
                )
            lines = [f"【{section_label} · {result.get('target_metric')}排序】"]
            for index, row in enumerate(result.get("ranking", []), 1):
                sample = row.get("sample") or {}
                lines.append(f"{index}. {sample.get('id')}（{sample.get('name')}）：{row.get('value')} {row.get('unit') or ''}".strip())
            lines.append(str(result.get("ranking_basis") or ""))
            lines.append(
                f"扫描 {result.get('scanned_sample_count', 0)} / "
                f"{result.get('total_matching_sample_count', result.get('scanned_sample_count', 0))} 个匹配样品；"
                f"有效数值 {result.get('numeric_sample_count', 0)} 条；"
                f"字段未记录 {result.get('field_absent_sample_count', 0)} 条，"
                f"空值 {result.get('empty_value_sample_count', 0)} 条，"
                f"非数值 {result.get('non_numeric_sample_count', 0)} 条，"
                f"重复字段 {result.get('ambiguous_sample_count', 0)} 条。"
            )
            if result.get("scan_truncated"):
                lines.append("分页读取未完整结束；当前结果只代表已读取记录，不代表全部授权数据的最终排名。")
            return "\n".join(lines)

        if intent == "performance_statistics":
            metric = str(result.get("target_metric") or "目标字段")
            section_label = str(result.get("target_section_label") or "字段")
            available_fields = result.get("available_fields") or []
            available = "、".join(
                f"{item.get('section_label')}.{item.get('name')}"
                for item in available_fields[:20]
            )
            if status == "field_not_found":
                suffix = f" 当前可见字段示例：{available}。" if available else ""
                return f"没有在当前授权字段目录中找到配方、工艺或性能字段“{metric}”。" + suffix
            if status == "ambiguous_field":
                candidates = "、".join((result.get("field_binding") or {}).get("candidates") or [])
                return f"字段“{metric}”对应多个类别（{candidates or '候选未明确'}），请明确是配方、工艺还是性能。"
            if status == "unit_mismatch":
                units = "、".join(result.get("observed_units") or []) or "存在缺失单位"
                if result.get("unitless_numeric_count"):
                    units += "，并含未记录单位的数值"
                return (
                    f"无法安全计算{metric}平均值：有效数值记录的单位不一致（{units}）。"
                    "系统没有混合单位计算。"
                )
            if status == "no_numeric_values":
                suffix = f" 当前可见字段示例：{available}。" if available else ""
                return (
                    f"已找到{section_label}字段“{metric}”，但没有可用于计算平均值的唯一有效数值。"
                    f"共扫描 {result.get('scanned_sample_count', 0)} 条："
                    f"字段未记录 {result.get('field_absent_sample_count', 0)} 条，"
                    f"空值 {result.get('empty_value_sample_count', 0)} 条，"
                    f"非数值 {result.get('non_numeric_sample_count', 0)} 条，"
                    f"重复字段 {result.get('ambiguous_sample_count', 0)} 条。"
                    + suffix
                )
            if status == "ok":
                statistics = result.get("statistics") or {}
                unit = str(result.get("unit") or "").strip()
                scope_text = (
                    "当前公司和授权项目范围内"
                    if not result.get("scan_truncated")
                    else "当前已读取的授权记录中"
                )
                lines = [
                    f"{scope_text}，{section_label}字段“{metric}”的平均值为 "
                    f"{statistics.get('mean_display')}"
                    f"{(' ' + unit) if unit else ''}。",
                    (
                        f"共扫描 {result.get('scanned_sample_count', 0)} 条样品记录；"
                        f"其中 {result.get('numeric_sample_count', 0)} 条有效数值参与计算，"
                        f"{result.get('field_absent_sample_count', 0)} 条未记录该字段，"
                        f"{result.get('empty_value_sample_count', 0)} 条为空值，"
                        f"{result.get('non_numeric_sample_count', 0)} 条为非数值，"
                        f"{result.get('ambiguous_sample_count', 0)} 条目标字段不唯一。"
                    ),
                ]
                if result.get("scan_truncated"):
                    lines.append("数据库分页读取未完整结束，因此该值不代表全部授权样品。")
                else:
                    lines.append("数据库分页读取完整；缺失、非数值和字段不唯一的记录未计入平均值。")
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
