from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from agent.tool_registry import ToolRegistry
from schemas.user_context import UserContext


class AnalysisSkill:
    intents = {"analyze_cause", "analyze_performance_difference"}

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def can_handle(self, intent: str) -> bool:
        return intent in self.intents

    def execute(self, tool_name: str, tool_args: dict, ctx: UserContext):
        if tool_name == "get_sample_context":
            return self.registry.execute(tool_name, ctx=ctx, **tool_args)

        if tool_name != "compare_samples":
            raise ValueError(f"AnalysisSkill 不允许调用工具：{tool_name}")

        left_identifier = tool_args["left_identifier"]
        right_identifier = tool_args["right_identifier"]
        target_metric = str(tool_args.get("target_metric") or "").strip()
        direction = str(tool_args.get("direction") or "").strip()

        comparison = self.registry.execute(
            "compare_samples",
            ctx=ctx,
            left_identifier=left_identifier,
            right_identifier=right_identifier,
        )

        if not isinstance(comparison, dict) or comparison.get("status") != "ok":
            return comparison

        changed_performance = comparison.get("performance_diff", {}).get("changed", [])
        target = self._find_target_metric(changed_performance, target_metric)

        if target is None:
            available = [str(item.get("field")) for item in changed_performance]
            return {
                "status": "target_metric_not_found",
                "target_metric": target_metric,
                "available_changed_performance": available,
                "comparison": comparison,
                "evidence": comparison.get("evidence", []),
                "warnings": [
                    f"在两个样品的性能差异中没有找到目标指标：{target_metric}"
                ],
            }

        numeric = self._numeric_difference(target.get("left"), target.get("right"))
        direction_check = self._check_direction(direction, numeric)

        formula_changed = comparison.get("formula_diff", {}).get("changed", [])
        process_changed = comparison.get("process_diff", {}).get("changed", [])
        condition_info = comparison.get("test_conditions", {})

        evidence_gaps: list[str] = []
        if formula_changed or process_changed:
            evidence_gaps.append(
                "两个样品存在多个配方和/或工艺变量同时变化，当前对比不能识别单一变量的独立贡献。"
            )
        if condition_info.get("status") in {"missing_both", "missing_left", "missing_right"}:
            evidence_gaps.append(
                "测试条件记录不完整，无法确认目标性能是否在完全一致的测试条件下获得。"
            )
        evidence_gaps.append(
            "当前只有样品间观察性对比，缺少单因素控制实验或其它因果识别设计。"
        )
        evidence_gaps.append(
            "当前结果未提供重复实验、误差范围或统计显著性，不能判断性能差异的统计稳定性。"
        )

        hypotheses: list[dict[str, Any]] = []
        if formula_changed:
            hypotheses.append(
                {
                    "statement": "配方差异可能与目标性能差异相关。",
                    "basis": f"检测到 {len(formula_changed)} 个配方字段发生变化。",
                    "causal_status": "hypothesis_only",
                }
            )
        if process_changed:
            hypotheses.append(
                {
                    "statement": "工艺参数差异可能与目标性能差异相关。",
                    "basis": f"检测到 {len(process_changed)} 个工艺字段发生变化。",
                    "causal_status": "hypothesis_only",
                }
            )

        warnings = list(comparison.get("warnings", []))
        if direction_check and direction_check.get("matches_user_claim") is False:
            warnings.append(
                "用户描述的高/低方向与数据库数值不一致，请以数据库事实为准。"
            )

        return {
            "status": "ok",
            "analysis_type": "performance_difference",
            "target_metric": target_metric,
            "direction_claim": direction,
            "facts": {
                "left_sample": comparison.get("left_sample"),
                "right_sample": comparison.get("right_sample"),
                "target_performance": target,
                "numeric_difference": numeric,
                "direction_check": direction_check,
                "formula_changes": formula_changed,
                "process_changes": process_changed,
                "test_conditions": condition_info,
            },
            "hypotheses": hypotheses,
            "evidence_gaps": evidence_gaps,
            "conclusion_limit": (
                "当前证据可以说明两个样品之间存在配方、工艺与性能差异，"
                "但不能据此认定其中任何一个差异是目标性能变化的原因。"
            ),
            "evidence": comparison.get("evidence", []),
            "warnings": warnings,
        }

    @staticmethod
    def _find_target_metric(
        changed_performance: list[dict[str, Any]],
        target_metric: str,
    ) -> dict[str, Any] | None:
        target_metric = target_metric.strip()
        if not target_metric:
            return None

        for item in changed_performance:
            if str(item.get("field") or "").strip() == target_metric:
                return item

        # Conservative fuzzy fallback: only accept a single unambiguous containment match.
        candidates = [
            item
            for item in changed_performance
            if target_metric in str(item.get("field") or "")
            or str(item.get("field") or "") in target_metric
        ]
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _numeric_difference(left: Any, right: Any) -> dict[str, Any] | None:
        try:
            left_d = Decimal(str(left))
            right_d = Decimal(str(right))
        except (InvalidOperation, TypeError, ValueError):
            return None

        absolute = left_d - right_d
        relative_percent = None
        if right_d != 0:
            relative_percent = (absolute / right_d) * Decimal("100")

        return {
            "left": str(left_d),
            "right": str(right_d),
            "left_minus_right": str(absolute),
            "relative_to_right_percent": (
                str(relative_percent.quantize(Decimal("0.01")))
                if relative_percent is not None
                else None
            ),
        }

    @staticmethod
    def _check_direction(direction: str, numeric: dict[str, Any] | None) -> dict[str, Any] | None:
        if not direction or numeric is None:
            return None

        diff = Decimal(numeric["left_minus_right"])
        lower_claims = {"更低", "低", "下降", "差"}
        higher_claims = {"更高", "高", "上升", "好"}

        if direction in lower_claims:
            matches = diff < 0
        elif direction in higher_claims:
            matches = diff > 0
        else:
            return None

        return {
            "claim": direction,
            "matches_user_claim": matches,
        }
