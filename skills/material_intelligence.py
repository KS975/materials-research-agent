from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
from threading import RLock
from time import monotonic
from typing import Any

from agent.field_catalog import (
    bind_metric_to_catalog,
    bind_filters_to_catalog,
    build_material_field_catalog,
    material_section_label,
    normalize_field_name,
)
from agent.multi_condition import normalize_filters, normalize_logic, normalize_unit
from agent.tool_registry import ToolRegistry
from runtime.progress import emit_progress
from schemas.user_context import UserContext


class MaterialIntelligenceSkill:
    """Deterministic material-R&D analysis built on read-only sample tools.

    Round 2A intentionally reuses permission-scoped read-only database tools. The LLM may
    explain the returned facts, but arithmetic, field differences and the
    recorded-data comparability grade are calculated here.
    """

    intents = {
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
    }

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._field_catalog_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
        self._field_catalog_lock = RLock()

    def get_field_catalog(
        self,
        ctx: UserContext,
        *,
        ttl_seconds: float = 60.0,
    ) -> dict[str, Any]:
        """Load and briefly cache a value-free catalogue for intent routing."""
        cache_key = (
            ctx.company_id,
            bool(ctx.all_projects),
            tuple(sorted(int(item) for item in ctx.project_ids)),
        )
        now = monotonic()
        with self._field_catalog_lock:
            cached = self._field_catalog_cache.get(cache_key)
            if cached and cached[0] > now:
                return cached[1]

        catalog = self.registry.execute("get_material_field_catalog", ctx=ctx)
        if not isinstance(catalog, dict) or catalog.get("status") != "ok":
            return catalog
        if not catalog.get("scan_complete", True):
            return {
                "status": "incomplete_catalog",
                "warnings": list(catalog.get("warnings") or []),
                "scan_truncated": bool(catalog.get("scan_truncated", True)),
            }
        with self._field_catalog_lock:
            self._field_catalog_cache[cache_key] = (
                now + max(1.0, float(ttl_seconds)),
                catalog,
            )
        return catalog

    def can_handle(self, intent: str) -> bool:
        return intent in self.intents

    def execute(self, tool_name: str, tool_args: dict, ctx: UserContext):
        intent = str(tool_args.pop("__intent", "") or "")
        # AgentCore passes the business intent separately in current versions,
        # so __intent is only a compatibility hook and normally empty.
        if not intent:
            intent = self._infer_intent(tool_name, tool_args)
        return self.execute_intent(intent, tool_name, tool_args, ctx)

    def execute_intent(
        self,
        intent: str,
        tool_name: str,
        tool_args: dict,
        ctx: UserContext,
    ) -> dict[str, Any]:
        scope_label = (
            "当前公司全部项目"
            if ctx.all_projects
            else "授权项目：" + "、".join(str(item) for item in ctx.project_ids)
        )
        if intent == "sample_full_profile":
            if tool_name != "get_sample_context":
                raise ValueError("sample_full_profile 只允许调用 get_sample_context")
            emit_progress(
                "sample_database_query",
                "running",
                f"读取样品 {tool_args.get('identifier')}",
                "正在从业务 MySQL 读取样品身份、配方、工艺、性能、测试条件和关联记录。",
                detail_items=[
                    {"label": "只读工具", "value": "get_sample_context"},
                    {"label": "样品标识", "value": str(tool_args.get("identifier"))},
                    {"label": "权限范围", "value": scope_label},
                ],
            )
            result = self.registry.execute(tool_name, ctx=ctx, **tool_args)
            enriched = self._full_profile(result)
            if isinstance(enriched, dict) and enriched.get("status") == "ok":
                sample = enriched.get("sample") or {}
                coverage = enriched.get("coverage") or {}
                emit_progress(
                    "sample_database_query",
                    "completed",
                    "样品档案读取完成",
                    (
                        f"已读取 {sample.get('id', tool_args.get('identifier'))}"
                        + (f"（{sample.get('name')}）" if sample.get("name") else "")
                        + f"，Project {sample.get('project_id', '-')}。"
                    ),
                    detail_items=[
                        {"label": "配方字段", "value": f"{coverage.get('formula_fields', 0)} 个"},
                        {"label": "工艺字段", "value": f"{coverage.get('process_fields', 0)} 个"},
                        {"label": "性能字段", "value": f"{coverage.get('performance_fields', 0)} 个"},
                        {"label": "测试条件", "value": "已记录" if coverage.get("has_test_conditions") else "缺失"},
                    ],
                )
            return enriched

        if intent in {
            "performance_rank",
            "performance_statistics",
            "experiment_series_analysis",
            "data_quality_check",
            "find_samples_multi_condition",
            "similar_samples",
        }:
            if tool_name != "list_samples_for_analysis":
                raise ValueError(f"{intent} 只允许调用 list_samples_for_analysis")
            normalized_filters: list[dict[str, Any]] = []
            if intent == "find_samples_multi_condition":
                normalized_filters, filter_errors = normalize_filters(
                    tool_args.get("filters")
                )
                if filter_errors:
                    return {
                        "status": "invalid_filters",
                        "analysis_type": intent,
                        "filters": [],
                        "validation_errors": filter_errors,
                        "evidence": [],
                        "warnings": ["筛选条件未通过安全校验，未查询数据库。"],
                    }
            reference = None
            if intent == "similar_samples":
                emit_progress(
                    "similarity_reference",
                    "running",
                    f"读取参照样品 {tool_args.get('identifier')}",
                    "正在确认参照样品及其可用配方、工艺字段。",
                    detail_items=[
                        {"label": "样品标识", "value": str(tool_args.get("identifier"))},
                        {"label": "相似范围", "value": str(tool_args.get("similarity_scope") or "combined")},
                    ],
                )
                reference = self.registry.execute(
                    "get_sample_context",
                    identifier=tool_args.get("identifier"),
                    ctx=ctx,
                )
                if not isinstance(reference, dict) or reference.get("status") != "ok":
                    return reference
                emit_progress(
                    "similarity_reference",
                    "completed",
                    "参照样品已确认",
                    "已读取参照样品，准备扫描授权候选集合。",
                )
            emit_progress(
                "database_scan",
                "running",
                "扫描授权样品库",
                "正在从业务 MySQL 分页读取候选样品及其配方、工艺、性能和测试条件。",
                detail_items=[
                    {"label": "任务", "value": intent},
                    {"label": "权限范围", "value": scope_label},
                    {"label": "检索关键词", "value": str(tool_args.get("keyword") or "全部授权样品")},
                    {"label": "单页读取", "value": str(tool_args.get("scan_limit") or 500)},
                ],
            )
            source = self.registry.execute(
                tool_name,
                keyword=str(tool_args.get("keyword") or ""),
                limit=int(tool_args.get("scan_limit") or 500),
                ctx=ctx,
            )
            if not isinstance(source, dict) or source.get("status") != "ok":
                return source
            emit_progress(
                "database_scan",
                "completed",
                "授权样品读取完成",
                f"已读取 {source.get('count', 0)} 条授权样品记录，准备执行确定性计算。",
                sample_count=source.get("count", 0),
                scan_complete=bool(source.get("scan_complete", True)),
                detail_items=[
                    {"label": "实际读取", "value": f"{source.get('count', 0)} 条"},
                    {"label": "分页数", "value": str(source.get("scan_page_count", source.get("page_count", "-")))},
                    {"label": "扫描完整", "value": "是" if source.get("scan_complete", True) else "否"},
                    {"label": "权限范围", "value": scope_label},
                ],
            )
            if intent == "performance_rank":
                result = self._performance_rank(
                    source,
                    target_metric=str(tool_args.get("target_metric") or "").strip(),
                    target_section=str(tool_args.get("target_section") or "auto").strip(),
                    top_n=int(tool_args.get("top_n") or 10),
                    order=str(tool_args.get("order") or "desc").lower(),
                )
                emit_progress(
                    "deterministic_calculation",
                    "completed",
                    "字段筛选与排序完成",
                    (
                        f"已将“{tool_args.get('target_metric')}”绑定为“"
                        f"{result.get('target_metric') or tool_args.get('target_metric')}”，"
                        f"使用 {result.get('numeric_sample_count', 0)} 条有效数值完成排序。"
                    ),
                    detail_items=[
                        {"label": "请求指标", "value": str(tool_args.get("target_metric") or "-")},
                        {"label": "字段类别", "value": str(result.get("target_section_label") or "未绑定")},
                        {"label": "数据库字段", "value": str(result.get("target_metric") or "未绑定")},
                        {"label": "有效数值", "value": f"{result.get('numeric_sample_count', 0)} 条"},
                        {"label": "字段未记录", "value": f"{result.get('field_absent_sample_count', 0)} 条"},
                        {"label": "空值", "value": f"{result.get('empty_value_sample_count', 0)} 条"},
                        {"label": "非数值", "value": f"{result.get('non_numeric_sample_count', 0)} 条"},
                        {"label": "重复字段", "value": f"{result.get('ambiguous_sample_count', 0)} 条"},
                        {"label": "排序方向", "value": str(tool_args.get("order") or "desc")},
                        {"label": "排名结果", "value": f"{len(result.get('ranking') or [])} 条"},
                    ],
                )
                return result
            if intent == "performance_statistics":
                result = self._performance_statistics(
                    source,
                    target_metric=str(tool_args.get("target_metric") or "").strip(),
                    target_section=str(tool_args.get("target_section") or "auto").strip(),
                )
                mean_value = (result.get("statistics") or {}).get("mean_display")
                emit_progress(
                    "deterministic_calculation",
                    "completed",
                    "字段平均值计算完成",
                    (
                        f"已对 {result.get('numeric_sample_count', 0)} 条有效"
                        f"{tool_args.get('target_metric')}记录计算平均值。"
                    ),
                    detail_items=[
                        {"label": "请求指标", "value": str(tool_args.get("target_metric") or "-")},
                        {"label": "字段类别", "value": str(result.get("target_section_label") or "未绑定")},
                        {"label": "数据库字段", "value": str(result.get("target_metric") or "未绑定")},
                        {"label": "有效数值", "value": f"{result.get('numeric_sample_count', 0)} 条"},
                        {"label": "字段未记录", "value": f"{result.get('field_absent_sample_count', 0)} 条"},
                        {"label": "空值", "value": f"{result.get('empty_value_sample_count', 0)} 条"},
                        {"label": "非数值", "value": f"{result.get('non_numeric_sample_count', 0)} 条"},
                        {"label": "重复字段", "value": f"{result.get('ambiguous_sample_count', 0)} 条"},
                        {"label": "平均值", "value": str(mean_value if mean_value is not None else "未计算")},
                    ],
                )
                return result
            if intent == "experiment_series_analysis":
                result = self._experiment_series(source)
                emit_progress(
                    "deterministic_calculation",
                    "completed",
                    "实验系列统计完成",
                    "已完成常量、变量、缺失字段和项目分组的确定性统计。",
                    detail_items=[{"label": "扫描样品", "value": f"{source.get('count', 0)} 条"}],
                )
                return result
            if intent == "find_samples_multi_condition":
                field_catalog = build_material_field_catalog(source)
                (
                    normalized_filters,
                    field_bindings,
                    binding_errors,
                ) = bind_filters_to_catalog(normalized_filters, field_catalog)
                if binding_errors:
                    ambiguous = [
                        item for item in binding_errors
                        if item.get("code") == "ambiguous_field"
                    ]
                    unknown = [
                        item for item in binding_errors
                        if item.get("code") == "field_not_found"
                    ]
                    return {
                        "status": (
                            "field_ambiguity" if ambiguous else "field_not_found"
                        ),
                        "analysis_type": intent,
                        "filters": [],
                        "field_bindings": field_bindings,
                        "ambiguous_filter_fields": ambiguous,
                        "unknown_filter_fields": unknown,
                        "scanned_sample_count": source.get("count", 0),
                        "total_matching_sample_count": source.get(
                            "total_matches", source.get("count", 0)
                        ),
                        "scan_complete": bool(source.get("scan_complete", True)),
                        "scan_truncated": bool(source.get("scan_truncated", False)),
                        "evidence": [],
                        "warnings": list(source.get("warnings") or []),
                    }
                result = self._multi_condition_filter(
                    source,
                    filters=normalized_filters,
                    logic=normalize_logic(tool_args.get("logic")),
                    result_limit=tool_args.get("result_limit", 50),
                    field_bindings=field_bindings,
                )
                emit_progress(
                    "deterministic_calculation",
                    "completed",
                    "多条件筛选完成",
                    f"已按 {len(normalized_filters)} 个真实字段条件筛选，命中 {result.get('total_matching_sample_count', 0)} 条。",
                    detail_items=[
                        {"label": "逻辑", "value": normalize_logic(tool_args.get("logic")).upper()},
                        {"label": "条件数量", "value": f"{len(normalized_filters)} 个"},
                        {"label": "命中样品", "value": f"{result.get('total_matching_sample_count', 0)} 条"},
                    ],
                    query_preview="\n".join(
                        f"{item.get('section')}.{item.get('field')} {item.get('operator')} {item.get('value', '')} {item.get('unit') or ''}".strip()
                        for item in normalized_filters
                    ),
                )
                return result
            if intent == "similar_samples":
                emit_progress(
                    "field_alignment",
                    "running",
                    "对齐可比字段",
                    "正在按字段名称和单位对齐数值型配方、工艺字段。",
                )
                result = self._similar_samples(
                    source,
                    reference=reference or {},
                    similarity_scope=str(
                        tool_args.get("similarity_scope") or "combined"
                    ),
                    top_n=tool_args.get("top_n", 5),
                )
                emit_progress(
                    "similarity_scoring",
                    "completed",
                    "相似度排名完成",
                    f"确定性计算已完成，返回前 {len(result.get('ranking') or [])} 个可比样品。",
                    comparable_candidate_count=result.get(
                        "comparable_candidate_count", 0
                    ),
                )
                return result
            result = self._data_quality(source)
            emit_progress(
                "deterministic_calculation",
                "completed",
                "数据质量检查完成",
                f"已对 {source.get('count', 0)} 条样品执行缺失、异常和解析状态检查。",
            )
            return result

        if tool_name != "compare_samples":
            raise ValueError(f"{intent} 只允许调用 compare_samples")

        compare_args = {
            "left_identifier": tool_args["left_identifier"],
            "right_identifier": tool_args["right_identifier"],
        }
        emit_progress(
            "sample_comparison_query",
            "running",
            f"读取样品 {compare_args['left_identifier']} 与 {compare_args['right_identifier']}",
            "正在从业务 MySQL 读取两条样品记录并对齐配方、工艺、性能和测试条件。",
            detail_items=[
                {"label": "只读工具", "value": "compare_samples"},
                {"label": "比较对象", "value": f"{compare_args['left_identifier']} ↔ {compare_args['right_identifier']}"},
                {"label": "权限范围", "value": scope_label},
            ],
        )
        comparison = self.registry.execute("compare_samples", ctx=ctx, **compare_args)
        if not isinstance(comparison, dict) or comparison.get("status") != "ok":
            return comparison
        left_sample = comparison.get("left_sample") or {}
        right_sample = comparison.get("right_sample") or {}
        emit_progress(
            "sample_comparison_query",
            "completed",
            "两条样品记录已对齐",
            (
                f"已读取 {left_sample.get('id', compare_args['left_identifier'])}"
                f"（{left_sample.get('name') or '未命名'}）与 "
                f"{right_sample.get('id', compare_args['right_identifier'])}"
                f"（{right_sample.get('name') or '未命名'}）。"
            ),
            detail_items=[
                {"label": "配方变化", "value": f"{len((comparison.get('formula_diff') or {}).get('changed') or [])} 个字段"},
                {"label": "工艺变化", "value": f"{len((comparison.get('process_diff') or {}).get('changed') or [])} 个字段"},
                {"label": "性能变化", "value": f"{len((comparison.get('performance_diff') or {}).get('changed') or [])} 个字段"},
                {"label": "测试条件", "value": str((comparison.get("test_conditions") or {}).get("status") or "-")},
            ],
        )

        if intent == "formula_difference":
            return self._field_difference_result(
                analysis_type="formula_difference",
                comparison=comparison,
                diff_key="formula_diff",
                include_totals=True,
            )
        if intent == "process_difference":
            return self._field_difference_result(
                analysis_type="process_difference",
                comparison=comparison,
                diff_key="process_diff",
                include_totals=False,
            )
        if intent == "comparability_check":
            return self._comparability_result(
                comparison,
                target_metric=str(tool_args.get("target_metric") or "").strip(),
            )
        raise ValueError(f"MaterialIntelligenceSkill 不支持 intent={intent}")

    @staticmethod
    def _infer_intent(tool_name: str, tool_args: dict[str, Any]) -> str:
        # Kept conservative. AgentCore normally calls execute_intent and this
        # method is only for direct compatibility with Skill.execute callers.
        if tool_name == "get_sample_context":
            return "sample_full_profile"
        raise ValueError("缺少 MaterialIntelligenceSkill 业务 intent")

    @staticmethod
    def _full_profile(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict) or result.get("status") != "ok":
            return result
        enriched = dict(result)
        enriched["analysis_type"] = "sample_full_profile"
        enriched["coverage"] = {
            "formula_fields": len(result.get("formula") or []),
            "process_fields": len(result.get("process") or []),
            "performance_fields": len(result.get("performance") or []),
            "service_performance_fields": len(result.get("service_performance") or []),
            "has_test_conditions": bool(result.get("conditions")),
            "has_recipe_batches": bool(result.get("recipe_batches")),
            "has_craft_detail": result.get("craft_detail") not in (None, "", {}, []),
            "synthesis_record_count": len(result.get("synthesis_records") or []),
            "verify_item_count": len(result.get("verify_items") or []),
        }
        return enriched

    @classmethod
    def _field_difference_result(
        cls,
        *,
        analysis_type: str,
        comparison: dict[str, Any],
        diff_key: str,
        include_totals: bool,
    ) -> dict[str, Any]:
        diff = comparison.get(diff_key) or {}
        changed = [cls._augment_delta(item) for item in (diff.get("changed") or [])]
        same = [cls._augment_delta(item) for item in (diff.get("same") or [])]
        payload: dict[str, Any] = {
            "status": "ok",
            "analysis_type": analysis_type,
            "left_sample": comparison.get("left_sample"),
            "right_sample": comparison.get("right_sample"),
            "changed_fields": changed,
            "same_fields": same,
            "summary": {
                "changed_count": len(changed),
                "same_count": len(same),
                "numeric_delta_count": sum(
                    1 for item in changed if item.get("numeric_delta") is not None
                ),
                "unit_mismatch_count": sum(
                    1 for item in changed + same if item.get("unit_match") is False
                ),
            },
            "evidence": comparison.get("evidence", []),
            "warnings": comparison.get("warnings", []),
            "calculation_policy": (
                "所有差值与相对变化均由后端 Decimal 确定性计算；LLM 不应重新计算。"
            ),
        }
        if include_totals:
            payload["raw_numeric_totals"] = cls._raw_totals(changed + same)
        return payload

    @classmethod
    def _augment_delta(cls, item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        result["numeric_delta"] = cls._numeric_difference(
            item.get("left"), item.get("right")
        )
        return result

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = Decimal(str(value).strip())
        except (InvalidOperation, ValueError, TypeError):
            return None
        return parsed if parsed.is_finite() else None

    @classmethod
    def _numeric_difference(cls, left: Any, right: Any) -> dict[str, Any] | None:
        left_d = cls._to_decimal(left)
        right_d = cls._to_decimal(right)
        if left_d is None or right_d is None:
            return None
        delta = left_d - right_d
        relative = None if right_d == 0 else (delta / right_d) * Decimal("100")
        return {
            "left": str(left_d),
            "right": str(right_d),
            "left_minus_right": str(delta),
            "absolute_difference": str(abs(delta)),
            "relative_to_right_percent": (
                str(relative.quantize(Decimal("0.01"))) if relative is not None else None
            ),
        }

    @classmethod
    def _raw_totals(cls, items: list[dict[str, Any]]) -> dict[str, Any]:
        left_values: list[Decimal] = []
        right_values: list[Decimal] = []
        units: set[str] = set()
        non_numeric_fields: list[str] = []

        for item in items:
            if item.get("left_present"):
                value = cls._to_decimal(item.get("left"))
                if value is None:
                    non_numeric_fields.append(str(item.get("field")))
                else:
                    left_values.append(value)
            if item.get("right_present"):
                value = cls._to_decimal(item.get("right"))
                if value is None and str(item.get("field")) not in non_numeric_fields:
                    non_numeric_fields.append(str(item.get("field")))
                elif value is not None:
                    right_values.append(value)
            for key in ("left_unit", "right_unit"):
                unit = str(item.get(key) or "").strip()
                if unit:
                    units.add(unit)

        return {
            "left_raw_numeric_sum": str(sum(left_values, Decimal("0"))),
            "right_raw_numeric_sum": str(sum(right_values, Decimal("0"))),
            "observed_units": sorted(units),
            "non_numeric_fields": non_numeric_fields,
            "interpretation": (
                "这是数据库记录值的算术和，不代表已归一化质量百分比；"
                "只有确认各字段计量基准一致后才能作配方总量解释。"
            ),
        }

    @classmethod
    def _comparability_result(
        cls,
        comparison: dict[str, Any],
        *,
        target_metric: str,
    ) -> dict[str, Any]:
        conditions = comparison.get("test_conditions") or {}
        left_sample = comparison.get("left_sample") or {}
        right_sample = comparison.get("right_sample") or {}
        perf_diff = comparison.get("performance_diff") or {}
        performance_items = list(perf_diff.get("changed") or []) + list(
            perf_diff.get("same") or []
        )
        common = [
            dict(item)
            for item in performance_items
            if item.get("left_present") and item.get("right_present")
        ]

        target = cls._find_metric(common, target_metric) if target_metric else None
        assessed = [target] if target is not None else common

        blockers: list[str] = []
        gaps: list[str] = []
        supports: list[str] = []

        if target_metric and target is None:
            blockers.append(f"两个样品没有共同记录目标性能“{target_metric}”。")
        elif assessed:
            mismatch = [x for x in assessed if x.get("unit_match") is False]
            if mismatch:
                blockers.append(
                    "存在性能单位不一致：" + "、".join(str(x.get("field")) for x in mismatch)
                )
            else:
                supports.append("共同性能字段的已记录单位未发现冲突。")
        else:
            gaps.append("两个样品没有共同的性能字段，无法评价性能数值可比性。")

        condition_status = str(conditions.get("status") or "")
        if condition_status == "different":
            blockers.append("数据库中记录的测试条件不同。")
        elif condition_status == "same":
            supports.append("数据库中记录的测试条件一致。")
        elif condition_status == "missing_both":
            gaps.append("双方测试条件均未记录，不能确认测试条件一致。")
        elif condition_status in {"missing_left", "missing_right"}:
            gaps.append("至少一方缺少测试条件，不能确认测试条件一致。")
        else:
            gaps.append("测试条件状态无法确认。")

        left_type = left_sample.get("sample_type")
        right_type = right_sample.get("sample_type")
        if left_type and right_type:
            if left_type == right_type:
                supports.append("样品类型记录一致。")
            else:
                gaps.append("样品类型记录不同，需要确认比较目的是否允许跨类型比较。")
        else:
            gaps.append("至少一方样品类型未记录，材料体系可比性证据不完整。")

        if blockers:
            grade = "NOT_DIRECTLY_COMPARABLE"
            strict = False
        elif gaps:
            grade = "PARTIALLY_COMPARABLE"
            strict = False
        else:
            grade = "COMPARABLE_ON_RECORDED_EVIDENCE"
            strict = True

        return {
            "status": "ok",
            "analysis_type": "comparability_check",
            "left_sample": left_sample,
            "right_sample": right_sample,
            "target_metric": target_metric or None,
            "assessment": {
                "grade": grade,
                "strict_comparable_on_recorded_evidence": strict,
                "supports": supports,
                "blockers": blockers,
                "evidence_gaps": gaps,
                "interpretation": (
                    "该结论只评价数据库中已记录的可比性证据；"
                    "缺失信息不会被自动视为一致。"
                ),
            },
            "common_performance_fields": common,
            "test_conditions": conditions,
            "evidence": comparison.get("evidence", []),
            "warnings": comparison.get("warnings", []),
        }

    @staticmethod
    def _find_metric(
        items: list[dict[str, Any]], target_metric: str
    ) -> dict[str, Any] | None:
        target = target_metric.strip()
        if not target:
            return None
        exact = [x for x in items if str(x.get("field") or "").strip() == target]
        if len(exact) == 1:
            return exact[0]
        fuzzy = [
            x
            for x in items
            if target in str(x.get("field") or "")
            or str(x.get("field") or "") in target
        ]
        return fuzzy[0] if len(fuzzy) == 1 else None

    @classmethod
    def _performance_rank(
        cls,
        source: dict[str, Any],
        *,
        target_metric: str,
        top_n: int,
        order: str,
        target_section: str = "auto",
    ) -> dict[str, Any]:
        catalog = build_material_field_catalog(source)
        binding = bind_metric_to_catalog(target_metric, catalog, section=target_section)
        available_fields = [
            {
                "section": section,
                "section_label": material_section_label(section),
                "name": str(item.get("name") or "").strip(),
            }
            for section in ("formula", "process", "performance")
            for item in (catalog.get("sections") or {}).get(section, [])
            if str(item.get("name") or "").strip()
        ]
        if binding.get("status") != "ok":
            return {
                "status": binding.get("status", "field_not_found"),
                "analysis_type": "performance_rank",
                "requested_target_metric": target_metric,
                "target_metric": target_metric,
                "target_section": None,
                "target_section_label": None,
                "field_binding": binding,
                "available_fields": available_fields,
                "available_performance_metrics": [
                    item["name"] for item in available_fields
                    if item["section"] == "performance"
                ],
                "ranking": [],
                "observed_units": [],
                "excluded_samples": [],
                "numeric_sample_count": 0,
                "field_absent_sample_count": source.get("count", 0),
                "empty_value_sample_count": 0,
                "non_numeric_sample_count": 0,
                "ambiguous_sample_count": 0,
                "scanned_sample_count": source.get("count", 0),
                "total_matching_sample_count": source.get(
                    "total_matches", source.get("count", 0)
                ),
                "scan_complete": bool(source.get("scan_complete", True)),
                "scan_truncated": bool(source.get("scan_truncated", False)),
                "evidence": source.get("evidence", []),
                "warnings": source.get("warnings", []),
            }

        canonical_metric = str(binding.get("canonical") or target_metric).strip()
        canonical_section = str(binding.get("section") or "performance").strip()
        section_label = material_section_label(canonical_section)
        canonical_key = normalize_field_name(canonical_metric)
        ranked = []
        excluded = []
        units: set[str] = set()
        unitless_numeric_count = 0
        field_absent_count = 0
        empty_value_count = 0
        non_numeric_count = 0
        ambiguous_count = 0
        for item in source.get("samples") or []:
            sample = item.get("sample") or {}
            matches = [
                field for field in (item.get(canonical_section) or [])
                if normalize_field_name(
                    field.get("name") or field.get("raw_key") or ""
                ) == canonical_key
            ]
            if not matches:
                field_absent_count += 1
                excluded.append({"sample": sample, "reason": f"目标{section_label}字段未记录"})
                continue
            if len(matches) != 1:
                ambiguous_count += 1
                excluded.append({"sample": sample, "reason": f"目标{section_label}字段记录不唯一"})
                continue
            field = matches[0]
            if cls._series_value_is_missing(field.get("value")):
                empty_value_count += 1
                excluded.append({"sample": sample, "reason": f"目标{section_label}字段值缺失"})
                continue
            value = cls._to_decimal(field.get("value"))
            if value is None:
                non_numeric_count += 1
                excluded.append({"sample": sample, "reason": f"目标{section_label}字段不是可排序数值"})
                continue
            unit = str(field.get("unit") or "").strip()
            if unit:
                units.add(unit)
            else:
                unitless_numeric_count += 1
            ranked.append({"sample": sample, "value": str(value), "unit": unit or None, "_value": value})

        numeric_sample_count = len(ranked)
        unit_mismatch = len(units) > 1 or (bool(units) and unitless_numeric_count > 0)
        if unit_mismatch:
            ranked = []
        else:
            reverse = order != "asc"
            ranked.sort(key=lambda row: (row["_value"], row["sample"].get("id")), reverse=reverse)
            ranked = ranked[: max(1, min(top_n, 100))]
        for row in ranked:
            row.pop("_value", None)
        return {
            "status": (
                "unit_mismatch"
                if unit_mismatch
                else "no_numeric_values"
                if numeric_sample_count == 0
                else "ok"
            ),
            "analysis_type": "performance_rank",
            "requested_target_metric": target_metric,
            "target_metric": canonical_metric,
            "target_section": canonical_section,
            "target_section_label": section_label,
            "field_binding": binding,
            "available_fields": available_fields,
            "order": "asc" if order == "asc" else "desc",
            "ranking": ranked,
            "observed_units": sorted(units),
            "unitless_numeric_count": unitless_numeric_count,
            "excluded_samples": excluded,
            "numeric_sample_count": numeric_sample_count,
            "field_absent_sample_count": field_absent_count,
            "empty_value_sample_count": empty_value_count,
            "non_numeric_sample_count": non_numeric_count,
            "ambiguous_sample_count": ambiguous_count,
            "excluded_sample_count": len(excluded),
            "scanned_sample_count": source.get("count", 0),
            "total_matching_sample_count": source.get("total_matches", source.get("count", 0)),
            "scan_limit": source.get("scan_limit"),
            "scan_page_size": source.get("scan_page_size", 500),
            "scan_page_count": source.get("scan_page_count", 1),
            "scan_complete": bool(source.get("scan_complete", True)),
            "scan_truncated": bool(source.get("scan_truncated", False)),
            "ranking_basis": (
                f"按数据库中的{section_label}字段“{canonical_metric}”排名；"
                "相同样品名称的不同 ID 不会自动合并。"
            ),
            "calculation_policy": "排序由后端 Decimal 确定性计算；不同单位不会混排。",
            "evidence": source.get("evidence", []),
            "warnings": source.get("warnings", []),
        }

    @classmethod
    def _performance_statistics(
        cls,
        source: dict[str, Any],
        *,
        target_metric: str,
        target_section: str = "auto",
    ) -> dict[str, Any]:
        catalog = build_material_field_catalog(source)
        binding = bind_metric_to_catalog(target_metric, catalog, section=target_section)
        available_fields = [
            {
                "section": section,
                "section_label": material_section_label(section),
                "name": str(item.get("name") or "").strip(),
            }
            for section in ("formula", "process", "performance")
            for item in (catalog.get("sections") or {}).get(section, [])
            if str(item.get("name") or "").strip()
        ]
        if binding.get("status") != "ok":
            return {
                "status": binding.get("status", "field_not_found"),
                "analysis_type": "performance_statistics",
                "requested_target_metric": target_metric,
                "target_metric": target_metric,
                "target_section": None,
                "target_section_label": None,
                "field_binding": binding,
                "requested_statistics": ["mean"],
                "statistics": {},
                "unit": None,
                "observed_units": [],
                "unitless_numeric_count": 0,
                "numeric_sample_count": 0,
                "missing_sample_count": source.get("count", 0),
                "field_absent_sample_count": source.get("count", 0),
                "empty_value_sample_count": 0,
                "non_numeric_sample_count": 0,
                "ambiguous_sample_count": 0,
                "excluded_sample_count": source.get("count", 0),
                "missing_samples": [],
                "non_numeric_samples": [],
                "ambiguous_samples": [],
                "available_fields": available_fields,
                "available_performance_metrics": sorted(
                    item["name"] for item in available_fields
                    if item["section"] == "performance"
                ),
                "scanned_sample_count": source.get("count", 0),
                "total_matching_sample_count": source.get(
                    "total_matches", source.get("count", 0)
                ),
                "scan_complete": bool(source.get("scan_complete", True)),
                "scan_truncated": bool(source.get("scan_truncated", False)),
                "evidence": source.get("evidence", []),
                "warnings": source.get("warnings", []),
            }

        canonical_metric = str(binding.get("canonical") or target_metric).strip()
        canonical_section = str(binding.get("section") or "performance").strip()
        section_label = material_section_label(canonical_section)
        canonical_key = normalize_field_name(canonical_metric)
        values: list[Decimal] = []
        missing_samples: list[dict[str, Any]] = []
        non_numeric_samples: list[dict[str, Any]] = []
        ambiguous_samples: list[dict[str, Any]] = []
        observed_units: set[str] = set()
        unitless_numeric_count = 0
        field_absent_count = 0
        empty_value_count = 0

        for item in source.get("samples") or []:
            sample = item.get("sample") or {}
            section_fields = item.get(canonical_section) or []
            matches = [
                field
                for field in section_fields
                if normalize_field_name(
                    field.get("name") or field.get("raw_key") or ""
                ) == canonical_key
            ]
            if not matches:
                field_absent_count += 1
                missing_samples.append({"sample": sample, "reason": f"目标{section_label}字段未记录"})
                continue
            if len(matches) != 1:
                ambiguous_samples.append({"sample": sample, "reason": f"目标{section_label}字段记录不唯一"})
                continue
            field = matches[0]
            if cls._series_value_is_missing(field.get("value")):
                empty_value_count += 1
                missing_samples.append({"sample": sample, "reason": f"目标{section_label}字段值缺失"})
                continue
            value = cls._to_decimal(field.get("value"))
            if value is None:
                non_numeric_samples.append({"sample": sample, "reason": f"目标{section_label}字段不是数值"})
                continue
            unit = str(field.get("unit") or "").strip()
            if unit:
                observed_units.add(unit)
            else:
                unitless_numeric_count += 1
            values.append(value)

        unit_mismatch = len(observed_units) > 1 or (
            bool(observed_units) and unitless_numeric_count > 0
        )
        statistics: dict[str, Any] = {}
        if values and not unit_mismatch:
            total = sum(values, Decimal("0"))
            mean = total / Decimal(len(values))
            mean_display = format(mean.quantize(Decimal("0.0001")), "f")
            mean_display = mean_display.rstrip("0").rstrip(".") or "0"
            statistics = {
                "mean": str(mean),
                "mean_display": mean_display,
                "sum": str(total),
            }

        if unit_mismatch:
            status = "unit_mismatch"
        elif not values:
            status = "no_numeric_values"
        else:
            status = "ok"

        return {
            "status": status,
            "analysis_type": "performance_statistics",
            "requested_target_metric": target_metric,
            "target_metric": canonical_metric,
            "target_section": canonical_section,
            "target_section_label": section_label,
            "field_binding": binding,
            "requested_statistics": ["mean"],
            "statistics": statistics,
            "unit": next(iter(observed_units)) if len(observed_units) == 1 else None,
            "observed_units": sorted(observed_units),
            "unitless_numeric_count": unitless_numeric_count,
            "numeric_sample_count": len(values),
            "missing_sample_count": len(missing_samples),
            "field_absent_sample_count": field_absent_count,
            "empty_value_sample_count": empty_value_count,
            "non_numeric_sample_count": len(non_numeric_samples),
            "ambiguous_sample_count": len(ambiguous_samples),
            "excluded_sample_count": (
                len(missing_samples)
                + len(non_numeric_samples)
                + len(ambiguous_samples)
            ),
            "missing_samples": missing_samples,
            "non_numeric_samples": non_numeric_samples,
            "ambiguous_samples": ambiguous_samples,
            "available_fields": available_fields,
            "available_performance_metrics": sorted(
                item["name"] for item in available_fields
                if item["section"] == "performance"
            ),
            "scanned_sample_count": source.get("count", 0),
            "total_matching_sample_count": source.get(
                "total_matches", source.get("count", 0)
            ),
            "scan_limit": source.get("scan_limit"),
            "scan_page_size": source.get("scan_page_size", 500),
            "scan_page_count": source.get("scan_page_count", 1),
            "scan_complete": bool(source.get("scan_complete", True)),
            "scan_truncated": bool(source.get("scan_truncated", False)),
            "calculation_policy": (
                "平均值由后端 Decimal 对有效数值记录确定性计算；缺失、非数值和"
                "重复目标字段不参与计算；不同单位或已知/缺失单位混合时停止计算。"
            ),
            "evidence": source.get("evidence", []),
            "warnings": source.get("warnings", []),
        }

    @staticmethod
    def _series_value_is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (dict, list, tuple, set)) and not value:
            return True
        text = str(value).strip()
        if not text:
            return True
        return text.lower() in {
            "none", "null", "nan", "n/a", "na", "-", "--", "—",
            "未填写", "未记录", "暂无", "未定义", "未定义工艺", "未知", "未设置",
        }

    @classmethod
    def _sort_series_values(cls, values: set[str]) -> list[str]:
        numeric = []
        for value in values:
            number = cls._to_decimal(value)
            if number is None:
                return sorted(values)
            numeric.append((number, value))
        return [value for _, value in sorted(numeric, key=lambda item: (item[0], item[1]))]

    @classmethod
    def _series_field_profile(
        cls,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        declared_presence: Counter[str] = Counter()
        observed_presence: Counter[str] = Counter()
        field_values: dict[str, set[str]] = {}
        field_units: dict[str, set[str]] = {}
        categories = ("formula", "process", "performance")

        for item in samples:
            declared_in_sample: set[str] = set()
            observed_in_sample: set[str] = set()
            for category in categories:
                for field in item.get(category) or []:
                    name = str(field.get("name") or field.get("raw_key") or "").strip()
                    if not name:
                        continue
                    key = f"{category}:{name}"
                    declared_in_sample.add(key)
                    value = field.get("value")
                    if cls._series_value_is_missing(value):
                        continue
                    observed_in_sample.add(key)
                    field_values.setdefault(key, set()).add(str(value).strip())
                    unit = str(field.get("unit") or "").strip()
                    if unit:
                        field_units.setdefault(key, set()).add(unit)
            declared_presence.update(declared_in_sample)
            observed_presence.update(observed_in_sample)

        constants: list[dict[str, Any]] = []
        variables: list[dict[str, Any]] = []
        missing_fields: list[dict[str, Any]] = []
        total = len(samples)
        for key in sorted(declared_presence):
            category, name = key.split(":", 1)
            values = cls._sort_series_values(field_values.get(key, set()))
            units = sorted(field_units.get(key, set()))
            observed_count = observed_presence[key]
            payload = {
                "category": category,
                "field": name,
                "present_count": declared_presence[key],
                "observed_count": observed_count,
                "missing_count": total - observed_count,
                "distinct_value_count": len(values),
                "values": values,
                "unit": units[0] if len(units) == 1 else None,
                "observed_units": units,
                "unit_mismatch": len(units) > 1,
            }
            if observed_count == 0:
                missing_fields.append(payload)
            elif observed_count == total and len(values) == 1:
                constants.append(payload)
            else:
                variables.append(payload)
        return {
            "constant_fields": constants,
            "variable_fields": variables,
            "missing_fields": missing_fields,
            "field_summary": {
                "constant_field_count": len(constants),
                "variable_field_count": len(variables),
                "missing_field_count": len(missing_fields),
            },
        }

    @staticmethod
    def _series_field_signature(profile: dict[str, Any]) -> tuple:
        fields = [
            ("observed", item)
            for item in profile["constant_fields"] + profile["variable_fields"]
        ] + [
            ("all_missing", item)
            for item in profile["missing_fields"]
        ]
        return tuple(sorted(
            (
                classification,
                item["category"],
                item["field"],
                tuple(item.get("values") or []),
                item.get("missing_count", 0),
            )
            for classification, item in fields
        ))

    @classmethod
    def _series_project_groups(
        cls,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for item in samples:
            project_id = (item.get("sample") or {}).get("project_id")
            grouped.setdefault(project_id, []).append(item)

        project_groups = []
        for project_id in sorted(grouped, key=lambda value: str(value)):
            group_samples = grouped[project_id]
            profile = cls._series_field_profile(group_samples)
            sample_rows = [item.get("sample") or {} for item in group_samples]
            project_groups.append({
                "project_id": project_id,
                "sample_count": len(group_samples),
                "samples": sample_rows,
                "sample_names": [str(item.get("name") or "") for item in sample_rows],
                **profile,
            })

        project_count = len(project_groups)
        name_patterns = {
            tuple(sorted(group.get("sample_names") or []))
            for group in project_groups
        }
        design_signatures = {
            cls._series_field_signature(group)
            for group in project_groups
        }
        multiple = project_count > 1
        assessment = {
            "project_count": project_count,
            "project_ids": [group.get("project_id") for group in project_groups],
            "status": "multiple_projects" if multiple else "single_project",
            "requires_separate_analysis": multiple,
            "same_sample_name_pattern": multiple and len(name_patterns) == 1,
            "same_recorded_design": multiple and len(design_signatures) == 1,
            "conclusion": (
                "命中多个 project_id；必须先按项目分别分析。即使样品名称和记录结构相同，也只能标记为可能的平行或重复系列，未经项目元数据验证不得直接合并。"
                if multiple else
                "全部命中记录属于同一 project_id。"
            ),
        }
        return project_groups, assessment

    @staticmethod
    def _format_series_factor(item: dict[str, Any]) -> str:
        values = "/".join(str(value) for value in (item.get("values") or [])[:8])
        unit = str(item.get("unit") or "").strip()
        suffix = f" {unit}" if unit else ""
        return f"{item.get('field')}（{values}{suffix}）" if values else str(item.get("field") or "")

    @classmethod
    def _infer_series_purpose(
        cls,
        profile: dict[str, Any],
        project_assessment: dict[str, Any],
    ) -> dict[str, Any]:
        controls = [
            item for item in profile["constant_fields"]
            if item.get("category") in {"formula", "process"}
        ]
        candidate_factors = [
            item for item in profile["variable_fields"]
            if item.get("category") in {"formula", "process"}
            and item.get("distinct_value_count", 0) > 1
        ]
        response_metrics = [
            item for item in profile["variable_fields"]
            if item.get("category") == "performance"
            and item.get("distinct_value_count", 0) > 1
        ]

        if len(candidate_factors) == 1 and controls and response_metrics:
            confidence = "medium_high"
            prefix = "较高置信度的实验设计推断"
        elif candidate_factors and response_metrics:
            confidence = "medium"
            prefix = "中等置信度的实验设计推断"
        elif candidate_factors:
            confidence = "low"
            prefix = "低置信度的实验设计推断"
        else:
            confidence = "insufficient"
            prefix = "证据不足"

        if candidate_factors:
            factor_text = "、".join(cls._format_series_factor(item) for item in candidate_factors[:6])
            control_text = "、".join(str(item.get("field") or "") for item in controls[:8]) or "其他记录条件"
            response_text = "、".join(str(item.get("field") or "") for item in response_metrics[:12])
            if response_text:
                summary = (
                    f"{prefix}：该系列在保持{control_text}等记录项不变时，系统改变{factor_text}，"
                    f"并记录{response_text}等响应指标；因此可能用于筛选或考察这些变化因素与性能响应之间的关系。"
                )
            else:
                summary = (
                    f"{prefix}：该系列记录了{factor_text}的系统变化，但缺少足够的可变性能响应指标，"
                    "无法进一步概括研究目的。"
                )
        else:
            summary = "数据库未记录显式实验目的，且没有识别到可验证的配方或工艺变化因素，无法形成实验设计目的推断。"

        if project_assessment.get("requires_separate_analysis"):
            summary += " 命中多个项目，以上仅概括共有的记录结构，各项目仍须分别验证，不能直接合并为重复实验。"

        return {
            "statement_type": "engineering_design_inference",
            "explicit_purpose_recorded": False,
            "confidence": confidence,
            "candidate_independent_factors": candidate_factors,
            "controlled_factors": controls,
            "response_metrics": response_metrics,
            "summary": summary,
            "causality_limit": "该结论只描述实验设计结构，不证明变化因素导致了性能变化，也不代表数据库明示了研究目的。",
        }

    @classmethod
    def _experiment_series(cls, source: dict[str, Any]) -> dict[str, Any]:
        samples = source.get("samples") or []
        profile = cls._series_field_profile(samples)
        project_groups, project_assessment = cls._series_project_groups(samples)
        purpose_inference = cls._infer_series_purpose(profile, project_assessment)
        total = len(samples)
        return {
            "status": "ok",
            "analysis_type": "experiment_series_analysis",
            "series_keyword": source.get("keyword") or None,
            "sample_count": total,
            "samples": [item.get("sample") for item in samples],
            "similar_names": source.get("similar_names") or [],
            "scanned_sample_count": source.get("count", 0),
            "total_matching_sample_count": source.get("total_matches", source.get("count", 0)),
            "scan_page_size": source.get("scan_page_size", 500),
            "scan_page_count": source.get("scan_page_count", 1),
            "scan_complete": bool(source.get("scan_complete", True)),
            "scan_truncated": bool(source.get("scan_truncated", False)),
            "project_groups": project_groups,
            "cross_project_assessment": project_assessment,
            **profile,
            "purpose_inference": purpose_inference,
            "interpretation_limit": "允许依据控制因素、系统变化因素和响应指标提出明确标注的实验设计推断；不得写成数据库明示目的或已证实的因果关系。",
            "evidence": source.get("evidence", []),
            "warnings": source.get("warnings", []),
        }

    @staticmethod
    def _normalized_field_name(value: Any) -> str:
        return "".join(str(value or "").strip().casefold().split())

    @classmethod
    def _filter_candidates(
        cls,
        sample_item: dict[str, Any],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        section = spec["section"]
        field = spec["field"]
        if section == "sample":
            sample = sample_item.get("sample") or {}
            return [{"value": sample.get(field), "unit": None}]

        if section == "conditions":
            conditions = sample_item.get("conditions") or {}
            if field == "*":
                return [
                    {"value": value, "unit": None, "condition_key": str(key)}
                    for key, value in conditions.items()
                ]
            target = cls._normalized_field_name(field)
            return [
                {"value": value, "unit": None, "condition_key": str(key)}
                for key, value in conditions.items()
                if cls._normalized_field_name(key) == target
            ]

        target = cls._normalized_field_name(field)
        return [
            {
                "value": item.get("value"),
                "unit": item.get("unit"),
                "resolved": item.get("resolved"),
            }
            for item in (sample_item.get(section) or [])
            if cls._normalized_field_name(
                item.get("name") or item.get("raw_key")
            ) == target
        ]

    @classmethod
    def _scalar_equal(cls, left: Any, right: Any) -> bool:
        left_number = cls._to_decimal(left)
        right_number = cls._to_decimal(right)
        if left_number is not None and right_number is not None:
            return left_number == right_number
        return str(left).strip().casefold() == str(right).strip().casefold()

    @classmethod
    def _evaluate_filter(
        cls,
        sample_item: dict[str, Any],
        spec: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any]]:
        candidates = cls._filter_candidates(sample_item, spec)
        operator = spec["operator"]

        if operator == "missing":
            matched = not candidates or all(
                cls._series_value_is_missing(item.get("value"))
                for item in candidates
            )
            return matched, "matched" if matched else "not_met", {
                "observed_count": sum(
                    not cls._series_value_is_missing(item.get("value"))
                    for item in candidates
                )
            }

        if operator == "exists":
            observed = [
                item for item in candidates
                if not cls._series_value_is_missing(item.get("value"))
            ]
            matched = bool(observed)
            return matched, "matched" if matched else "missing", {
                "observed_count": len(observed)
            }

        if not candidates:
            return False, "missing", {}
        if len(candidates) != 1:
            return False, "ambiguous", {"candidate_count": len(candidates)}

        candidate = candidates[0]
        value = candidate.get("value")
        unit = str(candidate.get("unit") or "").strip()
        detail = {"value": value, "unit": unit or None}
        if cls._series_value_is_missing(value):
            return False, "missing", detail

        expected_unit = str(spec.get("unit") or "").strip()
        if expected_unit and normalize_unit(unit) != normalize_unit(expected_unit):
            return False, "unit_mismatch", detail

        if operator == "contains":
            matched = str(spec.get("value")).casefold() in str(value).casefold()
            return matched, "matched" if matched else "not_met", detail

        if operator == "in":
            matched = any(
                cls._scalar_equal(value, expected)
                for expected in (spec.get("values") or [])
            )
            return matched, "matched" if matched else "not_met", detail

        if operator in {"eq", "ne"}:
            equal = cls._scalar_equal(value, spec.get("value"))
            matched = equal if operator == "eq" else not equal
            return matched, "matched" if matched else "not_met", detail

        number = cls._to_decimal(value)
        if number is None:
            return False, "non_numeric", detail

        if operator == "between":
            bounds = [cls._to_decimal(item) for item in spec.get("values") or []]
            if len(bounds) != 2 or any(item is None for item in bounds):
                return False, "invalid_expected_value", detail
            low, high = sorted((bounds[0], bounds[1]))
            matched = bool(low <= number <= high)
        else:
            expected = cls._to_decimal(spec.get("value"))
            if expected is None:
                return False, "invalid_expected_value", detail
            matched = {
                "gt": number > expected,
                "gte": number >= expected,
                "lt": number < expected,
                "lte": number <= expected,
            }[operator]
        return matched, "matched" if matched else "not_met", detail

    @classmethod
    def _multi_condition_preflight(
        cls,
        samples: list[dict[str, Any]],
        filters: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        unknown_fields: list[dict[str, Any]] = []
        unit_ambiguities: list[dict[str, Any]] = []
        numeric_operators = {"gt", "gte", "lt", "lte", "between"}

        for index, spec in enumerate(filters, 1):
            if spec["section"] == "sample":
                continue
            if spec["section"] == "conditions" and spec["field"] == "*":
                continue
            declared = []
            for sample_item in samples:
                declared.extend(cls._filter_candidates(sample_item, spec))
            if not declared:
                unknown_fields.append({
                    "filter_index": index,
                    "section": spec["section"],
                    "field": spec["field"],
                })
                continue

            if (
                spec["section"] in {"formula", "process", "performance"}
                and spec["operator"] in numeric_operators
                and not spec.get("unit")
            ):
                numeric_candidates = [
                    item for item in declared
                    if cls._to_decimal(item.get("value")) is not None
                ]
                normalized_units = {
                    normalize_unit(item.get("unit"))
                    for item in numeric_candidates
                }
                if len(normalized_units) > 1:
                    unit_ambiguities.append({
                        "filter_index": index,
                        "section": spec["section"],
                        "field": spec["field"],
                        "observed_units": sorted({
                            str(item.get("unit") or "未记录")
                            for item in numeric_candidates
                        }),
                    })
        return unknown_fields, unit_ambiguities

    @classmethod
    def _multi_condition_filter(
        cls,
        source: dict[str, Any],
        *,
        filters: list[dict[str, Any]],
        logic: str,
        result_limit: Any,
        field_bindings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            limit = max(1, min(int(result_limit), 100))
        except (TypeError, ValueError):
            limit = 50

        samples = list(source.get("samples") or [])
        unknown_fields, unit_ambiguities = cls._multi_condition_preflight(
            samples, filters
        ) if samples else ([], [])
        common = {
            "analysis_type": "find_samples_multi_condition",
            "filters": filters,
            "field_bindings": list(field_bindings or []),
            "logic": normalize_logic(logic),
            "scope_keyword": source.get("keyword") or None,
            "scanned_sample_count": source.get("count", len(samples)),
            "total_matching_sample_count": source.get(
                "total_matches", source.get("count", len(samples))
            ),
            "scan_page_size": source.get("scan_page_size", 500),
            "scan_page_count": source.get("scan_page_count", 1),
            "scan_complete": bool(source.get("scan_complete", True)),
            "scan_truncated": bool(source.get("scan_truncated", False)),
            "evidence": [],
            "warnings": list(source.get("warnings") or []),
        }
        if unknown_fields:
            return {
                "status": "field_not_found",
                **common,
                "unknown_filter_fields": unknown_fields,
                "matched_sample_count": 0,
                "matched_samples": [],
            }
        if unit_ambiguities:
            return {
                "status": "unit_ambiguity",
                **common,
                "unit_ambiguities": unit_ambiguities,
                "matched_sample_count": 0,
                "matched_samples": [],
                "calculation_policy": (
                    "同一筛选字段出现多个或缺失单位时，不在后端猜测换算关系。"
                ),
            }

        diagnostics = [
            {
                "filter_index": index,
                "filter": spec,
                "outcomes": Counter(),
            }
            for index, spec in enumerate(filters, 1)
        ]
        matched_rows: list[dict[str, Any]] = []
        effective_logic = normalize_logic(logic)
        for sample_item in samples:
            outcomes = []
            details = []
            for diagnostic, spec in zip(diagnostics, filters):
                matched, reason, detail = cls._evaluate_filter(sample_item, spec)
                diagnostic["outcomes"][reason] += 1
                outcomes.append(matched)
                if matched:
                    details.append({
                        "filter_index": diagnostic["filter_index"],
                        "section": spec["section"],
                        "field": spec["field"],
                        "operator": spec["operator"],
                        **detail,
                    })
            row_matches = (
                all(outcomes) if effective_logic == "and" else any(outcomes)
            )
            if row_matches:
                matched_rows.append({
                    "sample": sample_item.get("sample") or {},
                    "matched_conditions": details,
                })

        returned = matched_rows[:limit]
        evidence = [
            {"source": "eln_sample", "record_id": row["sample"].get("id")}
            for row in returned
        ]
        diagnostic_rows = []
        for item in diagnostics:
            diagnostic_rows.append({
                "filter_index": item["filter_index"],
                "filter": item["filter"],
                "outcomes": dict(sorted(item["outcomes"].items())),
            })

        return {
            "status": "ok",
            **common,
            "matched_sample_count": len(matched_rows),
            "returned_sample_count": len(returned),
            "excluded_sample_count": len(samples) - len(matched_rows),
            "result_limit": limit,
            "results_truncated": len(matched_rows) > limit,
            "matched_samples": returned,
            "filter_diagnostics": diagnostic_rows,
            "evidence": evidence,
            "calculation_policy": (
                "筛选、数值比较、缺失判断与计数均由后端确定性执行；"
                "不执行模型生成的 SQL，不自动换算单位，不把缺失值当作满足条件。"
            ),
        }

    @classmethod
    def _similarity_vector(
        cls,
        fields: list[dict[str, Any]],
        section: str,
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        """Build an unambiguous numeric vector keyed by section/name/unit."""
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for field in fields or []:
            if not field.get("resolved", bool(field.get("name"))):
                continue
            name = str(field.get("name") or "").strip()
            number = cls._to_decimal(field.get("value"))
            if not name or number is None:
                continue
            normalized_name = re.sub(r"\s+", "", name).casefold()
            unit = str(field.get("unit") or "").strip()
            key = (section, normalized_name, normalize_unit(unit))
            grouped.setdefault(key, []).append({
                "field": name,
                "value": number,
                "unit": unit or None,
            })
        # Repeated canonical fields are not silently collapsed.
        return {key: values[0] for key, values in grouped.items() if len(values) == 1}

    @classmethod
    def _similar_samples(
        cls,
        source: dict[str, Any],
        *,
        reference: dict[str, Any],
        similarity_scope: str,
        top_n: Any,
    ) -> dict[str, Any]:
        scope = str(similarity_scope or "combined").strip().casefold()
        if scope not in {"formula", "process", "combined"}:
            scope = "combined"
        try:
            result_limit = max(1, min(int(top_n), 20))
        except (TypeError, ValueError):
            result_limit = 5

        reference_sample = reference.get("sample") or {}
        reference_id = reference_sample.get("id")
        sections = (
            ("formula",)
            if scope == "formula"
            else ("process",)
            if scope == "process"
            else ("formula", "process")
        )
        reference_vectors = {
            section: cls._similarity_vector(
                list(reference.get(section) or []), section
            )
            for section in sections
        }
        usable_reference_sections = [
            section for section in sections if reference_vectors[section]
        ]
        common_payload = {
            "analysis_type": "similar_samples",
            "similarity_scope": scope,
            "reference_sample": reference_sample,
            "reference_field_counts": {
                section: len(reference_vectors[section]) for section in sections
            },
            "scanned_sample_count": source.get("count", 0),
            "total_matching_sample_count": source.get(
                "total_matches", source.get("count", 0)
            ),
            "scan_page_size": source.get("scan_page_size", 500),
            "scan_page_count": source.get("scan_page_count", 1),
            "scan_complete": bool(source.get("scan_complete", True)),
            "scan_truncated": bool(source.get("scan_truncated", False)),
            "result_limit": result_limit,
        }
        if not usable_reference_sections:
            return {
                "status": "insufficient_reference_fields",
                **common_payload,
                "comparable_candidate_count": 0,
                "ranking": [],
                "evidence": [{"source": "eln_sample", "record_id": reference_id}],
                "warnings": list(source.get("warnings") or []) + [
                    "参照样品在所选范围内没有可用于计算的唯一数值字段。"
                ],
            }

        candidates: list[tuple[dict[str, Any], dict[str, dict[tuple[str, str, str], dict[str, Any]]]]] = []
        field_values: dict[tuple[str, str, str], list[Decimal]] = {
            key: [payload["value"]]
            for section in usable_reference_sections
            for key, payload in reference_vectors[section].items()
        }
        for item in source.get("samples") or []:
            sample = item.get("sample") or {}
            if str(sample.get("id")) == str(reference_id):
                continue
            vectors = {
                section: cls._similarity_vector(
                    list(item.get(section) or []), section
                )
                for section in usable_reference_sections
            }
            candidates.append((item, vectors))
            for section in usable_reference_sections:
                for key, payload in vectors[section].items():
                    if key in field_values:
                        field_values[key].append(payload["value"])

        scales = {
            key: max(values) - min(values)
            for key, values in field_values.items()
        }
        ranking: list[dict[str, Any]] = []
        exclusion_counts: Counter[str] = Counter()
        for item, vectors in candidates:
            section_scores: dict[str, Decimal] = {}
            section_details: dict[str, Any] = {}
            all_details: list[dict[str, Any]] = []
            total_overlap = 0
            total_reference_fields = 0
            for section in usable_reference_sections:
                reference_vector = reference_vectors[section]
                candidate_vector = vectors[section]
                common_keys = sorted(set(reference_vector) & set(candidate_vector))
                reference_count = len(reference_vector)
                total_reference_fields += reference_count
                total_overlap += len(common_keys)
                normalized_distances: list[Decimal] = []
                for key in common_keys:
                    left = reference_vector[key]
                    right = candidate_vector[key]
                    scale = scales.get(key, Decimal("0"))
                    distance = (
                        abs(left["value"] - right["value"]) / scale
                        if scale > 0
                        else Decimal("0")
                    )
                    distance = min(Decimal("1"), max(Decimal("0"), distance))
                    normalized_distances.append(distance)
                    all_details.append({
                        "section": section,
                        "field": left["field"],
                        "reference_value": str(left["value"]),
                        "candidate_value": str(right["value"]),
                        "unit": left.get("unit"),
                        "normalized_distance": str(
                            distance.quantize(Decimal("0.0001"))
                        ),
                    })
                coverage = (
                    Decimal(len(common_keys)) / Decimal(reference_count)
                    if reference_count
                    else Decimal("0")
                )
                mean_distance = (
                    sum(normalized_distances, Decimal("0"))
                    / Decimal(len(normalized_distances))
                    if normalized_distances
                    else Decimal("1")
                )
                score = (Decimal("1") - mean_distance) * coverage
                section_scores[section] = max(Decimal("0"), score)
                section_details[section] = {
                    "reference_field_count": reference_count,
                    "compared_field_count": len(common_keys),
                    "field_coverage_percent": str(
                        (coverage * 100).quantize(Decimal("0.01"))
                    ),
                    "mean_normalized_distance": str(
                        mean_distance.quantize(Decimal("0.0001"))
                    ),
                    "similarity_percent": str(
                        (section_scores[section] * 100).quantize(Decimal("0.01"))
                    ),
                }
            if total_overlap == 0:
                exclusion_counts["没有名称和单位均一致的共同数值字段"] += 1
                continue

            combined_score = (
                sum(section_scores.values(), Decimal("0"))
                / Decimal(len(usable_reference_sections))
            )
            all_details.sort(
                key=lambda row: (
                    Decimal(str(row["normalized_distance"])),
                    row["section"],
                    row["field"],
                ),
                reverse=True,
            )
            ranking.append({
                "sample": item.get("sample") or {},
                "similarity_percent": str(
                    (combined_score * 100).quantize(Decimal("0.01"))
                ),
                "formula_similarity_percent": (
                    str((section_scores["formula"] * 100).quantize(Decimal("0.01")))
                    if "formula" in section_scores else None
                ),
                "process_similarity_percent": (
                    str((section_scores["process"] * 100).quantize(Decimal("0.01")))
                    if "process" in section_scores else None
                ),
                "compared_field_count": total_overlap,
                "reference_field_count": total_reference_fields,
                "section_details": section_details,
                "largest_normalized_differences": all_details[:8],
                "_score": combined_score,
            })

        ranking.sort(
            key=lambda row: (
                -row["_score"],
                -int(row["compared_field_count"]),
                int((row.get("sample") or {}).get("id") or 0),
            )
        )
        comparable_count = len(ranking)
        ranking = ranking[:result_limit]
        for row in ranking:
            row.pop("_score", None)
        evidence = [{"source": "eln_sample", "record_id": reference_id}]
        evidence.extend({
            "source": "eln_sample",
            "record_id": (row.get("sample") or {}).get("id"),
        } for row in ranking)
        return {
            "status": "ok" if ranking else "no_comparable_candidates",
            **common_payload,
            "comparable_candidate_count": comparable_count,
            "excluded_candidate_count": len(candidates) - comparable_count,
            "exclusion_counts": dict(exclusion_counts),
            "ranking": ranking,
            "results_truncated": comparable_count > result_limit,
            "calculation_policy": (
                "相似度由后端确定性计算：仅比较名称和单位均一致的唯一数值字段；"
                "各字段按授权候选集合的极差进行 Min-Max 距离归一化；"
                "字段平均相似度乘以参照字段覆盖率，综合模式对配方和工艺等权平均。"
            ),
            "interpretation_limit": (
                "该分数表示当前数据库结构化数值的接近程度，不是化学机理、"
                "语义相似或性能等价性证明。"
            ),
            "evidence": evidence,
            "warnings": list(source.get("warnings") or []),
        }

    @classmethod
    def _data_quality(cls, source: dict[str, Any]) -> dict[str, Any]:
        samples = source.get("samples") or []
        names = [str((item.get("sample") or {}).get("name") or "") for item in samples]
        duplicate_names = sorted(name for name, count in Counter(names).items() if name and count > 1)
        missing_conditions = []
        empty_sections = []
        non_numeric_performance = []
        formula_total_flags = []
        for item in samples:
            sample = item.get("sample") or {}
            sid = sample.get("id")
            if not item.get("conditions"):
                missing_conditions.append(sid)
            for section in ("formula", "process", "performance"):
                if not item.get(section):
                    empty_sections.append({"sample_id": sid, "section": section})
            for field in item.get("performance") or []:
                if cls._to_decimal(field.get("value")) is None:
                    non_numeric_performance.append({
                        "sample_id": sid,
                        "field": field.get("name") or field.get("raw_key"),
                        "value": field.get("value"),
                    })
            numeric_formula = [cls._to_decimal(x.get("value")) for x in item.get("formula") or []]
            numeric_formula = [x for x in numeric_formula if x is not None]
            if numeric_formula:
                total = sum(numeric_formula, Decimal("0"))
                if total < Decimal("99") or total > Decimal("101"):
                    formula_total_flags.append({
                        "sample_id": sid,
                        "raw_numeric_sum": str(total),
                        "warning": "记录值算术和偏离100；未确认计量基准前不能直接认定配方错误。",
                    })
        count = len(samples)
        return {
            "status": "ok",
            "analysis_type": "data_quality_check",
            "scope_keyword": source.get("keyword") or None,
            "sample_count": count,
            "summary": {
                "duplicate_name_count": len(duplicate_names),
                "missing_condition_count": len(missing_conditions),
                "missing_condition_percent": str((Decimal(len(missing_conditions)) * 100 / count).quantize(Decimal("0.01"))) if count else "0.00",
                "empty_section_count": len(empty_sections),
                "non_numeric_performance_count": len(non_numeric_performance),
                "formula_total_warning_count": len(formula_total_flags),
                "unresolved_sample_count": len(source.get("unresolved_dynamic_fields") or []),
            },
            "issues": {
                "duplicate_names": duplicate_names,
                "missing_condition_sample_ids": missing_conditions,
                "empty_sections": empty_sections,
                "non_numeric_performance": non_numeric_performance,
                "formula_total_flags": formula_total_flags,
                "unresolved_dynamic_fields": source.get("unresolved_dynamic_fields") or [],
            },
            "calculation_policy": "计数、比例、重复和算术和均由后端确定性计算。",
            "evidence": source.get("evidence", []),
            "warnings": source.get("warnings", []),
        }
