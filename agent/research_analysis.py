from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any, Mapping


_SECTION_LABELS = {
    "formula": "配方",
    "process": "工艺",
    "performance": "性能",
    "service_performance": "服务性能",
}
_TARGET_SECTIONS = ("performance", "service_performance")
_FEATURE_SECTIONS = ("formula", "process")


def run_research_analysis(
    *,
    scenario_id: int,
    source: Mapping[str, Any],
    args: Mapping[str, Any],
    message: str = "",
) -> dict[str, Any]:
    """Run one deterministic stage-4 analysis over an authorized MySQL scan."""
    dataset = build_evidence_dataset(source)
    payload = dict(args or {})
    if scenario_id == 8:
        return analyze_key_variables(dataset, payload, message=message)
    if scenario_id == 9:
        return discover_process_window(dataset, payload, message=message)
    if scenario_id == 10:
        return analyze_performance_conflicts(dataset, payload, message=message)
    if scenario_id == 11:
        return analyze_batch_differences(dataset, payload, message=message)
    if scenario_id == 18:
        return build_stage_report(dataset, payload, message=message)
    raise ValueError(f"场景 {scenario_id} 不属于阶段四分析 Workflow")


def build_evidence_dataset(source: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    field_profile: dict[str, dict[str, Any]] = {}
    warnings = list(source.get("warnings") or [])
    ambiguous_fields: list[str] = []

    for item in source.get("samples") or []:
        if not isinstance(item, Mapping):
            continue
        sample = dict(item.get("sample") or {})
        sample_id = sample.get("id")
        if sample_id is None:
            continue
        values: dict[str, float] = {}
        units: dict[str, str | None] = {}
        row_fields = 0
        for section in (*_FEATURE_SECTIONS, *_TARGET_SECTIONS):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for field in item.get(section) or []:
                if not isinstance(field, Mapping):
                    continue
                name = str(field.get("name") or field.get("raw_key") or "").strip()
                if not name:
                    continue
                grouped[name].append(dict(field))
            for name, fields in grouped.items():
                key = f"{section}.{name}"
                profile = field_profile.setdefault(
                    key,
                    {
                        "key": key,
                        "section": section,
                        "section_label": _SECTION_LABELS.get(section, section),
                        "name": name,
                        "unit": None,
                        "numeric_sample_count": 0,
                        "observed_sample_count": 0,
                        "_observed_units": set(),
                    },
                )
                profile["observed_sample_count"] += 1
                row_fields += 1
                if len(fields) != 1:
                    ambiguous_fields.append(f"sample:{sample_id}/{key}")
                    continue
                field = fields[0]
                unit = str(field.get("unit") or "").strip() or None
                profile["_observed_units"].add(unit)
                number = _number(field.get("value"))
                if number is None:
                    continue
                values[key] = number
                units[key] = unit
                profile["numeric_sample_count"] += 1
        rows.append(
            {
                "sample_id": str(sample_id),
                "sample_name": str(sample.get("name") or sample_id),
                "project_id": sample.get("project_id"),
                "sample_type": sample.get("sample_type"),
                "observed_at": str(sample.get("create_time") or ""),
                "values": values,
                "units": units,
                "field_count": row_fields,
            }
        )

    unit_excluded_fields: list[str] = []
    for profile in field_profile.values():
        observed_units = profile.pop("_observed_units", set())
        non_empty_units = {unit for unit in observed_units if unit}
        unit_mismatch = (
            len(non_empty_units) > 1
            or (bool(non_empty_units) and None in observed_units)
        )
        profile["unit"] = next(iter(non_empty_units)) if len(non_empty_units) == 1 else (
            "MIXED" if non_empty_units else None
        )
        key = str(profile["key"])
        if unit_mismatch:
            unit_excluded_fields.append(key)
            profile["excluded_reason"] = "unit_mismatch"
            for row in rows:
                row["values"].pop(key, None)
                row["units"].pop(key, None)
        profile["numeric_sample_count"] = sum(
            key in row["values"] for row in rows
        )

    if ambiguous_fields:
        warnings.append(
            f"有 {len(ambiguous_fields)} 个样品字段重名，相关字段未参与确定性计算。"
        )
    if unit_excluded_fields:
        warnings.append(
            "以下字段存在单位缺失或单位混杂，已整体排除确定性计算："
            + "、".join(unit_excluded_fields[:20])
            + ("等" if len(unit_excluded_fields) > 20 else "")
            + "。"
        )
    return {
        "schema_version": 1,
        "dataset_type": "read_only_evidence_dataset",
        "status": "ok" if rows else "no_samples",
        "sample_count": len(rows),
        "source_sample_count": source.get("count", len(rows)),
        "total_matching_sample_count": source.get(
            "total_matches", source.get("count", len(rows))
        ),
        "scan_complete": bool(source.get("scan_complete", True)),
        "scan_truncated": bool(source.get("scan_truncated", False)),
        "feature_fields": sorted(
            key for key in field_profile
            if key.split(".", 1)[0] in _FEATURE_SECTIONS
        ),
        "target_fields": sorted(
            key for key in field_profile
            if key.split(".", 1)[0] in _TARGET_SECTIONS
        ),
        "field_catalog": [
            field_profile[key] for key in sorted(field_profile)
        ],
        "rows": rows,
        "sample_refs": [
            f"eln_sample:{row['sample_id']}" for row in rows
        ],
        "evidence": list(source.get("evidence") or []),
        "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        "calculation_policy": (
            "只读授权样本的配方、工艺和性能数值；缺失、非数值和重名字段不参与计算，"
            "不做单位换算、不推断未记录实验。"
        ),
    }


def analyze_key_variables(
    dataset: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    message: str = "",
) -> dict[str, Any]:
    target = _requested_target(dataset, args)
    if not target:
        return _field_error(dataset, "key_variable_analysis", "缺少可分析的目标性能字段")
    rows = _rows_with_target(dataset, target)
    if len(rows) < 3:
        return _insufficient(dataset, "key_variable_analysis", target)

    variables: list[dict[str, Any]] = []
    for feature in dataset.get("feature_fields") or []:
        pairs = _pairs(dataset, feature, target)
        result = _correlation(pairs)
        coverage = len(pairs)
        if coverage < 3:
            continue
        unit = _field_unit(dataset, feature)
        variables.append(
            {
                "field": feature,
                "field_label": _field_label(dataset, feature),
                "unit": unit,
                "paired_sample_count": coverage,
                "coverage": round(coverage / len(rows), 4),
                "correlation": _round(result["correlation"]),
                "absolute_correlation": _round(abs(result["correlation"])),
                "direction": (
                    "positive" if result["correlation"] > 0
                    else "negative" if result["correlation"] < 0
                    else "none"
                ),
                "interpretation": (
                    "同步变化" if result["correlation"] > 0
                    else "反向变化" if result["correlation"] < 0
                    else "无线性相关"
                ),
                "constant": result["constant"],
            }
        )
    variables.sort(
        key=lambda item: (
            -float(item["absolute_correlation"] or 0),
            -int(item["paired_sample_count"]),
            str(item["field"]),
        )
    )
    return {
        "status": "ok",
        "analysis_type": "key_variable_analysis",
        "scenario_id": 8,
        "target": target,
        "target_label": _field_label(dataset, target),
        "target_unit": _field_unit(dataset, target),
        "sample_count": dataset.get("sample_count", 0),
        "target_valid_sample_count": len(rows),
        "minimum_paired_samples": 3,
        "variables": variables[:30],
        "variable_count": len(variables),
        "dataset": dataset,
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": dataset.get("warnings", []),
        "conclusion_limit": (
            "排名来自皮尔逊线性相关性，只代表历史样本中的同步变化，不构成因果结论；"
            "样本不足、单位混杂或缺失较多的字段已被排除。"
        ),
    }


def analyze_performance_conflicts(
    dataset: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    message: str = "",
) -> dict[str, Any]:
    requested = _requested_targets(dataset, args)
    targets = requested or _default_targets(dataset, count=2)
    if len(targets) < 2:
        return _field_error(dataset, "performance_conflict_analysis", "至少需要两个可分析性能字段")
    left, right = targets[:2]
    target_pairs = _pairs(dataset, left, right)
    if len(target_pairs) < 3:
        return _insufficient(dataset, "performance_conflict_analysis", f"{left} + {right}")
    target_correlation = _correlation(target_pairs)
    conflicts: list[dict[str, Any]] = []
    for feature in dataset.get("feature_fields") or []:
        left_result = _correlation(_pairs(dataset, feature, left))
        right_result = _correlation(_pairs(dataset, feature, right))
        if left_result["constant"] or right_result["constant"]:
            continue
        if len(left_result["pairs"]) < 3 or len(right_result["pairs"]) < 3:
            continue
        if left_result["correlation"] * right_result["correlation"] >= 0:
            continue
        if min(abs(left_result["correlation"]), abs(right_result["correlation"])) < 0.2:
            continue
        conflicts.append(
            {
                "field": feature,
                "field_label": _field_label(dataset, feature),
                "unit": _field_unit(dataset, feature),
                "left_target": left,
                "left_correlation": _round(left_result["correlation"]),
                "right_target": right,
                "right_correlation": _round(right_result["correlation"]),
                "evidence_samples": sorted(
                    {
                        row["sample_id"]
                        for row in _rows_with_target(dataset, left)
                        if feature in row["values"]
                    }
                )[:20],
            }
        )
    conflicts.sort(
        key=lambda item: -min(
            abs(float(item["left_correlation"] or 0)),
            abs(float(item["right_correlation"] or 0)),
        )
    )
    return {
        "status": "ok",
        "analysis_type": "performance_conflict_analysis",
        "scenario_id": 10,
        "targets": [left, right],
        "target_labels": [_field_label(dataset, left), _field_label(dataset, right)],
        "target_units": [_field_unit(dataset, left), _field_unit(dataset, right)],
        "target_correlation": _round(target_correlation["correlation"]),
        "target_paired_sample_count": len(target_pairs),
        "conflicts": conflicts[:30],
        "conflict_count": len(conflicts),
        "dataset": dataset,
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": dataset.get("warnings", []),
        "conclusion_limit": (
            "冲突结论基于历史样本相关性；相关性不证明因果，也不代表继续调整必然得到相同结果。"
        ),
    }


def analyze_batch_differences(
    dataset: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    message: str = "",
) -> dict[str, Any]:
    explicit_groups = _explicit_sample_groups(dataset, args)
    if explicit_groups is not None:
        groups, group_field = explicit_groups
    else:
        group_field = str(args.get("batch_field") or args.get("group_field") or "project_id")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset.get("rows") or []:
            value = row.get(group_field)
            if value is None or str(value).strip() == "":
                continue
            grouped[str(value)].append(row)
        groups = dict(grouped)
    groups = {
        str(name): rows for name, rows in groups.items() if rows
    }
    if len(groups) < 2:
        return {
            "status": "group_definition_required",
            "analysis_type": "batch_difference_analysis",
            "scenario_id": 11,
            "group_field": group_field,
            "available_group_fields": ["project_id", "sample_type", "sample_name"],
            "dataset": dataset,
            "evidence_refs": dataset.get("sample_refs", []),
            "warnings": [
                *dataset.get("warnings", []),
                "当前数据无法形成至少两个非空批次组；系统不把样品名称相似或数值差异自动解释为批次归属。",
            ],
        }
    names = sorted(groups)[:2]
    left_name, right_name = names
    left_rows, right_rows = groups[left_name], groups[right_name]
    all_fields = [*(dataset.get("feature_fields") or []), *(dataset.get("target_fields") or [])]
    differences: list[dict[str, Any]] = []
    for field in all_fields:
        left_values = [row["values"][field] for row in left_rows if field in row["values"]]
        right_values = [row["values"][field] for row in right_rows if field in row["values"]]
        if len(left_values) < 2 or len(right_values) < 2:
            continue
        left_mean = sum(left_values) / len(left_values)
        right_mean = sum(right_values) / len(right_values)
        pooled = _pooled_std(left_values, right_values)
        differences.append(
            {
                "field": field,
                "field_label": _field_label(dataset, field),
                "unit": _field_unit(dataset, field),
                "left_group": left_name,
                "left_sample_count": len(left_values),
                "left_mean": _round(left_mean),
                "right_group": right_name,
                "right_sample_count": len(right_values),
                "right_mean": _round(right_mean),
                "mean_difference": _round(right_mean - left_mean),
                "standardized_difference": _round(
                    (right_mean - left_mean) / pooled if pooled else None
                ),
            }
        )
    differences.sort(
        key=lambda item: -abs(float(item["standardized_difference"] or 0))
    )
    return {
        "status": "ok",
        "analysis_type": "batch_difference_analysis",
        "scenario_id": 11,
        "group_field": group_field,
        "groups": {
            name: [row["sample_id"] for row in rows]
            for name, rows in groups.items()
        },
        "compared_groups": [left_name, right_name],
        "left_sample_count": len(left_rows),
        "right_sample_count": len(right_rows),
        "differences": differences[:40],
        "difference_count": len(differences),
        "dataset": dataset,
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": [
            *dataset.get("warnings", []),
            "批次差异只比较当前授权样本和数值字段；组别不自动等价于正常/异常，原因需结合工艺记录和文档证据复核。",
        ],
    }


def discover_process_window(
    dataset: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    message: str = "",
) -> dict[str, Any]:
    filters = [item for item in args.get("filters") or [] if isinstance(item, Mapping)]
    if not filters:
        return _field_error(dataset, "process_window_discovery", "缺少目标性能条件")
    bindings = []
    for filter_item in filters:
        field = str(filter_item.get("field") or filter_item.get("target_metric") or "").strip()
        target = _resolve_field(dataset, field, sections=_TARGET_SECTIONS)
        if not target:
            return _field_error(
                dataset,
                "process_window_discovery",
                f"目标性能字段不存在或无法唯一绑定：{field}",
            )
        unit = str(filter_item.get("unit") or "").strip() or None
        observed_unit = _field_unit(dataset, target)
        if unit and observed_unit and unit != observed_unit:
            return {
                "status": "unit_mismatch",
                "analysis_type": "process_window_discovery",
                "scenario_id": 9,
                "field": target,
                "requested_unit": unit,
                "observed_unit": observed_unit,
                "dataset": dataset,
                "evidence_refs": dataset.get("sample_refs", []),
                "warnings": ["窗口发现不做单位猜测或自动换算。"],
            }
        bindings.append({**filter_item, "field": target, "unit": unit or observed_unit})

    feasible: list[dict[str, Any]] = []
    failures = Counter()
    for row in dataset.get("rows") or []:
        reasons = []
        for binding in bindings:
            value = row["values"].get(binding["field"])
            if value is None:
                reasons.append(f"{binding['field']}_missing")
                continue
            if not _filter_pass(value, binding):
                reasons.append(f"{binding['field']}_not_satisfied")
        if reasons:
            failures.update({reason: 1 for reason in reasons})
        else:
            feasible.append(row)
    if not feasible:
        return {
            "status": "no_feasible_samples",
            "analysis_type": "process_window_discovery",
            "scenario_id": 9,
            "filters": bindings,
            "failure_counts": dict(failures),
            "dataset": dataset,
            "evidence_refs": dataset.get("sample_refs", []),
            "warnings": ["授权样本中没有同时满足全部条件的记录；不外推虚拟窗口。"],
        }
    if len(feasible) < 3:
        return {
            "status": "insufficient_feasible_samples",
            "analysis_type": "process_window_discovery",
            "scenario_id": 9,
            "filters": bindings,
            "feasible_sample_count": len(feasible),
            "minimum_samples": 3,
            "dataset": dataset,
            "evidence_refs": dataset.get("sample_refs", []),
            "warnings": ["可行样本少于 3 条，不足以给出稳定窗口。"],
        }

    windows: list[dict[str, Any]] = []
    for field in dataset.get("feature_fields") or []:
        values = [row["values"][field] for row in feasible if field in row["values"]]
        if len(values) < 3:
            continue
        q1, q3 = _quartiles(values)
        windows.append(
            {
                "field": field,
                "field_label": _field_label(dataset, field),
                "unit": _field_unit(dataset, field),
                "sample_count": len(values),
                "observed_min": _round(min(values)),
                "observed_max": _round(max(values)),
                "recommended_min": _round(q1),
                "recommended_max": _round(q3),
                "robustness": round(len(values) / len(feasible), 4),
            }
        )
    windows.sort(
        key=lambda item: (-int(item["sample_count"]), str(item["field"]))
    )
    return {
        "status": "ok",
        "analysis_type": "process_window_discovery",
        "scenario_id": 9,
        "filters": bindings,
        "feasible_sample_count": len(feasible),
        "feasible_sample_ids": [row["sample_id"] for row in feasible],
        "windows": windows[:50],
        "window_count": len(windows),
        "dataset": dataset,
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": [
            *dataset.get("warnings", []),
            "推荐区间是满足条件的历史样本四分位区间，不是外推可行域；生产使用前必须复核约束和安全边界。",
        ],
    }


def build_stage_report(
    dataset: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    message: str = "",
) -> dict[str, Any]:
    projects = Counter(
        str(row.get("project_id")) for row in dataset.get("rows") or []
        if row.get("project_id") is not None
    )
    observed_times = sorted(
        row.get("observed_at") for row in dataset.get("rows") or []
        if row.get("observed_at")
    )
    target_statistics = []
    for field in dataset.get("target_fields") or []:
        values = [
            row["values"][field]
            for row in dataset.get("rows") or []
            if field in row["values"]
        ]
        if len(values) < 2:
            continue
        target_statistics.append(
            {
                "field": field,
                "field_label": _field_label(dataset, field),
                "unit": _field_unit(dataset, field),
                "sample_count": len(values),
                "mean": _round(sum(values) / len(values)),
                "min": _round(min(values)),
                "max": _round(max(values)),
            }
        )
    target_statistics.sort(key=lambda item: -int(item["sample_count"]))
    feature_coverage = []
    for field in dataset.get("feature_fields") or []:
        count = sum(field in row.get("values", {}) for row in dataset.get("rows") or [])
        feature_coverage.append(
            {
                "field": field,
                "field_label": _field_label(dataset, field),
                "unit": _field_unit(dataset, field),
                "sample_count": count,
                "coverage": round(count / max(1, int(dataset.get("sample_count") or 0)), 4),
            }
        )
    feature_coverage.sort(key=lambda item: (-int(item["sample_count"]), str(item["field"])))
    return {
        "status": "ok",
        "analysis_type": "stage_report_analysis",
        "scenario_id": 18,
        "scope": {
            "sample_count": dataset.get("sample_count", 0),
            "total_matching_sample_count": dataset.get("total_matching_sample_count"),
            "project_counts": dict(sorted(projects.items())),
            "observed_time_min": observed_times[0] if observed_times else None,
            "observed_time_max": observed_times[-1] if observed_times else None,
            "scan_complete": dataset.get("scan_complete", True),
            "scan_truncated": dataset.get("scan_truncated", False),
        },
        "target_statistics": target_statistics[:40],
        "feature_coverage": feature_coverage[:40],
        "dataset": dataset,
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": dataset.get("warnings", []),
        "conclusion_limit": (
            "阶段总结仅覆盖当前授权样本和资料证据；不推断未记录实验，不替代研发审批结论。"
        ),
    }


def research_analysis_chart_data(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    analysis_type = result.get("analysis_type")
    if analysis_type == "key_variable_analysis":
        return [
            {
                "chart_type": "bar",
                "title": "目标性能相关变量排名",
                "x_field": "field_label",
                "y_field": "correlation",
                "records": result.get("variables") or [],
            }
        ]
    if analysis_type == "performance_conflict_analysis":
        return [
            {
                "chart_type": "table",
                "title": "性能冲突变量",
                "records": result.get("conflicts") or [],
            }
        ]
    if analysis_type == "batch_difference_analysis":
        return [
            {
                "chart_type": "bar",
                "title": "批次标准化差异",
                "x_field": "field_label",
                "y_field": "standardized_difference",
                "records": result.get("differences") or [],
            }
        ]
    if analysis_type == "process_window_discovery":
        return [
            {
                "chart_type": "range",
                "title": "历史可行样本窗口",
                "records": result.get("windows") or [],
                "min_field": "recommended_min",
                "max_field": "recommended_max",
            }
        ]
    if analysis_type == "stage_report_analysis":
        return [
            {
                "chart_type": "table",
                "title": "性能结果概览",
                "records": result.get("target_statistics") or [],
            },
            {
                "chart_type": "bar",
                "title": "配方与工艺字段覆盖度",
                "x_field": "field_label",
                "y_field": "coverage",
                "records": result.get("feature_coverage") or [],
            },
        ]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return float(number)


def _normalize_field(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _resolve_field(
    dataset: Mapping[str, Any],
    requested: str,
    *,
    sections: tuple[str, ...],
) -> str | None:
    keys = [
        str(item["key"])
        for item in dataset.get("field_catalog") or []
        if item.get("section") in sections
    ]
    exact = [key for key in keys if _normalize_field(key) == _normalize_field(requested)]
    if len(exact) == 1:
        return exact[0]
    tail = [
        key for key in keys
        if _normalize_field(key).endswith("." + _normalize_field(requested))
    ]
    if len(tail) == 1:
        return tail[0]
    contains = [
        key for key in keys
        if _normalize_field(requested) in _normalize_field(key)
        or _normalize_field(key) in _normalize_field(requested)
    ]
    return contains[0] if len(contains) == 1 else None


def _requested_target(dataset: Mapping[str, Any], args: Mapping[str, Any]) -> str | None:
    targets = _requested_targets(dataset, args)
    return targets[0] if targets else None


def _requested_targets(
    dataset: Mapping[str, Any],
    args: Mapping[str, Any],
) -> list[str]:
    raw = args.get("target_metrics")
    if isinstance(raw, str):
        candidates = re.split(r"[，,、/]|和|与", raw)
    elif isinstance(raw, (list, tuple)):
        candidates = list(raw)
    elif args.get("target_metric"):
        candidates = [args["target_metric"]]
    else:
        candidates = []
    resolved = []
    for candidate in candidates:
        field = _resolve_field(dataset, str(candidate), sections=_TARGET_SECTIONS)
        if field and field not in resolved:
            resolved.append(field)
    return resolved


def _default_targets(dataset: Mapping[str, Any], *, count: int) -> list[str]:
    fields = sorted(
        dataset.get("target_fields") or [],
        key=lambda field: (-_numeric_count(dataset, field), field),
    )
    return fields[:count]


def _numeric_count(dataset: Mapping[str, Any], field: str) -> int:
    return sum(field in row.get("values", {}) for row in dataset.get("rows") or [])


def _rows_with_target(
    dataset: Mapping[str, Any], target: str
) -> list[dict[str, Any]]:
    return [row for row in dataset.get("rows") or [] if target in row["values"]]


def _pairs(
    dataset: Mapping[str, Any],
    left: str,
    right: str,
) -> list[tuple[float, float]]:
    return [
        (row["values"][left], row["values"][right])
        for row in dataset.get("rows") or []
        if left in row["values"] and right in row["values"]
    ]


def _correlation(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if len(pairs) < 2:
        return {"correlation": 0.0, "constant": True, "pairs": pairs}
    left_values = [item[0] for item in pairs]
    right_values = [item[1] for item in pairs]
    if _std(left_values) == 0 or _std(right_values) == 0:
        return {"correlation": 0.0, "constant": True, "pairs": pairs}
    covariance = sum(
        (left - sum(left_values) / len(left_values))
        * (right - sum(right_values) / len(right_values))
        for left, right in pairs
    )
    denominator = math.sqrt(
        sum((left - sum(left_values) / len(left_values)) ** 2 for left in left_values)
        * sum((right - sum(right_values) / len(right_values)) ** 2 for right in right_values)
    )
    correlation = covariance / denominator if denominator else 0.0
    return {
        "correlation": max(-1.0, min(1.0, correlation)),
        "constant": False,
        "pairs": pairs,
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _pooled_std(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((value - left_mean) ** 2 for value in left) + sum(
        (value - right_mean) ** 2 for value in right
    )
    denominator = max(1, len(left) + len(right) - 2)
    return math.sqrt(numerator / denominator)


def _quartiles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
    return percentile(0.25), percentile(0.75)


def _filter_pass(value: float, binding: Mapping[str, Any]) -> bool:
    operator = str(binding.get("operator") or "").lower()
    target = binding.get("value")
    values = binding.get("values") or []
    if operator == "gt":
        return value > float(target)
    if operator == "gte":
        return value >= float(target)
    if operator == "lt":
        return value < float(target)
    if operator == "lte":
        return value <= float(target)
    if operator == "between":
        return float(values[0]) <= value <= float(values[1])
    return False


def _explicit_sample_groups(
    dataset: Mapping[str, Any], args: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], str] | None:
    raw = args.get("batch_groups")
    if not isinstance(raw, Mapping):
        return None
    by_id = {str(row["sample_id"]): row for row in dataset.get("rows") or []}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, ids in raw.items():
        if not isinstance(ids, (list, tuple)):
            continue
        for sample_id in ids:
            row = by_id.get(str(sample_id))
            if row is not None:
                groups[str(name)].append(row)
    return groups, "explicit_sample_ids"


def _field_unit(dataset: Mapping[str, Any], field: str) -> str | None:
    for item in dataset.get("field_catalog") or []:
        if item.get("key") == field:
            unit = item.get("unit")
            return None if unit == "MIXED" else str(unit) if unit else None
    return None


def _field_label(dataset: Mapping[str, Any], field: str) -> str:
    for item in dataset.get("field_catalog") or []:
        if item.get("key") == field:
            return f"{item.get('section_label')}.{item.get('name')}"
    return field


def _field_error(
    dataset: Mapping[str, Any],
    analysis_type: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "field_not_found",
        "analysis_type": analysis_type,
        "dataset": dataset,
        "available_target_fields": dataset.get("target_fields", []),
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": [reason, *dataset.get("warnings", [])],
    }


def _insufficient(
    dataset: Mapping[str, Any],
    analysis_type: str,
    target: str,
) -> dict[str, Any]:
    return {
        "status": "insufficient_data",
        "analysis_type": analysis_type,
        "target": target,
        "dataset": dataset,
        "evidence_refs": dataset.get("sample_refs", []),
        "warnings": [
            "有效配对样本少于 3 条，不足以做稳定统计分析。",
            *dataset.get("warnings", []),
        ],
    }


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
