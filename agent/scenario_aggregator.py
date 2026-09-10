from __future__ import annotations

import json
from typing import Any, Mapping


def aggregate_scenario_result(
    scenario_id: int,
    mysql_result: Mapping[str, Any] | None,
    vector_result: Mapping[str, Any] | None = None,
    analysis_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, scenario-specific structured summary for the LLM.

    The aggregator is deterministic and never invents values. It extracts the
    highest-value fields from tool output and drops low-value metadata such as
    raw record references and timestamps.
    """
    result = dict(mysql_result or {})
    if scenario_id in {1, 2}:
        return _similarity_summary(scenario_id, result)
    if scenario_id == 3:
        return _filter_summary(result)
    if scenario_id == 4:
        return _profile_summary(result)
    if scenario_id in {5, 6}:
        return _material_summary(scenario_id, result)
    if scenario_id in {7, 12, 13}:
        return _failure_or_benchmark_summary(scenario_id, result)
    if scenario_id == 14:
        return _cold_start_summary(result)
    if scenario_id in {8, 9, 10, 11, 18}:
        return _analysis_summary(scenario_id, analysis_result or result)
    if scenario_id == 15:
        return _result_association_summary(result)
    if scenario_id == 16:
        return _spectrum_summary(result)
    if scenario_id == 19:
        return {"aggregated_type": "interface_reserved", "note": "实验回流暂未开放"}
    if scenario_id == 20:
        return _cross_project_summary(result)
    return _fallback_summary(result)


def _similarity_summary(scenario_id: int, result: Mapping[str, Any]) -> dict[str, Any]:
    ranking = list(result.get("ranking") or [])
    reference = result.get("reference_sample") or result.get("sample") or {}
    top = []
    for item in ranking[:8]:
        sample = item.get("sample") or {}
        row = {
            "sample_id": sample.get("id") or sample.get("name"),
            "name": sample.get("name"),
            "similarity": item.get("similarity_percent") or item.get("score"),
            "degrade_level": item.get("degrade_level"),
            "section_details": item.get("section_details") or item.get("details") or {},
        }
        for section in ("formula", "process", "performance", "service_performance"):
            fields = sample.get(section) or item.get(section) or []
            if fields:
                row[section] = _compact_fields(fields)
        top.append(row)
    return {
        "aggregated_type": "similarity_ranking",
        "reference_sample": {
            "id": reference.get("id") or reference.get("name"),
            "name": reference.get("name"),
            "formula": _compact_fields(
                result.get("reference_formula") or result.get("formula")
            ),
            "process": _compact_fields(
                result.get("reference_process") or result.get("process")
            ),
            "performance": _compact_fields(
                result.get("reference_performance") or result.get("performance")
            ),
        },
        "ranking": top,
        "ranking_count": len(ranking),
        "comparable_candidate_count": result.get("comparable_candidate_count", 0),
        "degrade_level": result.get("degrade_level"),
        "missing_fields": result.get("missing_fields") or [],
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _filter_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(result.get("samples") or result.get("matched_samples") or [])
    filters = result.get("filters") or []
    top = []
    for item in samples[:10]:
        sample = item.get("sample") if isinstance(item, Mapping) else item
        if not isinstance(sample, Mapping):
            continue
        top.append({
            "sample_id": sample.get("id") or sample.get("name"),
            "name": sample.get("name"),
            "project_id": sample.get("project_id"),
            "formula": _compact_fields(item.get("formula")),
            "performance": _compact_fields(item.get("performance")),
            "process": _compact_fields(item.get("process")),
        })
    return {
        "aggregated_type": "performance_filter",
        "filters": filters,
        "matched_count": result.get("count") or result.get("matched_count") or len(samples),
        "top_candidates": top,
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _profile_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "sample_profile",
        "sample": {
            "id": (result.get("sample") or {}).get("id"),
            "name": (result.get("sample") or {}).get("name"),
            "project_id": (result.get("sample") or {}).get("project_id"),
            "sample_type": (result.get("sample") or {}).get("sample_type"),
            "describe": (result.get("sample") or {}).get("describe"),
        },
        "formula": _compact_fields(result.get("formula")),
        "process": _compact_fields(result.get("process")),
        "performance": _compact_fields(result.get("performance")),
        "service_performance": _compact_fields(result.get("service_performance")),
        "synthesis_count": len(result.get("synthesis_records") or []),
        "verify_count": len(result.get("verify_items") or []),
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _material_summary(scenario_id: int, result: Mapping[str, Any]) -> dict[str, Any]:
    matched = list(result.get("matched_samples") or [])
    top = []
    for item in matched[:10]:
        sample = item.get("sample") if isinstance(item, Mapping) else item
        if not isinstance(sample, Mapping):
            continue
        top.append({
            "sample_id": sample.get("id") or sample.get("name"),
            "name": sample.get("name"),
            "formula": _compact_fields(item.get("formula") or item.get("matched_fields")),
            "performance": _compact_fields(item.get("performance")),
            "interpretation": item.get("interpretation"),
        })
    return {
        "aggregated_type": "material_usage" if scenario_id == 5 else "material_substitution",
        "material_name": result.get("material_name"),
        "original_material": result.get("original_material"),
        "replacement_material": result.get("replacement_material"),
        "matched_count": result.get("matched_count") or result.get("count") or len(matched),
        "truncated": result.get("truncated", False),
        "top_candidates": top,
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _failure_or_benchmark_summary(scenario_id: int, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "failure_case" if scenario_id == 7 else "anomaly" if scenario_id == 12 else "benchmark",
        "candidate_count": result.get("count") or len(result.get("samples") or []),
        "top_candidates": _compact_candidates(result.get("samples") or result.get("matched_samples")),
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _cold_start_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "cold_start",
        "filters": result.get("filters") or [],
        "reference_candidates": _compact_candidates(result.get("samples") or result.get("matched_samples")),
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _analysis_summary(scenario_id: int, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "analysis",
        "scenario_id": scenario_id,
        "variables": result.get("variables") or result.get("key_variables"),
        "conflicts": result.get("conflicts"),
        "differences": result.get("differences"),
        "windows": result.get("windows"),
        "sample_count": (result.get("dataset") or {}).get("sample_count"),
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _result_association_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "result_association",
        "matches": result.get("matches") or [],
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _spectrum_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "spectrum_features",
        "feature_count": result.get("feature_count") or len(result.get("features") or []),
        "features": (result.get("features") or [])[:20],
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _cross_project_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "cross_project_assets",
        "project_count": result.get("project_count") or len(result.get("projects") or []),
        "projects": result.get("projects") or [],
        "status": result.get("status"),
        "warnings": _compact_warnings(result),
    }


def _fallback_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregated_type": "generic",
        "status": result.get("status"),
        "samples": _compact_candidates(result.get("samples") or result.get("matched_samples")),
        "warnings": _compact_warnings(result),
    }


def _compact_fields(fields: Any) -> list[dict[str, Any]]:
    if not isinstance(fields, list):
        return []
    return [
        {
            "name": f.get("name"),
            "value": f.get("value"),
            "unit": f.get("unit"),
        }
        for f in fields[:15]
        if (
            isinstance(f, Mapping)
            and f.get("resolved", bool(f.get("name")))
            and f.get("value") is not None
            and str(f.get("value") or "").strip() != ""
        )
    ]


def _compact_candidates(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    top = []
    for item in items[:10]:
        sample = item.get("sample") if isinstance(item, Mapping) else item
        if not isinstance(sample, Mapping):
            continue
        top.append({
            "sample_id": sample.get("id") or sample.get("name"),
            "name": sample.get("name"),
            "formula": _compact_fields(item.get("formula")),
            "performance": _compact_fields(item.get("performance")),
        })
    return top


def _compact_warnings(result: Mapping[str, Any]) -> list[str]:
    warnings = list(result.get("warnings") or [])
    return [str(w) for w in warnings[:5] if str(w).strip()]


def serialize_aggregated_summary(summary: Mapping[str, Any]) -> str:
    """Compact JSON string for embedding as a derived evidence record."""
    return json.dumps(
        summary, ensure_ascii=False, sort_keys=True, default=str
    )


def build_data_cards(
    summary: Mapping[str, Any] | None,
    scenario_id: int,
) -> list[dict[str, Any]]:
    """Derive front-end data cards from the aggregated summary.

    Cards reuse the same deterministic summary that feeds the LLM so the
    narrative and the visible data never diverge.
    """
    if not isinstance(summary, Mapping):
        return []
    aggregated = str(summary.get("aggregated_type") or "")
    if aggregated == "similarity_ranking":
        return [_similarity_card(summary)]
    if aggregated == "performance_filter":
        return [_filter_card(summary)]
    if aggregated in {"material_usage", "material_substitution"}:
        return [_material_card(summary)]
    if aggregated == "sample_profile":
        return [_profile_card(summary)]
    if aggregated == "analysis":
        return [_analysis_card(summary)]
    if aggregated == "cross_project_assets":
        return [_cross_project_card(summary)]
    if aggregated == "spectrum_features":
        return [_spectrum_card(summary)]
    if aggregated == "result_association":
        return [_association_card(summary)]
    if aggregated in {"failure_case", "anomaly", "benchmark", "cold_start"}:
        return [_candidate_card(aggregated, summary)]
    return []


def _card_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "sample_name": row.get("name") or row.get("sample_name"),
        "similarity": row.get("similarity"),
        "degrade_level": row.get("degrade_level"),
        "formula": row.get("formula") or [],
        "process": row.get("process") or [],
        "performance": row.get("performance") or [],
        "status": row.get("status"),
        "recommendation": _recommendation(row),
    }


def _recommendation(row: Mapping[str, Any]) -> str:
    if row.get("degrade_level"):
        return "谨慎使用"
    raw = row.get("similarity")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return "高性能备选"
    if score >= 90:
        return "首选复用"
    if score >= 70:
        return "高性能备选"
    if score >= 50:
        return "谨慎使用"
    return "不推荐"


def _similarity_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [_card_item(row) for row in (summary.get("ranking") or [])[:5]]
    return {
        "card_type": "sample_list",
        "title": f"相似配方 Top{len(items)}",
        "reference": summary.get("reference_sample") or {},
        "items": items,
        "degrade_level": summary.get("degrade_level"),
        "missing_fields": summary.get("missing_fields") or [],
        "warnings": summary.get("warnings") or [],
    }


def _filter_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [_card_item(row) for row in (summary.get("top_candidates") or [])[:5]]
    return {
        "card_type": "sample_list",
        "title": f"满足条件样品 Top{len(items)}",
        "filters": summary.get("filters") or [],
        "matched_count": summary.get("matched_count"),
        "items": items,
        "warnings": summary.get("warnings") or [],
    }


def _material_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [_card_item(row) for row in (summary.get("top_candidates") or [])[:5]]
    return {
        "card_type": "sample_list",
        "title": "原料相关样品",
        "material_name": summary.get("material_name"),
        "original_material": summary.get("original_material"),
        "replacement_material": summary.get("replacement_material"),
        "matched_count": summary.get("matched_count"),
        "truncated": summary.get("truncated"),
        "items": items,
        "warnings": summary.get("warnings") or [],
    }


def _profile_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_type": "sample_detail",
        "title": f"样品画像 {(summary.get('sample') or {}).get('name') or ''}".strip(),
        "sample": summary.get("sample") or {},
        "formula": summary.get("formula") or [],
        "process": summary.get("process") or [],
        "performance": summary.get("performance") or [],
        "service_performance": summary.get("service_performance") or [],
        "synthesis_count": summary.get("synthesis_count"),
        "verify_count": summary.get("verify_count"),
        "warnings": summary.get("warnings") or [],
    }


def _analysis_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[list[Any]] = []
    for item in (summary.get("variables") or [])[:10]:
        if isinstance(item, Mapping):
            rows.append([
                item.get("field") or item.get("name"),
                item.get("correlation") or item.get("value"),
                item.get("sample_count"),
            ])
    for item in (summary.get("conflicts") or [])[:10]:
        if isinstance(item, Mapping):
            rows.append([
                item.get("field") or item.get("name"),
                item.get("direction") or item.get("correlation"),
                item.get("sample_count"),
            ])
    return {
        "card_type": "metric_table",
        "title": "确定性分析结果",
        "columns": ["字段", "指标", "样本数"],
        "rows": rows,
        "sample_count": summary.get("sample_count"),
        "warnings": summary.get("warnings") or [],
    }


def _cross_project_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_type": "metric_table",
        "title": "跨项目可复用资产",
        "columns": ["项目", "样品数", "资产", "权限"],
        "rows": [
            [
                project.get("project_id"),
                project.get("sample_count"),
                ", ".join(
                    str(asset.get("asset_type"))
                    for asset in (project.get("assets") or [])
                    if isinstance(asset, Mapping)
                ),
                "只读",
            ]
            for project in (summary.get("projects") or [])
            if isinstance(project, Mapping)
        ],
        "warnings": summary.get("warnings") or [],
    }


def _spectrum_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_type": "metric_table",
        "title": "图谱/曲线结构化特征",
        "columns": ["样品", "特征字段", "值"],
        "rows": [
            [item.get("sample_id"), item.get("field"), item.get("value")]
            for item in (summary.get("features") or [])
            if isinstance(item, Mapping)
        ],
        "warnings": summary.get("warnings") or [],
    }


def _association_card(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_type": "metric_table",
        "title": "检测结果关联建议",
        "columns": ["输入编号", "匹配状态", "匹配样品", "确认"],
        "rows": [
            [
                item.get("identifier"),
                item.get("match_status"),
                ((item.get("matched_sample") or {}) if isinstance(item, Mapping) else {}).get("name")
                or "-",
                "待确认" if item.get("confirmation_required") else "-",
            ]
            for item in (summary.get("matches") or [])
            if isinstance(item, Mapping)
        ],
        "warnings": summary.get("warnings") or [],
    }


def _candidate_card(kind: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = summary.get("top_candidates") or summary.get("reference_candidates") or []
    items = [_card_item(row) for row in rows[:5] if isinstance(row, Mapping)]
    titles = {
        "failure_case": "失败/异常案例候选",
        "anomaly": "异常与失效案例候选",
        "benchmark": "竞品对标候选",
        "cold_start": "冷启动参考候选",
    }
    return {
        "card_type": "sample_list",
        "title": titles.get(kind, "候选样品"),
        "items": items,
        "warnings": summary.get("warnings") or [],
    }
