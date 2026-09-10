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
