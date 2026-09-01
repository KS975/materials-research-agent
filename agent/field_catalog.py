from __future__ import annotations

import re
from typing import Any


CATALOG_SCHEMA_VERSION = "2B-1.1"
_DYNAMIC_SECTIONS = ("formula", "process", "performance")
_SECTION_LABELS = {
    "sample": "样品基础字段",
    "formula": "配方字段",
    "process": "工艺字段",
    "performance": "性能字段",
    "conditions": "测试条件字段",
}
_SAMPLE_FIELDS = ("id", "name", "project_id", "sample_type", "create_time")
_FIELD_PREFIXES = (
    "配方", "原料", "组分", "工艺", "过程", "性能", "指标", "测试条件",
)
_FIELD_SUFFIXES = (
    "含量", "添加量", "加入量", "用量", "比例", "占比", "份数",
    "工艺参数", "参数", "性能指标", "指标", "数值",
)


def normalize_field_name(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return re.sub(r"[\s·._\-—:：/\\()（）\[\]【】]+", "", text)


def _requested_name_variants(value: Any) -> set[str]:
    normalized = normalize_field_name(value)
    if not normalized:
        return set()
    variants = {normalized}
    changed = True
    while changed:
        changed = False
        for item in list(variants):
            for prefix in _FIELD_PREFIXES:
                key = normalize_field_name(prefix)
                if item.startswith(key) and len(item) > len(key):
                    candidate = item[len(key):]
                    if candidate and candidate not in variants:
                        variants.add(candidate)
                        changed = True
            for suffix in _FIELD_SUFFIXES:
                key = normalize_field_name(suffix)
                if item.endswith(key) and len(item) > len(key):
                    candidate = item[:-len(key)]
                    if candidate and candidate not in variants:
                        variants.add(candidate)
                        changed = True
    return variants


def build_material_field_catalog(source: dict[str, Any]) -> dict[str, Any]:
    """Build a value-free field catalogue from already authorized sample rows."""
    samples = list(source.get("samples") or [])
    entries: dict[str, dict[str, dict[str, Any]]] = {
        section: {} for section in ("sample", *_DYNAMIC_SECTIONS, "conditions")
    }
    for name in _SAMPLE_FIELDS:
        entries["sample"][normalize_field_name(name)] = {
            "name": name,
            "units": set(),
            "observed_sample_count": 0,
        }

    unresolved_count = 0
    for sample_item in samples:
        sample = sample_item.get("sample") or {}
        for name in _SAMPLE_FIELDS:
            if sample.get(name) not in (None, ""):
                entries["sample"][normalize_field_name(name)][
                    "observed_sample_count"
                ] += 1

        for section in _DYNAMIC_SECTIONS:
            seen_in_sample: set[str] = set()
            for item in sample_item.get(section) or []:
                name = str(item.get("name") or "").strip()
                if not name:
                    unresolved_count += 1
                    continue
                normalized = normalize_field_name(name)
                if not normalized:
                    continue
                entry = entries[section].setdefault(normalized, {
                    "name": name,
                    "units": set(),
                    "observed_sample_count": 0,
                })
                unit = str(item.get("unit") or "").strip()
                if unit:
                    entry["units"].add(unit)
                if normalized not in seen_in_sample:
                    entry["observed_sample_count"] += 1
                    seen_in_sample.add(normalized)

        seen_conditions: set[str] = set()
        for key in (sample_item.get("conditions") or {}):
            name = str(key or "").strip()
            normalized = normalize_field_name(name)
            if not normalized:
                continue
            entry = entries["conditions"].setdefault(normalized, {
                "name": name,
                "units": set(),
                "observed_sample_count": 0,
            })
            if normalized not in seen_conditions:
                entry["observed_sample_count"] += 1
                seen_conditions.add(normalized)

    sections: dict[str, list[dict[str, Any]]] = {}
    for section, mapping in entries.items():
        sections[section] = sorted(
            (
                {
                    "name": item["name"],
                    "units": sorted(item["units"]),
                    "observed_sample_count": item["observed_sample_count"],
                }
                for item in mapping.values()
            ),
            key=lambda item: (normalize_field_name(item["name"]), item["name"]),
        )

    counts = {section: len(items) for section, items in sections.items()}
    return {
        "status": "ok",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "sections": sections,
        "field_counts": counts,
        "total_field_count": sum(counts.values()),
        "source_sample_count": len(samples),
        "source_total_matching_sample_count": source.get(
            "total_matches", source.get("count", len(samples))
        ),
        "scan_complete": bool(source.get("scan_complete", True)),
        "scan_truncated": bool(source.get("scan_truncated", False)),
        "unresolved_field_count": unresolved_count,
        "value_disclosure": False,
        "warnings": list(source.get("warnings") or []),
    }


def _catalog_entries(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for section, items in (catalog.get("sections") or {}).items():
        for item in items or []:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            result.append({
                "section": section,
                "name": name,
                "normalized": normalize_field_name(name),
                "units": list(item.get("units") or []),
            })
    return result


def bind_filters_to_catalog(
    filters: list[dict[str, Any]],
    catalog: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind model field text to one authoritative catalogue entry.

    Returns ``(filters, bindings, errors)``. No fuzzy distance or model guess is
    used: only exact normalized names and conservative affix removal are allowed.
    """
    if not catalog or catalog.get("status") != "ok":
        return [dict(item) for item in filters], [], []

    entries = _catalog_entries(catalog)
    bound: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, original in enumerate(filters, 1):
        spec = dict(original)
        section = spec.get("section")
        field = str(spec.get("field") or "").strip()
        if section == "sample" or (section == "conditions" and field == "*"):
            bound.append(spec)
            continue

        exact_key = normalize_field_name(field)
        exact = [item for item in entries if item["normalized"] == exact_key]
        if len(exact) == 1:
            candidates = exact
            method = "exact_unique"
        elif len(exact) > 1:
            candidates = exact
            method = "ambiguous_exact"
        else:
            variants = _requested_name_variants(field)
            alias_matches = [item for item in entries if item["normalized"] in variants]
            if len(alias_matches) == 1:
                candidates = alias_matches
                method = "conservative_alias"
            else:
                candidates = alias_matches
                method = "ambiguous_alias" if alias_matches else "not_found"

        if len(candidates) != 1:
            errors.append({
                "filter_index": index,
                "code": "ambiguous_field" if candidates else "field_not_found",
                "requested_section": section,
                "requested_field": field,
                "section": section,
                "field": field,
                "candidates": [
                    {
                        "section": item["section"],
                        "field": item["name"],
                        "units": item["units"],
                    }
                    for item in candidates[:12]
                ],
            })
            continue

        selected = candidates[0]
        spec["section"] = selected["section"]
        spec["field"] = selected["name"]
        bound.append(spec)
        if selected["section"] != section or selected["name"] != field:
            bindings.append({
                "filter_index": index,
                "requested_section": section,
                "requested_field": field,
                "canonical_section": selected["section"],
                "canonical_field": selected["name"],
                "method": method,
            })

    if errors:
        return [], bindings, errors
    return bound, bindings, []


def field_catalog_for_prompt(
    catalog: dict[str, Any] | None,
    message: str,
    *,
    max_total_fields: int = 240,
) -> dict[str, Any]:
    """Return a compact, value-free catalogue for the intent-model prompt."""
    if not catalog or catalog.get("status") != "ok":
        return {}
    message_key = normalize_field_name(message)
    requested_variants = _requested_name_variants(message)
    ranked = []
    for section, items in (catalog.get("sections") or {}).items():
        for item in items or []:
            name = str(item.get("name") or "").strip()
            key = normalize_field_name(name)
            relevant = bool(
                key
                and (
                    key in message_key
                    or any(variant and key in variant for variant in requested_variants)
                )
            )
            ranked.append((0 if relevant else 1, section, key, name, item))
    ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    selected = ranked[: max(1, min(max_total_fields, 500))]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, section, _, name, item in selected:
        grouped.setdefault(section, []).append({
            "field": name,
            "units": list(item.get("units") or []),
        })
    return {
        "schema_version": catalog.get("schema_version"),
        "authoritative": True,
        "contains_values": False,
        "section_labels": _SECTION_LABELS,
        "sections": grouped,
        "selected_field_count": len(selected),
        "total_field_count": catalog.get("total_field_count", len(ranked)),
        "truncated": len(selected) < len(ranked),
        "instruction": (
            "字段类别和 canonical field 必须从本目录选择；用户口语如“PC含量”"
            "应绑定为目录中的“PC”，不得自行创造字段名。"
        ),
    }
