from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


MAX_FILTER_COUNT = 8
MAX_IN_VALUES = 20

ALLOWED_SECTIONS = {
    "sample",
    "formula",
    "process",
    "performance",
    "conditions",
}

SECTION_ALIASES = {
    "sample": "sample",
    "样品": "sample",
    "样本": "sample",
    "metadata": "sample",
    "formula": "formula",
    "配方": "formula",
    "原料": "formula",
    "material": "formula",
    "materials": "formula",
    "process": "process",
    "工艺": "process",
    "加工": "process",
    "performance": "performance",
    "性能": "performance",
    "指标": "performance",
    "conditions": "conditions",
    "condition": "conditions",
    "测试条件": "conditions",
    "条件": "conditions",
}

OPERATOR_ALIASES = {
    "eq": "eq",
    "=": "eq",
    "==": "eq",
    "等于": "eq",
    "ne": "ne",
    "!=": "ne",
    "不等于": "ne",
    "gt": "gt",
    ">": "gt",
    "大于": "gt",
    "高于": "gt",
    "gte": "gte",
    ">=": "gte",
    "≥": "gte",
    "大于等于": "gte",
    "不低于": "gte",
    "至少": "gte",
    "lt": "lt",
    "<": "lt",
    "小于": "lt",
    "低于": "lt",
    "lte": "lte",
    "<=": "lte",
    "≤": "lte",
    "小于等于": "lte",
    "不高于": "lte",
    "至多": "lte",
    "between": "between",
    "区间": "between",
    "介于": "between",
    "in": "in",
    "属于": "in",
    "contains": "contains",
    "包含": "contains",
    "exists": "exists",
    "not_null": "exists",
    "有记录": "exists",
    "非空": "exists",
    "missing": "missing",
    "is_null": "missing",
    "缺失": "missing",
    "未记录": "missing",
}

SAMPLE_FIELD_ALIASES = {
    "id": "id",
    "sample_id": "id",
    "样品id": "id",
    "样本id": "id",
    "name": "name",
    "sample_name": "name",
    "样品名称": "name",
    "样本名称": "name",
    "project_id": "project_id",
    "项目id": "project_id",
    "项目号": "project_id",
    "sample_type": "sample_type",
    "样品类型": "sample_type",
    "样本类型": "sample_type",
    "create_time": "create_time",
    "创建时间": "create_time",
}

_MULTI_CONDITION_MARKERS = (
    "大于", "小于", "高于", "低于", "不低于", "不高于",
    "至少", "至多", "介于", "等于", "不等于", "有记录",
    "未记录", "缺失", ">=", "<=", ">", "<", "＞", "＜",
)
_COLLECTION_MARKERS = (
    "样品", "样本", "实验", "找", "筛选", "列出", "查找", "搜索",
)


def looks_like_multi_condition_request(message: Any) -> bool:
    text = str(message or "").strip().casefold()
    return bool(text) and any(
        marker in text for marker in _MULTI_CONDITION_MARKERS
    ) and any(marker in text for marker in _COLLECTION_MARKERS)


def _short_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _safe_scalar(value: Any) -> str | int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float, Decimal)):
        if isinstance(value, str):
            return value.strip()[:200]
        if isinstance(value, Decimal):
            return str(value)
        return value
    return None


def _is_numeric_scalar(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        Decimal(str(value).strip())
        return True
    except (InvalidOperation, ValueError, TypeError):
        return False


def normalize_logic(raw: Any) -> str:
    value = str(raw or "and").strip().lower()
    return "or" if value in {"or", "any", "任一", "或者", "或"} else "and"


def normalize_unit(value: Any) -> str:
    """Normalize notation only; this intentionally performs no unit conversion."""
    text = str(value or "").strip().casefold()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("摄氏度", "°c").replace("℃", "°c").replace("ºc", "°c")
    text = text.replace("m²", "m2").replace("cm²", "cm2")
    text = text.replace("r_min", "r/min").replace("rpm", "r/min")
    text = "".join(text.split())
    return text


def unit_is_explicit_in_text(unit: Any, message: Any) -> bool:
    """Only keep a model-produced unit when the user actually wrote it."""
    normalized_unit = normalize_unit(unit)
    if not normalized_unit:
        return False
    normalized_message = normalize_unit(message)
    return normalized_unit in normalized_message


def normalize_filters(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate LLM filter JSON without ever accepting executable syntax."""
    if not isinstance(raw, list):
        return [], ["filters 必须是数组"]
    if not raw:
        return [], ["至少需要一个筛选条件"]
    if len(raw) > MAX_FILTER_COUNT:
        return [], [f"单次最多允许 {MAX_FILTER_COUNT} 个筛选条件"]

    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            errors.append(f"第{index}个条件必须是对象")
            continue

        section_raw = _short_text(
            item.get("section") or item.get("category"), 40
        ).lower()
        section = SECTION_ALIASES.get(section_raw)
        if section not in ALLOWED_SECTIONS:
            errors.append(f"第{index}个条件的字段类别无效")
            continue

        field = _short_text(item.get("field"), 120)
        if section == "conditions" and not field:
            field = "*"
        if not field:
            errors.append(f"第{index}个条件缺少字段名称")
            continue
        if section == "sample":
            field = SAMPLE_FIELD_ALIASES.get(field.casefold(), field)
            if field not in set(SAMPLE_FIELD_ALIASES.values()):
                errors.append(f"第{index}个条件不允许筛选该样品基础字段")
                continue

        operator_raw = _short_text(item.get("operator"), 40).lower()
        operator = OPERATOR_ALIASES.get(operator_raw)
        if operator is None:
            errors.append(f"第{index}个条件的比较运算符无效")
            continue
        if section == "conditions" and field == "*" and operator not in {
            "exists", "missing",
        }:
            errors.append(f"第{index}个测试条件整体筛选只支持 exists 或 missing")
            continue

        spec: dict[str, Any] = {
            "section": section,
            "field": field,
            "operator": operator,
        }
        unit = _short_text(item.get("unit"), 40)
        if unit:
            if section in {"sample", "conditions"}:
                errors.append(f"第{index}个条件不应为该字段类别指定单位")
                continue
            spec["unit"] = unit

        if operator in {"exists", "missing"}:
            normalized.append(spec)
            continue

        if operator in {"between", "in"}:
            values_raw = item.get("values")
            if values_raw is None and isinstance(item.get("value"), list):
                values_raw = item.get("value")
            if not isinstance(values_raw, list):
                errors.append(f"第{index}个条件需要 values 数组")
                continue
            if operator == "between" and len(values_raw) != 2:
                errors.append(f"第{index}个区间条件必须恰好有两个边界值")
                continue
            if operator == "in" and not (1 <= len(values_raw) <= MAX_IN_VALUES):
                errors.append(
                    f"第{index}个集合条件需要1到{MAX_IN_VALUES}个候选值"
                )
                continue
            values = [_safe_scalar(value) for value in values_raw]
            if any(value is None or value == "" for value in values):
                errors.append(f"第{index}个条件包含无效值")
                continue
            if operator == "between" and not all(
                _is_numeric_scalar(value) for value in values
            ):
                errors.append(f"第{index}个区间条件的边界必须是数值")
                continue
            spec["values"] = values
            normalized.append(spec)
            continue

        value = _safe_scalar(item.get("value"))
        if value is None or value == "":
            errors.append(f"第{index}个条件缺少比较值")
            continue
        if operator in {"gt", "gte", "lt", "lte"} and not _is_numeric_scalar(value):
            errors.append(f"第{index}个数值比较条件的值必须是数值")
            continue
        spec["value"] = value
        normalized.append(spec)

    return normalized, errors
