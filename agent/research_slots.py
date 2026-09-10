from __future__ import annotations

import re
from typing import Any


_SUBSTITUTION_REVERSED = re.compile(
    r"用(?P<replacement>[^，。？?；;]{1,40}?)(?:替换|替代)"
    r"(?P<original>[^，。？?；;]{1,40})"
)
_SUBSTITUTION_INTO = re.compile(
    r"(?P<original>[^，。？?；;]{1,40}?)(?:替换成|替换为)"
    r"(?P<replacement>[^，。？?；;]{1,40}?)"
    r"(?:的|历史|记录|配方|性能|，|。|？|$)"
)
_SUBSTITUTION_BY = re.compile(
    r"(?P<replacement>[^，。？?；;]{1,40}?)(?:替代|替换)"
    r"(?P<original>[^，。？?；;]{1,40}?)"
    r"(?:的|历史|记录|配方|性能|，|。|？|$)"
)
_EXPLICIT_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|\d{2,})"
    r"(?![A-Za-z0-9_.-])"
)


def normalize_research_slots(
    *,
    message: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    result = dict(args)
    if not str(result.get("material_name") or "").strip():
        material = extract_material_name(message)
        if material:
            result["material_name"] = material
    has_substitution = bool(
        str(result.get("original_material") or "").strip()
        and str(result.get("replacement_material") or "").strip()
    )
    if not (
        str(result.get("original_material") or "").strip()
        and str(result.get("replacement_material") or "").strip()
    ):
        substitution = extract_substitution_materials(message)
        if substitution is not None:
            result["original_material"], result["replacement_material"] = substitution
            has_substitution = True
    if (
        str(result.get("original_material") or "").strip()
        and str(result.get("replacement_material") or "").strip()
    ):
        for key in ("identifier", "left_identifier", "right_identifier"):
            result.pop(key, None)
    if not str(result.get("competitor_name") or "").strip():
        competitor = extract_competitor_name(message)
        if competitor:
            result["competitor_name"] = competitor
    if not str(result.get("phenomenon") or "").strip():
        phenomenon = extract_phenomenon(message)
        if phenomenon:
            result["phenomenon"] = phenomenon
    if not has_substitution and not str(result.get("identifier") or "").strip():
        identifier = extract_identifier(message)
        if identifier:
            result["identifier"] = identifier
    material_name = str(result.get("material_name") or "").strip()
    identifier = str(result.get("identifier") or "").strip()
    if material_name and identifier and identifier.casefold() in material_name.casefold():
        result.pop("identifier", None)
    if not isinstance(result.get("filters"), list) or not result["filters"]:
        parsed_filters = parse_target_filters(message)
        if parsed_filters:
            result["filters"] = parsed_filters
    if not isinstance(result.get("target_metrics"), list) or not result["target_metrics"]:
        target_metrics = extract_target_metrics(message)
        if target_metrics:
            result["target_metrics"] = target_metrics
        elif str(result.get("target_metric") or "").strip():
            result["target_metrics"] = [str(result["target_metric"]).strip()]
    return result


def extract_material_name(message: str) -> str | None:
    text = str(message or "").strip()
    patterns = (
        r"(?:查询|查找|查)?\s*(?P<material>[^，。？?；;]{1,40}?)的原料使用效果",
        r"使用(?P<material>[^，。？?；;]{1,40}?)(?:的)?(?:样品|性能|原料)",
        r"(?P<material>[^，。？?；;]{1,40}?)(?:用在哪些|用量多少)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        material = _clean_material(match.group("material"))
        if material:
            return material
    return None


def extract_substitution_materials(message: str) -> tuple[str, str] | None:
    text = str(message or "").strip()
    listed = re.search(
        r"[：:](?P<original>[^，。；;]{1,40}?)和(?P<replacement>[^，。；;]{1,40}?)"
        r"(?:，|。|？|并结合|$)",
        text,
    )
    if listed and ("替代" in text or "替换" in text):
        original = _clean_material(listed.group("original"))
        replacement = _clean_material(listed.group("replacement"))
        if original and replacement:
            return original, replacement
    reversed_match = _SUBSTITUTION_REVERSED.search(text)
    if reversed_match:
        original = _clean_material(reversed_match.group("original"))
        replacement = _clean_material(reversed_match.group("replacement"))
        if original and replacement:
            return original, replacement
    into_match = _SUBSTITUTION_INTO.search(text)
    if into_match:
        original = _clean_material(into_match.group("original"))
        replacement = _clean_material(into_match.group("replacement"))
        if original and replacement:
            return original, replacement
    by_match = _SUBSTITUTION_BY.search(text)
    if by_match:
        original = _clean_material(by_match.group("original"))
        replacement = _clean_material(by_match.group("replacement"))
        if original and replacement:
            return original, replacement
    return None


def extract_competitor_name(message: str) -> str | None:
    match = re.search(
        r"竞品\s*(?P<name>[A-Za-z0-9][A-Za-z0-9./+＋ -]{0,60})",
        str(message or ""),
    )
    if not match:
        return None
    return match.group("name").strip() or None


def extract_phenomenon(message: str) -> str | None:
    text = str(message or "")
    for phenomenon in (
        "粘接失效", "粘接失败", "开裂", "析出", "变色", "老化", "失效",
    ):
        if phenomenon in text:
            return phenomenon
    return None


def extract_identifier(message: str) -> str | None:
    found = _EXPLICIT_IDENTIFIER.findall(str(message or ""))
    return str(found[-1]) if len(found) == 1 else None


def extract_target_metrics(message: str) -> list[str]:
    text = str(message or "").strip()
    patterns = (
        r"哪些(?:关键)?变量影响(?P<target>[^，。；:：？?]{1,50})",
        r"(?P<target>[^，。；:：？?]{1,50})受哪些(?:关键)?变量影响",
        r"(?:影响|决定)(?P<target>[^，。；:：？?]{1,50}?)(?:的)?(?:关键变量|关键因素|影响因素|变量|因素)",
        r"(?P<target>[^，。；:：？?]{1,50}?)(?:的)?(?:关键变量|影响因素)",
        r"(?P<left>[^，。；:：？?]{1,40}?)(?:和|与|跟)(?P<right>[^，。；:：？?]{1,40}?)(?:为什么|怎么|如何)?(?:冲突|权衡|此消彼长)",
        r"提高(?P<left>[^，。；:：？?]{1,40}?).*?(?P<right>[^，。；:：？?]{1,40}?)(?:下降|降低)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if "left" in match.groupdict() and "right" in match.groupdict():
            left = _clean_target_metric(match.group("left"))
            right = _clean_target_metric(match.group("right"))
            if left and right:
                return [left, right]
            continue
        target = _clean_target_metric(match.group("target"))
        if target:
            return [target]
    return []


def parse_target_filters(message: str) -> list[dict[str, Any]]:
    text = str(message or "")
    filters: list[dict[str, Any]] = []
    operator_pattern = (
        r"(?P<field>[^，。；：:]{1,50}?)"
        r"(?P<operator>大于等于|小于等于|大于|小于|高于|低于|不低于|不高于|至少|至多)"
        r"\s*(?P<value>\d+(?:\.\d+)?)"
    )
    operators = {
        "大于等于": "gte",
        "小于等于": "lte",
        "大于": "gt",
        "小于": "lt",
        "高于": "gt",
        "低于": "lt",
        "不低于": "gte",
        "不高于": "lte",
        "至少": "gte",
        "至多": "lte",
    }
    for match in re.finditer(operator_pattern, text):
        field = _clean_filter_field(match.group("field"))
        if not field:
            continue
        filters.append(
            {
                "section": "performance",
                "field": field,
                "operator": operators[match.group("operator")],
                "value": _number(match.group("value")),
            }
        )

    between = re.search(
        r"(?P<field>[^，。；：:]{1,50}?)在\s*(?P<low>\d+(?:\.\d+)?)"
        r"\s*(?:到|至)\s*(?P<high>\d+(?:\.\d+)?)\s*(?:之间|以内)",
        text,
    )
    if between:
        field = _clean_filter_field(between.group("field"))
        if field:
            filters.append(
                {
                    "section": "performance",
                    "field": field,
                    "operator": "between",
                    "values": [
                        _number(between.group("low")),
                        _number(between.group("high")),
                    ],
                }
            )
    return filters


def _clean_material(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("查找", "查询", "有没有用", "请查", "查", "使用"):
        while text.startswith(prefix):
            text = text[len(prefix) :].strip()
    for suffix in ("的配方记录", "配方记录", "的配方", "的", "配方", "记录", "历史", "样品"):
        while text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text.strip(" ：:，,。？?；;")


def _clean_filter_field(value: str) -> str | None:
    text = str(value or "").strip()
    for separator in (
        "新项目", "冷启动", "目标", "要求", "查找", "筛选", "且", "并且",
    ):
        if separator in text:
            text = text.rsplit(separator, 1)[-1].strip()
    for prefix in ("查找", "筛选", "找", "目标", "要求", "请"):
        while text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip(" ：:，,。；;？?的")
    return text or None


def _clean_target_metric(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("哪些", "什么", "分析", "查看", "查找", "查询", "目标", "要求"):
        while text.startswith(prefix):
            text = text[len(prefix):].strip()
    for suffix in ("性能", "指标", "结果", "的", "为什么会", "为什么"):
        while text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text.strip(" ：:，,。；;？?的")


def _number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number
