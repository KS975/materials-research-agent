from __future__ import annotations

from math import isfinite
from typing import Any, Iterable

from agent.field_catalog import normalize_field_name


class EngineModelSelector:
    """Select registered models and map business fields to model fields."""

    def __init__(self, allowed_model_statuses: Iterable[str] = ("CANDIDATE",)):
        self.allowed_model_statuses = {
            str(item).strip().upper()
            for item in allowed_model_statuses
            if str(item).strip()
        }
        if not self.allowed_model_statuses:
            raise ValueError("allowed_model_statuses must not be empty")

    def select(
        self,
        records: list[dict[str, Any]],
        target_metric: str,
        *,
        model_id: Any = None,
        version: Any = None,
    ) -> dict[str, Any] | None:
        candidates = [
            item for item in records
            if str(item.get("status") or "CANDIDATE").upper()
            in self.allowed_model_statuses
        ]
        if model_id is not None:
            candidates = [
                item for item in candidates
                if str(item.get("model_id") or "") == str(model_id)
            ]
        if version is not None:
            candidates = [
                item for item in candidates
                if str(item.get("version") or "") == str(version)
            ]
        requested = normalize_field_name(target_metric)
        requested_suffix = normalize_field_name(target_metric.rsplit(".", 1)[-1])
        matched = [
            item for item in candidates
            if normalize_field_name(item.get("target_name")) == requested
            or normalize_field_name(
                str(item.get("target_name") or "").rsplit(".", 1)[-1]
            ) == requested_suffix
        ]
        if len(matched) > 1:
            return None
        if not matched:
            return None
        return max(
            enumerate(matched),
            key=lambda pair: (
                str(pair[1].get("created_at") or ""),
                self.version_number(pair[1].get("version")),
                pair[0],
            ),
        )[1]

    def align_record(
        self,
        record: dict[str, Any],
        feature_names: list[str],
    ) -> dict[str, Any]:
        aligned: dict[str, Any] = {}
        missing: list[str] = []
        for feature_name in feature_names:
            if feature_name in record:
                aligned[feature_name] = record[feature_name]
                continue
            requested = normalize_field_name(feature_name)
            requested_suffix = normalize_field_name(
                feature_name.rsplit(".", 1)[-1]
            )
            matches = [
                key for key, value in record.items()
                if value is not None
                and (
                    normalize_field_name(key) == requested
                    or normalize_field_name(key.rsplit(".", 1)[-1])
                    == requested_suffix
                )
            ]
            if len(matches) == 1:
                aligned[feature_name] = record[matches[0]]
            else:
                missing.append(feature_name)
        if missing:
            raise ValueError(f"模型输入缺少字段：{', '.join(missing)}")
        return aligned

    def observed_targets(
        self,
        row: dict[str, Any],
        target_names: list[str],
    ) -> dict[str, float] | None:
        observed: dict[str, float] = {}
        for target_name in target_names:
            value = row.get(target_name)
            if value is None:
                return None
            try:
                parsed = float(str(value).strip())
            except (TypeError, ValueError):
                return None
            if not isfinite(parsed):
                return None
            observed[target_name] = parsed
        return observed

    def map_variables(
        self,
        variables: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        if variables is None:
            return []
        if not isinstance(variables, list):
            raise ValueError("variables 必须是 JSON 对象数组。")
        mapped = []
        for item in variables:
            if not isinstance(item, dict):
                raise ValueError("每个变量必须是 JSON 对象。")
            copied = dict(item)
            copied["name"] = self.resolve_feature_name(
                copied.get("name"), feature_names
            )
            mapped.append(copied)
        return mapped

    def map_constraints(
        self,
        constraints: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        if constraints is None:
            return []
        if not isinstance(constraints, list):
            raise ValueError("约束必须是 JSON 对象数组。")
        mapped = []
        for item in constraints:
            if not isinstance(item, dict):
                raise ValueError("每条约束必须是 JSON 对象。")
            copied = dict(item)
            variables = [
                self.resolve_feature_name(name, feature_names)
                for name in copied.get("variables") or []
            ]
            if variables:
                copied["variables"] = variables
            mapped.append(copied)
        return mapped

    @staticmethod
    def resolve_feature_name(name: Any, feature_names: list[str]) -> str:
        requested = str(name or "").strip()
        if not requested:
            raise ValueError("变量名不能为空。")
        if requested in feature_names:
            return requested
        requested_normalized = normalize_field_name(requested)
        requested_suffix = normalize_field_name(requested.rsplit(".", 1)[-1])
        matches = [
            feature_name for feature_name in feature_names
            if normalize_field_name(feature_name) == requested_normalized
            or normalize_field_name(feature_name.rsplit(".", 1)[-1])
            == requested_suffix
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"变量 {requested} 未用于当前模型。")
        raise ValueError(f"变量 {requested} 匹配到多个模型字段，请明确字段区段。")

    @staticmethod
    def version_number(value: Any) -> int:
        try:
            return int(str(value).lstrip("vV"))
        except (TypeError, ValueError):
            return 0
