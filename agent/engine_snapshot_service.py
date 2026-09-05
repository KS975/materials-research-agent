from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from agent.engine_workflow_types import ArtifactScope, SourceSnapshot
from agent.field_catalog import (
    bind_metric_to_catalog,
    build_material_field_catalog,
    normalize_field_name,
)


class EngineSnapshotService:
    """Build and validate authorized project snapshots for engine workflows."""

    def __init__(self, registry: Any, *, max_source_rows: int = 5000):
        self.registry = registry
        self.max_source_rows = int(max_source_rows)

    def build_dataset_inputs(
        self,
        scope: ArtifactScope,
        args: dict[str, Any],
    ) -> tuple[SourceSnapshot, str, dict[str, Any]]:
        target_metric = self._required_string(args.get("target_metric"))
        snapshot = self.snapshot(scope)
        catalog = snapshot.catalog
        binding = bind_metric_to_catalog(
            target_metric,
            catalog,
            section=str(args.get("target_section") or "auto"),
        )
        if binding.get("status") != "ok":
            candidates = ", ".join(
                str(item) for item in binding.get("candidates") or []
            )
            suffix = f"候选字段：{candidates}。" if candidates else ""
            raise ValueError(
                f"目标字段“{target_metric}”未唯一绑定到授权数据字段。{suffix}"
            )
        section = str(binding.get("section"))
        normalized = normalize_field_name(binding.get("canonical"))
        entry = self._catalog_entry(catalog, section, normalized)
        if int(entry.get("ambiguous_sample_count") or 0) > 0:
            raise ValueError(f"目标字段 {target_metric} 在授权数据中存在同名多条记录。")
        units = [str(item) for item in entry.get("units") or []]
        requested_unit = str(args.get("target_unit") or "").strip()
        if len(units) > 1:
            raise ValueError(
                f"目标字段 {target_metric} 存在多种单位：{', '.join(units)}。"
            )
        if requested_unit and units and requested_unit not in units:
            raise ValueError(
                f"目标单位 {requested_unit} 与数据记录单位 {units[0]} 不一致。"
            )

        column = self._dynamic_column(section, normalized, catalog)
        metadata = {
            "target_fields": [column],
            "identifier_fields": ["sample_id"],
            "units": {column: requested_unit or (units[0] if units else "")},
        }
        return snapshot, column, metadata

    def snapshot(self, scope: ArtifactScope) -> SourceSnapshot:
        source = self.registry.execute(
            "list_samples_for_analysis",
            keyword="",
            ctx=scope.ctx,
            limit=500,
        )
        if not isinstance(source, dict) or source.get("status") != "ok":
            raise ValueError("授权项目数据快照读取失败。")
        if not source.get("scan_complete", True):
            raise ValueError("授权项目数据扫描未完整结束，已阻止建模或优化。")
        sample_count = int(source.get("count") or 0)
        if sample_count <= 0:
            raise ValueError("当前授权项目没有可用于建模或优化的样品数据。")
        if sample_count > self.max_source_rows:
            raise ValueError(
                f"授权样品数 {sample_count} 超过配置上限 {self.max_source_rows}。"
            )
        catalog = build_material_field_catalog(source)
        records, warnings = self.flatten_samples(source.get("samples") or [])
        if not records:
            raise ValueError("授权项目字段解析后没有可用数据行。")
        numeric_feature_fields = self._coerce_numeric_columns(records)
        serialized = json.dumps(
            records, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        data_hash = hashlib.sha256(serialized).hexdigest()
        source_dir = scope.session_root / "source_snapshots"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"authorized_project_{data_hash[:20]}.csv"
        if not source_path.exists():
            pd.DataFrame.from_records(records).to_csv(
                source_path, index=False, encoding="utf-8"
            )
        file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        return SourceSnapshot(
            path=source_path,
            source_hash=file_hash,
            records=records,
            catalog=catalog,
            numeric_feature_fields=numeric_feature_fields,
            sample_count=sample_count,
            warnings=warnings,
        )

    @staticmethod
    def apply_dataset_field_config(
        user_config: dict[str, Any],
        snapshot: SourceSnapshot,
        target_column: str,
    ) -> None:
        user_config["target_fields"] = [target_column]
        user_config["identifier_fields"] = ["sample_id"]
        if user_config.get("feature_fields") is None:
            user_config["feature_fields"] = [
                item for item in snapshot.numeric_feature_fields
                if item != target_column and not item.startswith("performance.")
            ]
            return
        feature_fields = [
            str(item).strip() for item in user_config["feature_fields"]
            if str(item).strip()
        ]
        if not feature_fields:
            raise ValueError("feature_fields 不能为空。")
        if target_column in feature_fields:
            raise ValueError("目标字段不能同时作为特征字段。")
        user_config["feature_fields"] = feature_fields

    def sample_context(
        self,
        scope: ArtifactScope,
        identifier: Any,
    ) -> dict[str, Any]:
        result = self.registry.execute(
            "get_sample_context",
            identifier=identifier,
            ctx=scope.ctx,
        )
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ValueError("未在当前 Company + Project 权限范围内找到待预测样品。")
        return result

    def flatten_samples(
        self,
        samples: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        records: list[dict[str, Any]] = []
        ambiguous: set[str] = set()
        for sample_item in samples:
            sample = dict(sample_item.get("sample") or {})
            row: dict[str, Any] = {
                "sample_id": sample.get("id"),
                "sample_name": sample.get("name"),
                "project_id": sample.get("project_id"),
                "sample_type": sample.get("sample_type"),
                "create_time": sample.get("create_time"),
            }
            for section in ("formula", "process", "performance"):
                grouped: dict[str, list[dict[str, Any]]] = {}
                for field in sample_item.get(section) or []:
                    name = str(field.get("name") or field.get("raw_key") or "").strip()
                    if name:
                        grouped.setdefault(normalize_field_name(name), []).append(field)
                for normalized, fields in grouped.items():
                    display_name = str(
                        fields[0].get("name")
                        or fields[0].get("raw_key")
                        or normalized
                    )
                    column = f"{section}.{display_name}"
                    if len(fields) == 1:
                        row[column] = self._scalar(fields[0].get("value"))
                    else:
                        ambiguous.add(f"{section}.{display_name}")
            for name, value in dict(sample_item.get("conditions") or {}).items():
                row[f"condition.{name}"] = self._scalar(value)
            records.append(row)
        warnings = (
            [f"存在同名多条动态字段，已置为缺失：{', '.join(sorted(ambiguous))}"]
            if ambiguous
            else []
        )
        return records, warnings

    @staticmethod
    def _coerce_numeric_columns(
        records: list[dict[str, Any]],
    ) -> list[str]:
        columns = sorted({
            key for record in records for key in record
        })
        blocked = {
            "sample_id", "sample_name", "project_id", "sample_type", "create_time"
        }
        numeric_columns: list[str] = []
        for column in columns:
            if column in blocked:
                continue
            values = [record.get(column) for record in records]
            non_missing = [value for value in values if value is not None]
            if not non_missing:
                continue
            parsed: list[Decimal | None] = []
            for value in values:
                if value is None:
                    parsed.append(None)
                    continue
                try:
                    candidate = Decimal(str(value).strip())
                    parsed.append(candidate if candidate.is_finite() else None)
                except (InvalidOperation, ValueError):
                    parsed.append(None)
            parsed_non_missing = [item for item in parsed if item is not None]
            if len(parsed_non_missing) != len(non_missing):
                continue
            numeric_columns.append(column)
            for record, value in zip(records, parsed):
                if value is None:
                    record[column] = None
                elif value == value.to_integral_value():
                    record[column] = int(value)
                else:
                    record[column] = float(value)
        return numeric_columns

    @staticmethod
    def _catalog_entry(
        catalog: dict[str, Any],
        section: str,
        normalized: str,
    ) -> dict[str, Any]:
        for item in (catalog.get("sections") or {}).get(section) or []:
            if normalize_field_name(item.get("name")) == normalized:
                return dict(item)
        raise ValueError("目标字段目录绑定失败。")

    @staticmethod
    def _dynamic_column(
        section: str,
        normalized: str,
        catalog: dict[str, Any],
    ) -> str:
        sections = catalog.get("sections") or {}
        entry = next((
            dict(item) for item in sections.get(section) or []
            if normalize_field_name(item.get("name")) == normalized
        ), None)
        if entry is None:
            raise ValueError("目标字段目录绑定失败。")
        name = str(entry.get("name") or normalized)
        return f"{section}.{name}"

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return value

    @staticmethod
    def _required_string(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("缺少必填业务字段。")
        return text
