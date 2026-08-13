from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from data.json_utils import decode_json_mapping
from data.mysql.repositories.column_repository import ColumnDefinitionRepository
from data.mysql.repositories.material_repository import MaterialRepository
from data.mysql.repositories.dataset_repository import DatasetRealityRepository
from schemas.user_context import UserContext


_RECIPE_KEY = re.compile(r"^R3-(\d+)$")
_PROCESS_KEY = re.compile(r"^S(\d+)$")
_PERFORMANCE_KEY = re.compile(r"^P(\d+)$")
_SERVICE_KEY = re.compile(r"^SP(\d+)$")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _canon_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _norm_name(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


@dataclass(frozen=True)
class DatasetRealityResult:
    project_id: int
    target_metric: str
    report: dict[str, Any]
    wide_rows: list[dict[str, Any]]
    wide_columns: list[str]


class DatasetRealityAnalyzer:
    """Pure analysis layer for V0.1.3-A.

    No model is trained here. This layer only answers:
    "What data really exists, and how much of it closes at sample level?"
    """

    def analyze(
        self,
        *,
        project_id: int,
        company_id: str,
        rows: list[dict[str, Any]],
        material_definitions: dict[int, dict[str, Any]],
        column_definitions: dict[int, dict[str, Any]],
        target_metric: str,
    ) -> DatasetRealityResult:
        target_norm = _norm_name(target_metric)

        target_field_ids = sorted(
            int(field_id)
            for field_id, definition in column_definitions.items()
            if _norm_name(definition.get("name")) == target_norm
            and str(definition.get("belonging_column") or "").lower()
            in ("performance", "performances", "")
        )

        # If belonging_column is populated differently in the real DB, fall back
        # to exact name among all P-field definitions discovered in this project.
        if not target_field_ids:
            target_field_ids = sorted(
                int(field_id)
                for field_id, definition in column_definitions.items()
                if _norm_name(definition.get("name")) == target_norm
            )

        formula_coverage: Counter[str] = Counter()
        process_coverage: Counter[str] = Counter()
        performance_coverage: Counter[str] = Counter()
        service_coverage: Counter[str] = Counter()
        unresolved_keys: Counter[str] = Counter()
        condition_signatures: Counter[str] = Counter()
        sample_names: defaultdict[str, list[int]] = defaultdict(list)
        feature_signatures: defaultdict[str, list[int]] = defaultdict(list)

        all_feature_columns: set[str] = set()
        wide_rows: list[dict[str, Any]] = []

        counts = Counter()
        target_numbers: list[float] = []
        target_non_numeric = 0
        conflicting_target_rows: list[int] = []

        for row in rows:
            sample_id = int(row["id"])
            sample_name = str(row.get("name") or "")
            sample_names[sample_name].append(sample_id)

            recipes = decode_json_mapping(row.get("recipes"))
            process = decode_json_mapping(row.get("craft_param"))
            performance = decode_json_mapping(row.get("performances"))
            service = decode_json_mapping(row.get("service_performances"))
            conditions = decode_json_mapping(row.get("conditions"))

            formula_values: dict[str, Any] = {}
            process_values: dict[str, Any] = {}
            performance_values: dict[str, Any] = {}
            service_values: dict[str, Any] = {}
            target_values: list[Any] = []

            for raw_key, value in recipes.items():
                match = _RECIPE_KEY.match(str(raw_key))
                field_id = int(match.group(1)) if match else None
                definition = material_definitions.get(field_id) if field_id else None
                if definition:
                    name = str(definition.get("name") or raw_key)
                    key = f"formula::{name}"
                    formula_values[key] = value
                    if _present(value):
                        formula_coverage[key] += 1
                    all_feature_columns.add(key)
                else:
                    unresolved_keys[str(raw_key)] += 1

            for raw_key, value in process.items():
                match = _PROCESS_KEY.match(str(raw_key))
                field_id = int(match.group(1)) if match else None
                definition = column_definitions.get(field_id) if field_id else None
                if definition:
                    name = str(definition.get("name") or raw_key)
                    key = f"process::{name}"
                    process_values[key] = value
                    if _present(value):
                        process_coverage[key] += 1
                    all_feature_columns.add(key)
                else:
                    unresolved_keys[str(raw_key)] += 1

            for raw_key, value in performance.items():
                match = _PERFORMANCE_KEY.match(str(raw_key))
                field_id = int(match.group(1)) if match else None
                definition = column_definitions.get(field_id) if field_id else None
                if definition:
                    name = str(definition.get("name") or raw_key)
                    key = f"performance::{name}"
                    performance_values[key] = value
                    if _present(value):
                        performance_coverage[key] += 1
                    if field_id in target_field_ids and _present(value):
                        target_values.append(value)
                else:
                    unresolved_keys[str(raw_key)] += 1

            for raw_key, value in service.items():
                match = _SERVICE_KEY.match(str(raw_key))
                field_id = int(match.group(1)) if match else None
                definition = column_definitions.get(field_id) if field_id else None
                if definition:
                    name = str(definition.get("name") or raw_key)
                    key = f"service::{name}"
                    service_values[key] = value
                    if _present(value):
                        service_coverage[key] += 1
                else:
                    unresolved_keys[str(raw_key)] += 1

            has_formula = any(_present(v) for v in formula_values.values())
            has_process = any(_present(v) for v in process_values.values())
            has_target = bool(target_values)
            has_conditions = bool(conditions)

            counts["total_samples"] += 1
            counts["formula_present"] += int(has_formula)
            counts["process_present"] += int(has_process)
            counts["target_present"] += int(has_target)
            counts["conditions_present"] += int(has_conditions)
            counts["core_closed_formula_process_target"] += int(
                has_formula and has_process and has_target
            )
            counts["strict_closed_with_conditions"] += int(
                has_formula and has_process and has_target and has_conditions
            )

            target_value = target_values[0] if target_values else None
            if len({str(v) for v in target_values}) > 1:
                conflicting_target_rows.append(sample_id)

            if has_target:
                number = _as_number(target_value)
                if number is None:
                    target_non_numeric += 1
                else:
                    target_numbers.append(number)

            condition_signature = _canon_json(conditions) if conditions else ""
            if condition_signature:
                condition_signatures[condition_signature] += 1

            wide = {
                "sample_id": sample_id,
                "sample_name": sample_name,
                "company_id": company_id,
                "project_id": int(project_id),
                "_has_formula": has_formula,
                "_has_process": has_process,
                "_has_target": has_target,
                "_has_conditions": has_conditions,
                "_core_closed": has_formula and has_process and has_target,
                "_strict_closed": has_formula
                and has_process
                and has_target
                and has_conditions,
                f"target::{target_metric}": target_value,
                "_conditions_json": condition_signature,
            }
            wide.update(formula_values)
            wide.update(process_values)
            wide_rows.append(wide)

            # Duplicate-candidate signature intentionally excludes sample name/id.
            signature_payload = {
                "formula": formula_values,
                "process": process_values,
                "target": target_value,
            }
            signature = hashlib.sha256(
                _canon_json(signature_payload).encode("utf-8")
            ).hexdigest()
            feature_signatures[signature].append(sample_id)

        duplicate_name_groups = [
            {"sample_name": name, "sample_ids": ids}
            for name, ids in sorted(sample_names.items())
            if name and len(ids) > 1
        ]
        duplicate_feature_groups = [
            {"signature": signature, "sample_ids": ids}
            for signature, ids in feature_signatures.items()
            if len(ids) > 1
        ]

        total = counts["total_samples"]
        warnings: list[str] = []
        if not target_field_ids:
            warnings.append(
                f"未在本项目已出现的性能字段定义中解析到目标性能“{target_metric}”"
            )
        if counts["target_present"] == 0:
            warnings.append("没有样品包含目标性能值；当前不能进入真实模型训练")
        if counts["core_closed_formula_process_target"] == 0:
            warnings.append("没有形成“配方 + 工艺 + 目标性能”的样品级闭合记录")
        if counts["conditions_present"] < total and total:
            warnings.append(
                f"测试条件缺失 {total - counts['conditions_present']}/{total}；"
                "后续 Modeling Gate 必须检查测试一致性，不能默认可比"
            )
        if unresolved_keys:
            warnings.append(
                f"存在 {sum(unresolved_keys.values())} 个未解析动态字段实例；"
                "正式 Dataset Builder 前需确认映射"
            )
        if target_non_numeric:
            warnings.append(
                f"目标性能中有 {target_non_numeric} 条非数值记录，不能直接用于回归"
            )
        if conflicting_target_rows:
            warnings.append(
                "部分样品同一目标性能出现多个不一致值，需要在建模前消歧"
            )

        target_stats: dict[str, Any] = {
            "metric": target_metric,
            "resolved_field_ids": target_field_ids,
            "present_count": counts["target_present"],
            "numeric_count": len(target_numbers),
            "non_numeric_count": target_non_numeric,
            "conflicting_sample_ids": conflicting_target_rows,
        }
        if target_numbers:
            target_stats.update(
                {
                    "min": min(target_numbers),
                    "max": max(target_numbers),
                    "mean": sum(target_numbers) / len(target_numbers),
                }
            )

        report = {
            "stage": "V0.1.3-A_dataset_reality_check",
            "project_id": int(project_id),
            "company_id": company_id,
            "target_metric": target_metric,
            "summary": dict(counts),
            "target": target_stats,
            "test_conditions": {
                "present_count": counts["conditions_present"],
                "missing_count": total - counts["conditions_present"],
                "unique_nonempty_signatures": len(condition_signatures),
                "signature_counts": dict(condition_signatures),
            },
            "field_coverage": {
                "formula": dict(formula_coverage.most_common()),
                "process": dict(process_coverage.most_common()),
                "performance": dict(performance_coverage.most_common()),
                "service_performance": dict(service_coverage.most_common()),
            },
            "duplicates": {
                "duplicate_sample_name_groups": duplicate_name_groups,
                "duplicate_formula_process_target_groups": duplicate_feature_groups,
            },
            "unresolved_dynamic_fields": dict(unresolved_keys.most_common()),
            "warnings": warnings,
            "decision_boundary": (
                "本阶段只做数据现实映射，不训练模型、不输出 R²/MAE/RMSE，"
                "也不把样本数量自动等同于“可建模”。下一阶段由 Modeling Gate 决定 PASS / CONDITIONAL_PASS / FAIL。"
            ),
        }

        base_columns = [
            "sample_id",
            "sample_name",
            "company_id",
            "project_id",
            "_has_formula",
            "_has_process",
            "_has_target",
            "_has_conditions",
            "_core_closed",
            "_strict_closed",
            f"target::{target_metric}",
            "_conditions_json",
        ]
        wide_columns = [*base_columns, *sorted(all_feature_columns)]

        return DatasetRealityResult(
            project_id=int(project_id),
            target_metric=target_metric,
            report=report,
            wide_rows=wide_rows,
            wide_columns=wide_columns,
        )


class DatasetRealityService:
    def __init__(
        self,
        *,
        samples: DatasetRealityRepository,
        materials: MaterialRepository,
        columns: ColumnDefinitionRepository,
    ) -> None:
        self.samples = samples
        self.materials = materials
        self.columns = columns
        self.analyzer = DatasetRealityAnalyzer()

    @staticmethod
    def _collect_ids(rows: Iterable[dict[str, Any]]) -> tuple[set[int], set[int]]:
        material_ids: set[int] = set()
        column_ids: set[int] = set()

        for row in rows:
            for key in decode_json_mapping(row.get("recipes")):
                match = _RECIPE_KEY.match(str(key))
                if match:
                    material_ids.add(int(match.group(1)))

            for field_name, pattern in (
                ("craft_param", _PROCESS_KEY),
                ("performances", _PERFORMANCE_KEY),
                ("service_performances", _SERVICE_KEY),
            ):
                for key in decode_json_mapping(row.get(field_name)):
                    match = pattern.match(str(key))
                    if match:
                        column_ids.add(int(match.group(1)))

        return material_ids, column_ids

    def run(
        self,
        *,
        project_id: int,
        target_metric: str,
        ctx: UserContext,
    ) -> DatasetRealityResult:
        rows = self.samples.list_project_samples(project_id=project_id, ctx=ctx)
        material_ids, column_ids = self._collect_ids(rows)

        material_definitions = self.materials.get_sample_materials(
            material_ids,
            ctx.company_id,
        )
        column_definitions = self.columns.get_by_ids(
            column_ids,
            ctx.company_id,
        )

        return self.analyzer.analyze(
            project_id=project_id,
            company_id=ctx.company_id,
            rows=rows,
            material_definitions=material_definitions,
            column_definitions=column_definitions,
            target_metric=target_metric,
        )
