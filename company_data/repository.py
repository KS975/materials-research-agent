from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .ingestion import CompanyDataValidationError


def _safe(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(value or "")).strip()
    return value[:100] or "value"


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        out = float(text)
        return out if math.isfinite(out) else None
    except ValueError:
        return None


class CompanyDataRepository:
    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.root = self.runtime_root / "company_data"

    def current_pointer(self) -> dict[str, Any]:
        path = self.root / "current.json"
        if not path.exists():
            raise CompanyDataValidationError(
                "尚未导入单位真实数据。请先运行 scripts.import_company_real_data。"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def import_dir(self, dataset_id: str | None = None) -> Path:
        if dataset_id is None:
            dataset_id = self.current_pointer()["dataset_id"]
        return self.root / "imports" / str(dataset_id)

    def manifest(self, dataset_id: str | None = None) -> dict[str, Any]:
        path = self.import_dir(dataset_id) / "manifest.json"
        if not path.exists():
            raise CompanyDataValidationError(
                f"manifest 不存在: {path}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def product(
        self,
        *,
        product_name: str | None = None,
        local_project_id: int | None = None,
    ) -> dict[str, Any]:
        manifest = self.manifest()
        products = manifest.get("products") or []
        if product_name is not None:
            key = str(product_name).strip().casefold()
            for item in products:
                if str(item.get("product_type") or "").casefold() == key:
                    return item
            raise CompanyDataValidationError(
                f"未找到产品类型: {product_name}"
            )
        if local_project_id is not None:
            for item in products:
                if int(item.get("local_project_id")) == int(local_project_id):
                    return item
            raise CompanyDataValidationError(
                f"未找到 local_project_id={local_project_id}"
            )
        raise CompanyDataValidationError(
            "product_name / local_project_id 至少提供一个"
        )

    def detect_product_in_text(self, text: str) -> dict[str, Any] | None:
        lowered = str(text or "").casefold()
        products = sorted(
            self.manifest().get("products") or [],
            key=lambda x: len(str(x.get("product_type") or "")),
            reverse=True,
        )
        for item in products:
            name = str(item.get("product_type") or "")
            if name and name.casefold() in lowered:
                return item

        match = re.search(
            r"(?:local\s*project|本地项目|project)\s*#?\s*(93\d{4})",
            text,
            re.IGNORECASE,
        )
        if match:
            try:
                return self.product(
                    local_project_id=int(match.group(1))
                )
            except CompanyDataValidationError:
                return None
        return None

    def export_modeling_dataset(
        self,
        *,
        product_name: str,
        target_metric: str,
    ) -> dict[str, Any]:
        product = self.product(product_name=product_name)
        manifest = self.manifest()
        dataset_id = manifest["dataset_id"]
        local_project_id = int(product["local_project_id"])
        target_metric = str(target_metric).strip()
        if not target_metric:
            raise CompanyDataValidationError("target_metric 不能为空")

        wide_path = self.import_dir(dataset_id) / "catalog_wide.csv"
        target_col = f"performance::{target_metric}"

        with wide_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            if target_col not in fields:
                raise CompanyDataValidationError(
                    f"性能指标不存在: {target_metric}"
                )
            product_rows = [
                row for row in reader
                if row.get("product_type") == product_name
            ]

        formula_cols = [
            c for c in fields if c.startswith("formula::")
        ]

        active_formula_cols = []
        for col in formula_cols:
            if any(_num(row.get(col)) is not None for row in product_rows):
                active_formula_cols.append(col)

        numeric_target_rows = [
            row for row in product_rows
            if _num(row.get(target_col)) is not None
        ]

        formula_present = sum(
            any(_num(row.get(c)) is not None for c in active_formula_cols)
            for row in product_rows
        )
        target_present = sum(
            str(row.get(target_col) or "").strip() != ""
            for row in product_rows
        )
        target_numeric_count = len(numeric_target_rows)

        output_dir = (
            self.runtime_root
            / "company_data"
            / "modeling_exports"
            / f"project_{local_project_id}_{_safe(product_name)}_{_safe(target_metric)}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        wide_export = output_dir / "dataset.csv"
        export_fields = (
            [
                "sample_id",
                "sample_name",
                "product_type",
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
            + active_formula_cols
        )
        export_rows = []
        for row in product_rows:
            has_formula = any(
                _num(row.get(c)) is not None
                for c in active_formula_cols
            )
            has_target = str(row.get(target_col) or "").strip() != ""
            export = {
                "sample_id": row.get("sample_key"),
                "sample_name": row.get("sample_name"),
                "product_type": product_name,
                "project_id": local_project_id,
                "_has_formula": has_formula,
                "_has_process": False,
                "_has_target": has_target,
                "_has_conditions": False,
                "_core_closed": False,
                "_strict_closed": False,
                f"target::{target_metric}": row.get(target_col),
                "_conditions_json": "",
            }
            for col in active_formula_cols:
                export[col] = row.get(col)
            export_rows.append(export)

        with wide_export.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=export_fields)
            writer.writeheader()
            writer.writerows(export_rows)

        # V0.1.3-A compatible report, intentionally reflecting missing process/conditions.
        reality = {
            "stage": "V0.1.3-A_dataset_reality_check",
            "project_id": local_project_id,
            "target_metric": target_metric,
            "company_id": "LOCAL_COMPANY_DATA",
            "source": {
                "kind": "company_real_data",
                "dataset_id": dataset_id,
                "product_type": product_name,
                "source_sha256": manifest.get("source", {}).get("sha256"),
            },
            "summary": {
                "total_samples": len(product_rows),
                "formula_present": int(formula_present),
                "process_present": 0,
                "target_present": int(target_present),
                "conditions_present": 0,
                "core_closed_formula_process_target": 0,
                "strict_closed_with_conditions": 0,
            },
            "target": {
                "resolved_field_ids": [],
                "numeric_count": int(target_numeric_count),
                "non_numeric_count": int(
                    max(target_present - target_numeric_count, 0)
                ),
            },
            "test_conditions": {
                "unique_nonempty_signatures": 0,
                "signatures": [],
            },
            "duplicates": {
                "duplicate_sample_name_groups": [],
                "duplicate_formula_process_target_groups": [],
            },
            "unresolved_dynamic_fields": {},
            "warnings": [
                "单位数据的测试条件.xlsx 当前没有显式数据。",
                "工艺.xlsx 中 LOGINCATEGORY/TASKCATEGORY 属于工作流元数据，未作为材料工艺特征。",
                "该导出用于 Reality/Gate 对接；不得绕过 Modeling Gate 强行训练正式模型。",
            ],
            "decision_boundary": (
                "REAL COMPANY DATA IMPORTED. "
                "Modeling readiness must be decided by the existing V0.1.3 Modeling Gate."
            ),
        }

        local_reality_path = output_dir / "reality.json"
        local_reality_path.write_text(
            json.dumps(reality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        standard_reality_dir = (
            self.runtime_root / "v013" / "reality"
        )
        standard_reality_dir.mkdir(parents=True, exist_ok=True)
        standard_reality_path = (
            standard_reality_dir
            / f"project_{local_project_id}_{target_metric}_reality.json"
        )
        standard_reality_path.write_text(
            json.dumps(reality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "dataset_id": dataset_id,
            "product": product,
            "target_metric": target_metric,
            "rows": len(product_rows),
            "target_numeric_count": target_numeric_count,
            "active_formula_features": len(active_formula_cols),
            "process_parameter_rows": 0,
            "condition_rows": 0,
            "official_model_allowed_from_import_alone": False,
            "dataset_csv": str(wide_export),
            "reality_json": str(standard_reality_path),
        }
