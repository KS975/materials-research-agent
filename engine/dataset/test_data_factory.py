from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.exceptions import ValidationError
from engine.ingestion.reader import hash_dataframe, read_tabular


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str
    target_fields: list[str]
    affected_ratio: float = 0.1
    magnitude: float = 0.05
    random_seed: int = 42
    params: dict[str, Any] | None = None

    def validate(self) -> None:
        allowed = {
            "normal_jitter",
            "missing",
            "duplicate",
            "outlier",
            "target_conflict",
            "leakage",
            "missing_field",
            "closure_missing",
            "test_consistency",
            "sample_size_reduction",
            "combined_faults",
        }
        if self.kind not in allowed:
            raise ValidationError(f"unsupported perturbation kind: {self.kind}")
        if not self.target_fields:
            raise ValidationError("target_fields must not be empty")
        if not 0 < self.affected_ratio <= 1:
            raise ValidationError("affected_ratio must be in (0, 1]")
        if self.magnitude <= 0:
            raise ValidationError("magnitude must be positive")


@dataclass(frozen=True)
class PerturbationResult:
    dataframe: pd.DataFrame
    source_uri: str
    source_hash: str
    perturbation_hash: str
    perturbation_spec: PerturbationSpec
    fault_mask: dict[str, Any]
    expected_findings: list[str]
    quality_config: dict[str, Any]


def generate_perturbation(
    source: str | Path | pd.DataFrame,
    spec: PerturbationSpec,
) -> PerturbationResult:
    spec.validate()
    if isinstance(source, pd.DataFrame):
        dataframe = source.copy(deep=True)
        source_frame = source.copy(deep=True)
        source_uri = "dataframe"
        source_hash = hash_dataframe(source)
    else:
        dataframe = read_tabular(source)
        source_frame = dataframe.copy(deep=True)
        source_uri = str(Path(source).resolve())
        source_hash = _hash_file(Path(source))

    missing_targets = set(spec.target_fields) - set(dataframe.columns)
    if missing_targets:
        raise ValidationError(f"target fields missing from source: {sorted(missing_targets)}")

    output, quality_config = _apply_perturbation(dataframe, spec)
    fault_mask = _build_fault_mask(source_frame, output, spec)
    return PerturbationResult(
        dataframe=output,
        source_uri=source_uri,
        source_hash=source_hash,
        perturbation_hash=hash_dataframe(output),
        perturbation_spec=spec,
        fault_mask=fault_mask,
        expected_findings=_expected_findings(spec),
        quality_config=quality_config,
    )


def save_perturbation(
    result: PerturbationResult,
    output_dir: str | Path,
    *,
    engine_version: str = "0.1.0",
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec_hash = _hash_object(asdict(result.perturbation_spec))
    dataset_id = f"{result.perturbation_spec.kind}_{result.source_hash[:12]}_{spec_hash[:8]}"
    version = 1
    artifact_dir = root / dataset_id / f"v{version:03d}"
    while artifact_dir.exists():
        version += 1
        artifact_dir = root / dataset_id / f"v{version:03d}"
    artifact_dir.mkdir(parents=True)

    result.dataframe.to_parquet(artifact_dir / "dataset.parquet", index=False)
    metadata = {
        "artifact_type": "dataset",
        "dataset_id": dataset_id,
        "version": f"v{version:03d}",
        "data_hash": result.perturbation_hash,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    lineage = {
        "source_uri": result.source_uri,
        "source_hash": result.source_hash,
        "parent_dataset_id": None,
        "perturbation_spec": asdict(result.perturbation_spec),
        "engine_version": engine_version,
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifact_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifact_dir / "fault_mask.json").write_text(
        json.dumps(result.fault_mask, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (artifact_dir / "expected_findings.json").write_text(
        json.dumps({"expected_findings": result.expected_findings}, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "quality-config.json").write_text(
        json.dumps(result.quality_config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return artifact_dir


def _apply_perturbation(
    dataframe: pd.DataFrame,
    spec: PerturbationSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if spec.kind == "combined_faults":
        components = (spec.params or {}).get("components") or [
            {"kind": "missing"},
            {"kind": "duplicate"},
            {"kind": "outlier"},
            {"kind": "leakage"},
        ]
        output = dataframe.copy(deep=True)
        child_expected: list[dict[str, Any]] = []
        for index, component in enumerate(components):
            child_spec = PerturbationSpec(
                kind=str(component.get("kind", "")),
                target_fields=list(component.get("target_fields") or spec.target_fields),
                affected_ratio=float(component.get("affected_ratio", spec.affected_ratio)),
                magnitude=float(component.get("magnitude", spec.magnitude)),
                random_seed=spec.random_seed + index,
                params=dict(component.get("params") or {}),
            )
            child_spec.validate()
            output, _ = _apply_perturbation(output, child_spec)
            child_expected.append({
                "kind": child_spec.kind,
                "expected_findings": _expected_findings(child_spec),
            })
        return output, {"combined_faults": child_expected}

    perturb = globals()[f"_{spec.kind}"]
    output = perturb(dataframe, spec)
    return output, _quality_config(spec)


def _feature_fields(dataframe: pd.DataFrame, target_fields: list[str]) -> list[str]:
    return [
        column for column in dataframe.columns
        if column not in set(target_fields)
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def _normal_jitter(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    rng = np.random.default_rng(spec.random_seed)
    for column in _feature_fields(output, spec.target_fields):
        std = float(output[column].std(skipna=True))
        if not np.isfinite(std) or std == 0:
            continue
        noise = rng.normal(0.0, spec.magnitude * std, len(output))
        lower = float(output[column].min(skipna=True))
        upper = float(output[column].max(skipna=True))
        output[column] = (output[column] + noise).clip(lower, upper)
    return output


def _missing(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    rng = np.random.default_rng(spec.random_seed)
    for column in spec.target_fields:
        mask = rng.random(len(output)) < spec.affected_ratio
        output.loc[mask, column] = np.nan
    return output


def _duplicate(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.random_seed)
    count = max(1, int(len(dataframe) * spec.affected_ratio))
    indices = rng.choice(len(dataframe), size=count, replace=False)
    return pd.concat([dataframe, dataframe.iloc[indices]], ignore_index=True)


def _outlier(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    rng = np.random.default_rng(spec.random_seed)
    columns = _feature_fields(output, spec.target_fields)
    if not columns:
        return output
    count = max(1, int(len(output) * spec.affected_ratio))
    rows = rng.choice(len(output), size=count, replace=False)
    for row in rows:
        column = str(rng.choice(columns))
        std = float(output[column].std(skipna=True))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        output.iloc[row, output.columns.get_loc(column)] += (5 + spec.magnitude) * std
    return output


def _target_conflict(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    rng = np.random.default_rng(spec.random_seed)
    count = max(1, int(len(output) * spec.affected_ratio))
    indices = rng.choice(len(output), size=count, replace=False)
    appended = output.iloc[indices].copy(deep=True)
    for column in spec.target_fields:
        std = float(appended[column].std(skipna=True))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        appended[column] = appended[column] + (3 + spec.magnitude) * std
    return pd.concat([output, appended], ignore_index=True)


def _leakage(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    rng = np.random.default_rng(spec.random_seed)
    primary = spec.target_fields[0]
    std = float(output[primary].std(skipna=True))
    if not np.isfinite(std) or std == 0:
        std = 1.0
    output["engine_post_experiment_proxy"] = (
        output[primary] + rng.normal(0, spec.magnitude * std, len(output))
    )
    return output


def _missing_field(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    return output.drop(columns=[spec.target_fields[0]])


def _closure_missing(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    params = spec.params or {}
    field = str(params.get("closure_field") or spec.target_fields[0])
    if field not in output.columns:
        raise ValidationError(f"closure field missing from source: {field}")
    rng = np.random.default_rng(spec.random_seed)
    mask = rng.random(len(output)) < spec.affected_ratio
    output.loc[mask, field] = np.nan
    return output


def _test_consistency(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    output = dataframe.copy(deep=True)
    params = spec.params or {}
    test_field = str(params.get("test_field", "engine_test_name"))
    test_value = str(params.get("test_value", "injected_test_variant"))
    unit_field = str(params.get("unit_field", "engine_test_unit"))
    unit_value = str(params.get("unit_value", "injected_unit_variant"))
    output[test_field] = test_value
    output[unit_field] = unit_value
    condition_field = params.get("condition_field")
    if condition_field is not None:
        condition_field = str(condition_field)
        output[condition_field] = np.nan
    return output


def _sample_size_reduction(dataframe: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    keep_ratio = max(0.01, min(1.0, 1.0 - spec.affected_ratio))
    count = max(1, int(len(dataframe) * keep_ratio))
    rng = np.random.default_rng(spec.random_seed)
    indices = rng.choice(len(dataframe), size=count, replace=False)
    return dataframe.iloc[sorted(indices)].reset_index(drop=True)


def _quality_config(spec: PerturbationSpec) -> dict[str, Any]:
    params = spec.params or {}
    config: dict[str, Any] = {}
    if spec.kind == "closure_missing":
        config["closure"] = {
            "required_fields": [str(params.get("closure_field") or spec.target_fields[0])],
        }
    elif spec.kind == "leakage":
        config["leakage"] = {
            "post_experiment_fields": ["engine_post_experiment_proxy"],
        }
    elif spec.kind == "test_consistency":
        params = spec.params or {}
        test_field = str(params.get("test_field", "engine_test_name"))
        test_value = str(params.get("test_value", "injected_test_variant"))
        unit_field = str(params.get("unit_field", "engine_test_unit"))
        unit_value = str(params.get("unit_value", "injected_unit_variant"))
        config["consistency_specs"] = [
            {
                "target_field": target,
                "test_field": test_field,
                "expected_test": str(params.get("expected_test", "expected_test")),
                "unit_field": unit_field,
                "expected_unit": str(params.get("expected_unit", "expected_unit")),
            }
            for target in spec.target_fields
        ]
    return config


def _expected_findings(spec: PerturbationSpec) -> list[str]:
    return {
        "normal_jitter": [],
        "missing": ["target_missing"],
        "duplicate": ["exact_duplicate"],
        "outlier": ["feature_outlier"],
        "target_conflict": ["target_conflict"],
        "leakage": ["explicit_leakage"],
        "missing_field": ["schema_validation"],
        "closure_missing": ["sample_closure"],
        "test_consistency": ["test_consistency"],
        "sample_size_reduction": ["sample_count"],
        "combined_faults": [],
    }[spec.kind]


def _build_fault_mask(
    source: pd.DataFrame,
    output: pd.DataFrame,
    spec: PerturbationSpec,
) -> dict[str, Any]:
    changed_cells: dict[str, list[int]] = {}
    common_columns = [column for column in source.columns if column in output.columns]
    if len(source) == len(output):
        for column in common_columns:
            source_values = source[column].reset_index(drop=True)
            output_values = output[column].reset_index(drop=True)
            try:
                changed = source_values.ne(output_values)
            except TypeError:
                changed = source_values.astype("string").ne(output_values.astype("string"))
            indices = [int(index) for index in output.index[changed.fillna(False)]]
            if indices:
                changed_cells[column] = indices

    return {
        "kind": spec.kind,
        "target_fields": spec.target_fields,
        "affected_ratio": spec.affected_ratio,
        "random_seed": spec.random_seed,
        "source_row_count": len(source),
        "output_row_count": len(output),
        "changed_columns": sorted(changed_cells),
        "changed_cells": changed_cells,
        "params": spec.params or {},
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_object(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
