from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine.contracts import (
    CleaningConfig,
    CleaningExecutionReport,
    DataQualityReport,
    DatasetArtifact,
    GateDecision,
    ModelingGateResult,
)
from engine.exceptions import ValidationError
from engine.ingestion.reader import hash_dataframe


def build_dataset(
    dataframe: pd.DataFrame,
    *,
    target_fields: list[str],
    feature_fields: list[str] | None = None,
    identifier_fields: list[str] | None = None,
    quality_report: DataQualityReport,
    gate_result: ModelingGateResult,
    cleaning_config: CleaningConfig | None = None,
    source_uri: str = "dataframe",
    source_hash: str | None = None,
    parent_dataset_id: str | None = None,
    output_dir: str | Path = "engine/artifacts/datasets",
    cleaning_already_applied: bool = False,
    precleaning_report: CleaningExecutionReport | None = None,
) -> DatasetArtifact:
    if gate_result.decision is GateDecision.failed:
        raise ValidationError("ModelingGateResult is FAIL; cannot build a formal dataset")
    if not target_fields:
        raise ValidationError("target_fields must not be empty")

    config = cleaning_config or CleaningConfig()
    targets = list(target_fields)
    identifiers = list(identifier_fields or [])
    required_drop_fields = [] if cleaning_already_applied else config.drop_fields
    missing_columns = set(targets + identifiers + required_drop_fields) - set(dataframe.columns)
    if missing_columns:
        raise ValidationError(f"configured fields missing from DataFrame: {sorted(missing_columns)}")

    features = feature_fields or [
        column for column in dataframe.columns
        if column not in set(targets) | set(identifiers)
    ]
    missing_features = set(features) - set(dataframe.columns)
    if missing_features:
        raise ValidationError(f"feature fields missing from DataFrame: {sorted(missing_features)}")

    cleaned = dataframe.copy(deep=True)
    removed_fields = []
    for field in config.drop_fields:
        if field in cleaned.columns:
            cleaned = cleaned.drop(columns=[field])
            removed_fields.append(field)
            if field in features:
                features.remove(field)

    dropped_duplicate_count = 0
    dropped_missing_target_count = 0
    winsorized_fields: list[str] = []
    if not cleaning_already_applied:
        if config.drop_exact_duplicates:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates(keep="first")
            dropped_duplicate_count = before - len(cleaned)

        if config.drop_missing_target_rows:
            before = len(cleaned)
            cleaned = cleaned.dropna(subset=targets)
            dropped_missing_target_count = before - len(cleaned)

        if cleaned.empty:
            raise ValidationError("cleaning removed all rows; cannot build a dataset artifact")

        if config.winsorize_numeric_outliers:
            numeric_fields = [
                field for field in features + targets
                if pd.api.types.is_numeric_dtype(cleaned[field])
            ]
            for field in numeric_fields:
                q1 = cleaned[field].quantile(0.25)
                q3 = cleaned[field].quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                cleaned[field] = cleaned[field].clip(lower, upper)
                winsorized_fields.append(field)

    cleaned = cleaned.reset_index(drop=True)
    cleaned_hash = hash_dataframe(cleaned)
    source_data_hash = source_hash or hash_dataframe(dataframe)
    dataset_id = f"dataset_{source_data_hash[:16]}_{cleaned_hash[:8]}"
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    version_number = 1
    artifact_dir = artifact_root / dataset_id / f"v{version_number:03d}"
    while artifact_dir.exists():
        version_number += 1
        artifact_dir = artifact_root / dataset_id / f"v{version_number:03d}"
    artifact_dir.mkdir(parents=True)

    data_path = artifact_dir / "dataset.parquet"
    cleaned.to_parquet(data_path, index=False)
    metadata = {
        "artifact_type": "dataset",
        "dataset_id": dataset_id,
        "version": f"v{version_number:03d}",
        "data_hash": cleaned_hash,
        "feature_fields": features,
        "target_fields": targets,
        "identifier_fields": identifiers,
        "cleaning_config": vars(config),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    lineage = {
        "source_uri": source_uri,
        "source_hash": source_data_hash,
        "parent_dataset_id": parent_dataset_id,
        "quality_report": quality_report.to_dict(),
        "modeling_gate": gate_result.to_dict(),
        "cleaning_summary": {
            **(
                precleaning_report.to_dict()
                if precleaning_report is not None
                else {
                    "input_row_count": len(dataframe),
                    "output_row_count": len(cleaned),
                    "dropped_duplicate_count": dropped_duplicate_count,
                    "dropped_missing_target_count": dropped_missing_target_count,
                    "removed_fields": removed_fields,
                    "winsorized_fields": winsorized_fields,
                }
            ),
        },
        "cleaning_already_applied": cleaning_already_applied,
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifact_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return DatasetArtifact(
        dataset_id=dataset_id,
        version=f"v{version_number:03d}",
        artifact_dir=str(artifact_dir),
        file_path=str(data_path),
        source_uri=source_uri,
        source_hash=source_data_hash,
        data_hash=cleaned_hash,
        parent_dataset_id=parent_dataset_id,
        feature_fields=features,
        target_fields=targets,
        identifier_fields=identifiers,
        cleaning_config=config,
    )
