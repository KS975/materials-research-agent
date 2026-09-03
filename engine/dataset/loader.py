from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from engine.contracts import CleaningConfig, DatasetArtifact
from engine.exceptions import ArtifactError


@dataclass(frozen=True)
class LoadedDatasetArtifact:
    dataframe: pd.DataFrame
    metadata: dict[str, Any]
    lineage: dict[str, Any]
    artifact: DatasetArtifact
    artifact_dir: Path


def load_dataset_artifact(path: str | Path) -> LoadedDatasetArtifact:
    """Load a versioned dataset artifact without mutating its files."""
    source = Path(path)
    if source.is_dir():
        artifact_dir = source
        data_path = source / "dataset.parquet"
    elif source.is_file() and source.name == "dataset.parquet":
        artifact_dir = source.parent
        data_path = source
    else:
        raise ArtifactError(
            "dataset artifact input must be an artifact directory or dataset.parquet"
        )

    metadata_path = artifact_dir / "metadata.json"
    lineage_path = artifact_dir / "lineage.json"
    required = [data_path, metadata_path, lineage_path]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ArtifactError(f"dataset artifact is incomplete: {missing}")

    try:
        dataframe = pd.read_parquet(data_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"dataset artifact cannot be read: {exc}") from exc

    if not isinstance(metadata, dict) or not isinstance(lineage, dict):
        raise ArtifactError("dataset artifact metadata and lineage must be JSON objects")
    required_metadata = {
        "dataset_id", "version", "data_hash", "feature_fields", "target_fields"
    }
    missing_metadata = required_metadata - set(metadata)
    if missing_metadata:
        raise ArtifactError(
            f"dataset metadata is incomplete: {sorted(missing_metadata)}"
        )

    feature_fields = _strings(metadata["feature_fields"])
    target_fields = _strings(metadata["target_fields"])
    identifier_fields = _strings(metadata.get("identifier_fields", []))
    expected_columns = set(feature_fields + target_fields + identifier_fields)
    missing_columns = expected_columns - set(dataframe.columns)
    if missing_columns:
        raise ArtifactError(
            f"dataset artifact fields missing from parquet: {sorted(missing_columns)}"
        )

    cleaning_payload = dict(metadata.get("cleaning_config") or {})
    cleaning_config = CleaningConfig(**cleaning_payload)
    artifact = DatasetArtifact(
        dataset_id=str(metadata["dataset_id"]),
        version=str(metadata["version"]),
        artifact_dir=str(artifact_dir),
        file_path=str(data_path),
        source_uri=str(lineage.get("source_uri", "")),
        source_hash=str(lineage.get("source_hash", "")),
        data_hash=str(metadata["data_hash"]),
        parent_dataset_id=lineage.get("parent_dataset_id"),
        feature_fields=feature_fields,
        target_fields=target_fields,
        identifier_fields=identifier_fields,
        cleaning_config=cleaning_config,
    )
    return LoadedDatasetArtifact(
        dataframe=dataframe,
        metadata=metadata,
        lineage=lineage,
        artifact=artifact,
        artifact_dir=artifact_dir.resolve(),
    )


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]
