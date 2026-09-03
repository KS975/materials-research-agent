from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from engine.exceptions import ArtifactError, ValidationError


@dataclass(frozen=True)
class LoadedModel:
    bundle: dict[str, Any]
    registry_entry: dict[str, Any]


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"model artifact does not exist: {source}")
    try:
        bundle = joblib.load(source)
    except Exception as exc:
        raise ArtifactError(f"model artifact cannot be loaded: {exc}") from exc
    required = {
        "schema_version", "model_id", "version", "algorithm", "target_name",
        "feature_names", "pipeline",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ArtifactError("model artifact bundle is incomplete")
    return bundle


def select_registry_entry(
    registry_path: str | Path,
    *,
    model_id: str | None = None,
    target_name: str | None = None,
    dataset_artifact_id: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.is_file():
        raise ArtifactError(f"model registry does not exist: {path}")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"model registry cannot be read: {exc}") from exc
    models = [
        item for item in registry.get("models", [])
        if isinstance(item, dict)
        and (model_id is None or item.get("model_id") == model_id)
        and (target_name is None or item.get("target_name") == target_name)
        and (
            dataset_artifact_id is None
            or item.get("dataset_artifact_id") == dataset_artifact_id
        )
        and (version is None or item.get("version") == version)
    ]
    if not models:
        raise ArtifactError("no registered model matches the requested selector")
    if version is not None:
        if len(models) != 1:
            raise ArtifactError("model registry contains duplicate selectors")
        return models[0]
    return max(
        enumerate(models),
        key=lambda pair: _selection_key(pair[1], pair[0]),
    )[1]


def _selection_key(entry: dict[str, Any], insertion_index: int) -> tuple[str, int, int]:
    return (
        str(entry.get("created_at", "")),
        _version_number(entry.get("version")),
        insertion_index,
    )


def load_registered_model(
    registry_path: str | Path,
    *,
    model_id: str | None = None,
    target_name: str | None = None,
    dataset_artifact_id: str | None = None,
    version: str | None = None,
) -> LoadedModel:
    entry = select_registry_entry(
        registry_path,
        model_id=model_id,
        target_name=target_name,
        dataset_artifact_id=dataset_artifact_id,
        version=version,
    )
    file_path = entry.get("file_path")
    if not file_path:
        raise ArtifactError("model registry entry has no file_path")
    bundle = load_model_bundle(file_path)
    if (
        bundle.get("model_id") != entry.get("model_id")
        or bundle.get("version") != entry.get("version")
    ):
        raise ArtifactError("model registry entry and model bundle do not match")
    return LoadedModel(bundle=bundle, registry_entry=entry)


def _version_number(value: Any) -> int:
    try:
        return int(str(value).lstrip("vV"))
    except ValueError as exc:
        raise ValidationError(f"invalid model version: {value}") from exc
