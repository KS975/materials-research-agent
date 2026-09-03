from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.config import EnginePathConfig
from engine.exceptions import ArtifactError, ValidationError
from engine.tools.common import (
    ensure_allowed_keys,
    run_wrapped_tool,
    success,
)


TOOL_NAME = "list_artifacts"

LIST_ARTIFACTS_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": "List registered dataset and model artifacts without loading payloads.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dataset_roots": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Defaults to ENGINE_ARTIFACT_ROOT/datasets.",
            },
            "model_registry_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Defaults to ENGINE_ARTIFACT_ROOT/models/model-registry.json.",
            },
            "target_name": {"type": "string"},
            "status": {"type": "string"},
        },
    },
    "output_schema": {
        "type": "object",
        "required": ["schema_version", "tool", "status", "result"],
    },
}


def run_tool(payload: dict[str, Any]) -> dict[str, Any]:
    def handler(request: dict[str, Any]) -> dict[str, Any]:
        ensure_allowed_keys(
            request,
            {"dataset_roots", "model_registry_paths", "target_name", "status"},
        )
        config = EnginePathConfig.from_env()
        dataset_roots = _strings(
            request.get("dataset_roots"), [str(config.dataset_dir)]
        )
        registry_paths = _strings(
            request.get("model_registry_paths"),
            [str(config.model_registry_path)],
        )
        target_name = _optional_string(request.get("target_name"))
        status = _optional_string(request.get("status"))

        datasets: list[dict[str, Any]] = []
        for root_value in dataset_roots:
            datasets.extend(_datasets_from_root(root_value))
        models: list[dict[str, Any]] = []
        for registry_value in registry_paths:
            models.extend(_models_from_registry(registry_value))

        if target_name is not None:
            datasets = [
                item for item in datasets
                if target_name in item["target_fields"]
            ]
            models = [
                item for item in models
                if item.get("target_name") == target_name
            ]
        if status is not None:
            models = [item for item in models if item.get("status") == status]

        return success(TOOL_NAME, {
            "datasets": sorted(
                datasets,
                key=lambda item: (item["dataset_id"], item["version"]),
            ),
            "models": sorted(
                models,
                key=lambda item: (
                    item["target_name"], item["model_id"], item["version"]
                ),
            ),
        })

    return run_wrapped_tool(TOOL_NAME, payload, handler)


def _datasets_from_root(root_value: str) -> list[dict[str, Any]]:
    root = Path(root_value)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValidationError(f"dataset root is not a directory: {root}")
    records: list[dict[str, Any]] = []
    for metadata_path in root.glob("**/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactError(
                f"dataset metadata cannot be read: {metadata_path}; {exc}"
            ) from exc
        if metadata.get("artifact_type") != "dataset":
            continue
        artifact_dir = metadata_path.parent
        records.append({
            "dataset_id": str(metadata["dataset_id"]),
            "version": str(metadata["version"]),
            "artifact_uri": str(artifact_dir),
            "data_hash": str(metadata["data_hash"]),
            "target_fields": list(metadata.get("target_fields", [])),
            "feature_fields": list(metadata.get("feature_fields", [])),
            "created_at": metadata.get("created_at"),
        })
    return records


def _models_from_registry(registry_value: str) -> list[dict[str, Any]]:
    registry_path = Path(registry_value)
    if not registry_path.is_file():
        return []
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(
            f"model registry cannot be read: {registry_path}; {exc}"
        ) from exc
    records: list[dict[str, Any]] = []
    for entry in registry.get("models", []):
        if not isinstance(entry, dict):
            continue
        records.append({
            "model_id": str(entry["model_id"]),
            "version": str(entry["version"]),
            "target_name": str(entry["target_name"]),
            "algorithm": str(entry["algorithm"]),
            "dataset_artifact_id": entry.get("dataset_artifact_id"),
            "artifact_uri": entry.get("artifact_dir"),
            "file_uri": entry.get("file_path"),
            "status": entry.get("status", "UNKNOWN"),
            "metrics": entry.get("metrics", {}),
            "feature_names": entry.get("feature_names", []),
            "created_at": entry.get("created_at"),
            "registry_uri": str(registry_path),
        })
    return records


def _strings(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValidationError("expected a string or array of strings")
    return [str(item) for item in value]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("expected a non-empty string when provided")
    return value


__all__ = ["LIST_ARTIFACTS_SPEC", "TOOL_NAME", "run_tool"]
