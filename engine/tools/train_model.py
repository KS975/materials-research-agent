from __future__ import annotations

from typing import Any

from engine.config import default_artifact_dir
from engine.dataset.loader import load_dataset_artifact
from engine.exceptions import ValidationError
from engine.modeling.config import resolve_training_config
from engine.modeling.trainer import train_models
from engine.tools.common import (
    ensure_allowed_keys,
    normalize_result_mode,
    optional_string,
    require_string,
    run_wrapped_tool,
    success,
)


TOOL_NAME = "train_model"

TRAIN_MODEL_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Train target models from a versioned DatasetArtifact and register candidates."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["dataset_artifact_uri"],
        "properties": {
            "dataset_artifact_uri": {
                "type": "string",
                "description": "Dataset artifact directory or dataset.parquet path.",
            },
            "config": {
                "type": "object",
                "description": "Optional modeling overrides; artifact metadata is the preset.",
            },
            "output_dir": {
                "type": "string",
                "description": "Defaults to ENGINE_ARTIFACT_ROOT/models.",
            },
            "model_registry_path": {
                "type": "string",
                "description": "Defaults to model-registry.json under output_dir.",
            },
            "result_mode": {
                "type": "string",
                "enum": ["summary", "full"],
                "default": "full",
            },
        },
    },
    "output_schema": {
        "type": "object",
        "required": ["schema_version", "tool", "status", "result"],
    },
}


def run_tool(payload: dict[str, Any]) -> dict[str, Any]:
    def handler(request: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "dataset_artifact_uri", "config", "output_dir",
            "model_registry_path", "result_mode",
        }
        ensure_allowed_keys(request, allowed)
        artifact_uri = require_string(request, "dataset_artifact_uri")
        output_dir = optional_string(
            request, "output_dir", default_artifact_dir("model")
        )
        registry_path = request.get("model_registry_path")
        if registry_path is not None and not isinstance(registry_path, str):
            raise ValueError("model_registry_path must be a string")
        config_payload = request.get("config")
        if config_payload is not None and not isinstance(config_payload, dict):
            raise ValueError("config must be a JSON object")

        loaded = load_dataset_artifact(artifact_uri)
        gate = loaded.lineage.get("modeling_gate", {})
        if gate.get("decision") == "FAIL":
            raise ValidationError(
                "DatasetArtifact modeling gate is FAIL; training is blocked"
            )
        config, resolution = resolve_training_config(
            loaded.dataframe,
            metadata=loaded.metadata,
            user_config=config_payload,
        )
        result = train_models(
            loaded.dataframe,
            config,
            dataset_artifact_id=loaded.artifact.dataset_id,
            dataset_data_hash=loaded.artifact.data_hash,
            source_uri=loaded.artifact.source_uri or str(loaded.artifact_dir),
            output_dir=output_dir,
            registry_path=registry_path,
        )
        payload = {
            "training_config_resolution": resolution,
            "modeling_gate": gate,
            "dataset_artifact": loaded.artifact.to_dict(),
            "training_run": result.to_dict(),
        }
        if normalize_result_mode(request) == "summary":
            payload["training_run"] = {
                "strategies": result.to_dict()["strategies"],
                "model_artifacts": [
                    {
                        key: value
                        for key, value in artifact.to_dict().items()
                        if key != "evaluation_records"
                    }
                    for artifact in result.model_artifacts
                ],
                "candidate_records": [
                    {
                        key: value
                        for key, value in record.to_dict().items()
                        if key not in {"interpretability", "hyperparameters"}
                    }
                    for record in result.candidate_records
                ],
                "technical_summary": result.technical_summary,
            }
        return success(TOOL_NAME, payload)

    return run_wrapped_tool(TOOL_NAME, payload, handler)


__all__ = ["TOOL_NAME", "TRAIN_MODEL_SPEC", "run_tool"]
