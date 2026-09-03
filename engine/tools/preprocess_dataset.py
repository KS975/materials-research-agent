from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.dataset.preprocessing import run_dataset_preprocessing
from engine.ingestion.reader import read_tabular
from engine.config import default_artifact_dir
from engine.tools.common import (
    ensure_allowed_keys,
    normalize_result_mode,
    optional_string,
    require_string,
    run_wrapped_tool,
    success,
)


TOOL_NAME = "preprocess_dataset"

PREPROCESS_DATASET_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Run preset-driven data preprocessing and create a versioned dataset artifact."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["input_uri"],
        "properties": {
            "input_uri": {
                "type": "string",
                "description": "CSV or Parquet URI/path already resolved by the host.",
            },
            "config": {
                "type": "object",
                "description": "Optional user preprocessing overrides.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional field-role and domain metadata.",
            },
            "source_hash": {
                "type": "string",
                "description": "Optional external source hash recorded in lineage.",
            },
            "output_dir": {
                "type": "string",
                "description": "Defaults to ENGINE_ARTIFACT_ROOT/datasets.",
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
            "input_uri", "config", "metadata", "source_hash", "output_dir",
            "result_mode",
        }
        ensure_allowed_keys(request, allowed)
        input_uri = require_string(request, "input_uri")
        output_dir = optional_string(
            request, "output_dir", default_artifact_dir("dataset")
        )
        result_mode = normalize_result_mode(request)
        config = request.get("config")
        metadata = request.get("metadata")
        if config is not None and not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")

        source_path = Path(input_uri)
        dataframe = read_tabular(source_path)
        result = run_dataset_preprocessing(
            dataframe,
            user_config=config,
            metadata=metadata,
            source_uri=str(source_path.resolve()),
            source_hash=request.get("source_hash"),
            output_dir=output_dir,
        )
        payload = result.to_dict()
        if result_mode == "summary":
            payload = {
                "initial_gate": payload["initial_gate"],
                "final_gate": payload["final_gate"],
                "dataset_artifact": payload["dataset_artifact"],
                "warnings": payload["warnings"],
                "stage_technical_summaries": payload["stage_technical_summaries"],
            }
        return success(TOOL_NAME, payload)

    return run_wrapped_tool(TOOL_NAME, payload, handler)


__all__ = ["PREPROCESS_DATASET_SPEC", "TOOL_NAME", "run_tool"]
