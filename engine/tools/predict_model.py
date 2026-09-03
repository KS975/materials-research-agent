from __future__ import annotations

from typing import Any

import pandas as pd

from engine.ingestion.reader import read_tabular
from engine.modeling.predictor import predict_with_model
from engine.modeling.registry import load_registered_model
from engine.tools.common import normalize_result_mode
from engine.tools.common import (
    ensure_allowed_keys,
    require_string,
    run_wrapped_tool,
    success,
)


TOOL_NAME = "predict_model"

PREDICT_MODEL_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Predict one target with a registered model and return applicability-domain results."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["model_registry_path", "model_selector"],
        "properties": {
            "model_registry_path": {"type": "string"},
            "model_selector": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "model_id": {"type": "string"},
                    "target_name": {"type": "string"},
                    "dataset_artifact_id": {"type": "string"},
                    "version": {"type": "string"},
                },
            },
            "inputs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Inline prediction records for small requests.",
            },
            "input_uri": {
                "type": "string",
                "description": "CSV or Parquet path for batch prediction.",
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
            "model_registry_path", "model_selector", "inputs", "input_uri",
            "result_mode",
        }
        ensure_allowed_keys(request, allowed)
        registry_path = require_string(request, "model_registry_path")
        selector = request.get("model_selector")
        if not isinstance(selector, dict):
            raise ValueError("model_selector must be a JSON object")
        ensure_allowed_keys(
            selector,
            {"model_id", "target_name", "dataset_artifact_id", "version"},
            context="model_selector",
        )
        if not (selector.get("model_id") or selector.get("target_name")):
            raise ValueError("model_selector requires model_id or target_name")

        has_inline_inputs = "inputs" in request
        has_input_uri = "input_uri" in request
        if has_inline_inputs == has_input_uri:
            raise ValueError("provide exactly one of inputs or input_uri")
        if has_inline_inputs:
            records = request["inputs"]
            if not isinstance(records, list) or not records:
                raise ValueError("inputs must be a non-empty array of objects")
            if not all(isinstance(item, dict) for item in records):
                raise ValueError("every prediction input must be a JSON object")
            dataframe = pd.DataFrame.from_records(records)
        else:
            if not isinstance(request["input_uri"], str) or not request["input_uri"]:
                raise ValueError("input_uri must be a non-empty string")
            dataframe = read_tabular(request["input_uri"])

        loaded = load_registered_model(
            registry_path,
            model_id=selector.get("model_id"),
            target_name=selector.get("target_name"),
            dataset_artifact_id=selector.get("dataset_artifact_id"),
            version=selector.get("version"),
        )
        predictions = predict_with_model(loaded.bundle, dataframe)
        prediction_rows = [item.to_dict() for item in predictions]
        payload = {
            "model": {
                "model_id": loaded.bundle["model_id"],
                "version": loaded.bundle["version"],
                "target_name": loaded.bundle["target_name"],
                "algorithm": loaded.bundle["algorithm"],
                "feature_names": list(loaded.bundle["feature_names"]),
            },
            "prediction_count": len(prediction_rows),
            "predictions": prediction_rows,
        }
        if normalize_result_mode(request) == "summary":
            domain_counts: dict[str, int] = {}
            for item in prediction_rows:
                domain = str(item["applicability_domain"])
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            payload["applicability_domain_counts"] = domain_counts
            payload["prediction_preview"] = prediction_rows[:5]
            payload.pop("predictions")
        return success(TOOL_NAME, payload)

    return run_wrapped_tool(TOOL_NAME, payload, handler)


__all__ = ["PREDICT_MODEL_SPEC", "TOOL_NAME", "run_tool"]
