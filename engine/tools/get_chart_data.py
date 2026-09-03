from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.tools.common import (
    ensure_allowed_keys,
    require_string,
    run_wrapped_tool,
    success,
)
from engine.visualization import build_visualization_bundle


TOOL_NAME = "get_chart_data"

GET_CHART_DATA_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": "Expose UI-neutral chart and table datasets from an engine report.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["input_uri"],
        "properties": {
            "input_uri": {
                "type": "string",
                "description": "Persisted preprocessing/training/prediction/optimization report.",
            },
            "source_kind": {
                "type": "string",
                "enum": ["auto", "preprocessing", "training", "prediction", "optimization"],
                "default": "auto",
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
        ensure_allowed_keys(request, {"input_uri", "source_kind"})
        input_uri = require_string(request, "input_uri")
        source_kind = request.get("source_kind", "auto")
        if source_kind not in {
            "auto", "preprocessing", "training", "prediction", "optimization"
        }:
            raise ValueError(f"unsupported source_kind: {source_kind}")
        source = Path(input_uri)
        try:
            report = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"visualization report cannot be read: {exc}") from exc
        if source_kind == "auto":
            source_kind = _infer_source_kind(report)
        bundle = build_visualization_bundle(
            report,
            source_kind=source_kind,
            source_uri=str(source.resolve()),
        )
        return success(TOOL_NAME, bundle.to_dict())

    return run_wrapped_tool(TOOL_NAME, payload, handler)


def _infer_source_kind(report: dict[str, Any]) -> str:
    if "training_run" in report:
        return "training"
    if "predictions" in report:
        return "prediction"
    if "execution_report" in report:
        return "preprocessing"
    if report.get("record_type") == "optimization_result":
        return "optimization"
    raise ValueError("cannot infer visualization source_kind")


__all__ = ["GET_CHART_DATA_SPEC", "TOOL_NAME", "run_tool"]
