from __future__ import annotations

from typing import Any

from engine.config import default_artifact_dir
from engine.exceptions import ValidationError
from engine.optimization.contracts import OptimizationRequest
from engine.optimization.service import optimize_next_experiments
from engine.tools.common import run_wrapped_tool, success
from engine.tools.common import normalize_result_mode


TOOL_NAME = "recommend_next_experiments"

RECOMMEND_NEXT_EXPERIMENTS_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Recommend the next high-value experiments using BO or cold-start design."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["request"],
        "properties": {
            "request": {
                "type": "object",
                "description": "OptimizationRequest JSON contract with observed history.",
            },
            "output_dir": {
                "type": "string",
                "description": "Defaults to ENGINE_ARTIFACT_ROOT/optimizations.",
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
        if set(request) - {"request", "output_dir", "result_mode"}:
            raise ValidationError(
                "unknown tool input fields: "
                f"{sorted(set(request) - {'request', 'output_dir', 'result_mode'})}"
            )
        request_payload = request.get("request")
        if not isinstance(request_payload, dict):
            raise ValidationError("request must be a JSON object")
        mode = request_payload.get("mode", "recommend_next_experiments")
        if mode != "recommend_next_experiments":
            raise ValidationError(
                "recommend_next_experiments requires mode recommend_next_experiments"
            )
        request_payload = dict(request_payload)
        request_payload["mode"] = "recommend_next_experiments"
        optimization_request = OptimizationRequest.from_dict(request_payload)
        output_dir = request.get(
            "output_dir", default_artifact_dir("optimization")
        )
        if not isinstance(output_dir, str) or not output_dir:
            raise ValidationError("output_dir must be a non-empty string when provided")
        result = optimize_next_experiments(
            optimization_request, output_dir=output_dir
        )
        payload = result.to_dict()
        if normalize_result_mode(request) == "summary":
            payload.pop("diagnostic_candidates")
        return success(TOOL_NAME, payload)

    return run_wrapped_tool(TOOL_NAME, payload, handler)


__all__ = ["RECOMMEND_NEXT_EXPERIMENTS_SPEC", "TOOL_NAME", "run_tool"]
