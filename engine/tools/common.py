from __future__ import annotations

import json
from typing import Any, Mapping

from engine.exceptions import ArtifactError, EngineError, ValidationError


TOOL_RESULT_SCHEMA_VERSION = 1
RESULT_MODES = {"summary", "full"}


def normalize_payload(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Accept a JSON object or its serialized string form."""
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"tool input is not valid JSON: {exc}") from exc
    else:
        parsed = payload
    if not isinstance(parsed, dict):
        raise ValidationError("engine tool input must be a JSON object")
    return dict(parsed)


def success(tool_name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": tool_name,
        "status": "OK",
        "result": dict(result),
    }


def normalize_result_mode(payload: Mapping[str, Any]) -> str:
    value = str(payload.get("result_mode", "full")).lower()
    if value not in RESULT_MODES:
        raise ValidationError("result_mode must be 'summary' or 'full'")
    return value


def failure(
    tool_name: str,
    exc: Exception,
    *,
    error_code: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if error_code is None:
        if isinstance(exc, ValidationError):
            error_code = "VALIDATION_ERROR"
        elif isinstance(exc, ArtifactError):
            error_code = "ARTIFACT_ERROR"
        elif isinstance(exc, EngineError):
            error_code = "ENGINE_ERROR"
        elif isinstance(exc, (ValueError, TypeError)):
            error_code = "INVALID_INPUT"
        else:
            error_code = "TOOL_EXECUTION_ERROR"
    payload: dict[str, Any] = {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": tool_name,
        "status": "ERROR",
        "error": {
            "code": error_code,
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
    if details is not None:
        payload["details"] = dict(details)
    return payload


def run_wrapped_tool(
    tool_name: str,
    payload: Mapping[str, Any] | str,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    try:
        normalized = normalize_payload(payload)
        return handler(normalized)
    except Exception as exc:  # Host agents require a structured JSON response.
        return failure(tool_name, exc)


def require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return value


def optional_string(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key} must be a non-empty string when provided")
    return value


def ensure_allowed_keys(
    payload: dict[str, Any],
    allowed: set[str],
    *,
    context: str = "tool input",
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValidationError(f"unknown {context} fields: {sorted(unknown)}")
