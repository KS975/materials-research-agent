from __future__ import annotations

import json
from typing import Any


def decode_json_value(value: Any, default: Any = None) -> Any:
    """Decode MySQL JSON values defensively.

    PyMySQL installations may expose JSON columns as strings. The application must not
    assume they are already dict/list objects.
    """
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def decode_json_mapping(value: Any) -> dict[str, Any]:
    decoded = decode_json_value(value, default={})
    return dict(decoded) if isinstance(decoded, dict) else {}
