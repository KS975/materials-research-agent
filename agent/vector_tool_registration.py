from __future__ import annotations

from typing import Any

from agent.tool_registry import ToolRegistry
from agent.vector_search import VectorSearchTool


VECTOR_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_vector_knowledge": {
        "description": "通过后端受管向量检索 Gateway 检索授权项目知识片段，并做二次权限校验",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 0, "maxLength": 2000},
                "project_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 100,
                    "uniqueItems": True,
                },
                "all_projects": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "score_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    }
}


def register_vector_tools(
    registry: ToolRegistry,
    tool: VectorSearchTool,
) -> None:
    for name, spec in VECTOR_TOOL_SPECS.items():
        registry.register(
            name,
            str(spec["description"]),
            tool,
            input_schema=spec["input_schema"],
            access_scope="user_context",
            handler_context="required",
        )
