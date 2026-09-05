from __future__ import annotations

from typing import Any

from agent.tools import MaterialsTools
from agent.tool_registry import ToolRegistry


_IDENTIFIER: dict[str, Any] = {
    "type": ["string", "integer"],
    "minLength": 1,
    "description": "Sample numeric ID or exact sample name.",
}


MATERIAL_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_sample_context": {
        "description": "读取样品完整研发上下文：项目、配方、工艺、性能、测试条件和可选实验记录",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["identifier"],
            "properties": {"identifier": _IDENTIFIER},
        },
    },
    "get_formula": {
        "description": "读取样品配方",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["identifier"],
            "properties": {"identifier": _IDENTIFIER},
        },
    },
    "get_process": {
        "description": "读取样品工艺",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["identifier"],
            "properties": {"identifier": _IDENTIFIER},
        },
    },
    "get_performance": {
        "description": "读取样品性能及测试条件",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["identifier"],
            "properties": {"identifier": _IDENTIFIER},
        },
    },
    "compare_samples": {
        "description": "比较两个样品的配方、工艺、性能和测试条件",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["left_identifier", "right_identifier"],
            "properties": {
                "left_identifier": _IDENTIFIER,
                "right_identifier": _IDENTIFIER,
            },
        },
    },
    "find_samples": {
        "description": "在当前公司/项目权限范围内查找样品",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["keyword"],
            "properties": {
                "keyword": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
        },
    },
    "list_samples_for_analysis": {
        "description": "按授权项目和样品名范围读取有界样品集合，供确定性排序、系列和质量分析",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["keyword"],
            "properties": {
                "keyword": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
        },
    },
    "get_material_field_catalog": {
        "description": "读取当前公司/项目授权范围内实际出现的材料字段名称、类别和单位；不返回字段值",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    },
}


def register_material_tools(registry: ToolRegistry, tools: MaterialsTools) -> None:
    """Register permission-scoped material tools with explicit JSON contracts."""
    for name, spec in MATERIAL_TOOL_SPECS.items():
        registry.register(
            name,
            str(spec["description"]),
            getattr(tools, name),
            input_schema=spec["input_schema"],
            access_scope="user_context",
            handler_context="required",
        )
