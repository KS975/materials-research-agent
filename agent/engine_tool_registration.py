from __future__ import annotations

from agent.tool_registry import ToolRegistry
from engine.tools import TOOL_FUNCTIONS, TOOL_SPECS


def register_engine_tools(registry: ToolRegistry) -> None:
    """Register framework-neutral engine tools without coupling them to routing."""
    for spec in TOOL_SPECS:
        registry.register(
            spec["name"],
            spec["description"],
            TOOL_FUNCTIONS[spec["name"]],
        )
