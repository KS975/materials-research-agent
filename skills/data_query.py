from __future__ import annotations

from agent.tool_registry import ToolRegistry
from schemas.user_context import UserContext


class DataQuerySkill:
    intents = {
        "get_sample_context",
        "get_formula",
        "get_process",
        "get_performance",
        "find_samples",
        "list_samples_for_analysis",
    }

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def can_handle(self, intent: str) -> bool:
        return intent in self.intents

    def execute(self, tool_name: str, tool_args: dict, ctx: UserContext):
        return self.registry.execute(tool_name, ctx=ctx, **tool_args)
