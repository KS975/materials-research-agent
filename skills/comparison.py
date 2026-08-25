from __future__ import annotations

from agent.tool_registry import ToolRegistry
from schemas.user_context import UserContext


class ComparisonSkill:
    intents = {"compare_samples"}

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def can_handle(self, intent: str) -> bool:
        return intent in self.intents

    def execute(self, tool_name: str, tool_args: dict, ctx: UserContext):
        if tool_name != "compare_samples":
            raise ValueError(f"ComparisonSkill 不允许调用工具：{tool_name}")

        # Tool arguments from the LLM/router are metadata-bearing objects.  The
        # underlying read-only DB tool has a deliberately narrow signature and
        # must never receive analysis-only keys such as target_metric,
        # direction, project_id, etc.  Keep an explicit execution-boundary
        # allow-list so an imperfect route cannot turn into a Python TypeError.
        missing = [
            key
            for key in ("left_identifier", "right_identifier")
            if tool_args.get(key) is None or str(tool_args.get(key)).strip() == ""
        ]
        if missing:
            raise ValueError("compare_samples 缺少参数：" + ", ".join(missing))

        safe_args = {
            "left_identifier": tool_args["left_identifier"],
            "right_identifier": tool_args["right_identifier"],
        }
        return self.registry.execute("compare_samples", ctx=ctx, **safe_args)
