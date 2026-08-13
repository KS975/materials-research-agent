from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        if name in self._tools:
            raise ValueError(f"工具已注册：{name}")
        self._tools[name] = ToolSpec(name=name, description=description, handler=handler)

    def execute(self, name: str, **kwargs: Any) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"未知工具：{name}")
        return spec.handler(**kwargs)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": spec.name, "description": spec.description}
            for spec in self._tools.values()
        ]
