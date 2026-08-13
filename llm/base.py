from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class DisabledLLMProvider:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM 未启用")
