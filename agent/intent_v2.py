from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class IntentToolPlanStep:
    """A validated, descriptive execution step.

    The plan is produced by backend code, not trusted directly from the LLM.
    It is currently used for observability and future multi-tool orchestration.
    Existing V0.1.1/V0.1.2 execution paths remain unchanged.
    """

    kind: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "args": dict(self.args),
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class DeepSeekIntentDecision:
    """Intent Router V2 decision with V1-compatible properties.

    ``intent``, ``tool_name`` and ``tool_args`` remain available so the
    existing chat execution code can migrate incrementally.
    """

    domain: str
    primary_intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    secondary_intents: tuple[str, ...] = ()
    entities: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    context_reference: dict[str, Any] = field(default_factory=dict)
    tool_plan: tuple[IntentToolPlanStep, ...] = ()
    needs_clarification: bool = False
    clarification_question: str = ""
    reasoning_summary: str = ""
    router_version: str = "2.1"

    @property
    def intent(self) -> str:
        return self.primary_intent

    def to_routing_meta(self) -> dict[str, Any]:
        return {
            "version": self.router_version,
            "domain": self.domain,
            "primary_intent": self.primary_intent,
            "secondary_intents": list(self.secondary_intents),
            "entities": dict(self.entities),
            "scope": dict(self.scope),
            "constraints": dict(self.constraints),
            "context_reference": dict(self.context_reference),
            "tool_plan": [step.to_dict() for step in self.tool_plan],
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
        }
