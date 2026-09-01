from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.skill_registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class SkillPlanStep:
    step_id: str
    skill_name: str
    skill_display_name: str
    operation: str
    tool_name: str | None
    tool_args: dict[str, Any]
    executor_family: str
    workflow: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "kind": "skill",
            "skill_name": self.skill_name,
            "skill_display_name": self.skill_display_name,
            "operation": self.operation,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "executor_family": self.executor_family,
            "workflow": list(self.workflow),
        }


@dataclass(frozen=True, slots=True)
class ScenarioPlan:
    scenario_name: str
    primary_operation: str
    steps: tuple[SkillPlanStep, ...]
    requires_approval: bool

    @property
    def primary_skill(self) -> str:
        return self.steps[0].skill_name

    @property
    def executor_family(self) -> str:
        return self.steps[0].executor_family

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "skill-scenario-v1",
            "scenario_name": self.scenario_name,
            "primary_operation": self.primary_operation,
            "primary_skill": self.primary_skill,
            "requires_approval": self.requires_approval,
            "steps": [step.to_dict() for step in self.steps],
        }


class ScenarioWorkflowComposer:
    """Convert a fine-grained intent into an auditable Skill plan.

    V1 intentionally composes one atomic Skill per existing operation.  The
    contract already supports multiple ordered steps, so future scenarios such
    as Ensure Model -> Optimization -> Prediction can be added without changing
    the API or checkpoint shape.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def compose(
        self,
        *,
        intent: str,
        tool_name: str | None,
        tool_args: Mapping[str, Any] | None = None,
    ) -> ScenarioPlan:
        args = dict(tool_args or {})
        spec = self.registry.validate_dispatch(
            intent=intent,
            tool_name=tool_name,
            tool_args=args,
        )
        step = SkillPlanStep(
            step_id="skill-1",
            skill_name=spec.name,
            skill_display_name=spec.display_name,
            operation=intent,
            tool_name=tool_name,
            tool_args=args,
            executor_family=spec.executor_family_for(intent),
            workflow=spec.workflow,
        )
        return ScenarioPlan(
            scenario_name=f"{spec.name}.{intent}",
            primary_operation=intent,
            steps=(step,),
            requires_approval=bool(spec.approval_points),
        )

