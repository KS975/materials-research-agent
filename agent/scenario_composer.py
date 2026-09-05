from __future__ import annotations

from dataclasses import dataclass, replace
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
        return self.terminal_step.skill_name

    @property
    def terminal_step(self) -> SkillPlanStep:
        return self.steps[-1]

    @property
    def terminal_skill_display_name(self) -> str:
        return self.terminal_step.skill_display_name

    @property
    def executor_family(self) -> str:
        return self.terminal_step.executor_family

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "skill-scenario-v2",
            "scenario_name": self.scenario_name,
            "primary_operation": self.primary_operation,
            "primary_skill": self.primary_skill,
            "terminal_operation": self.terminal_step.operation,
            "requires_approval": self.requires_approval,
            "steps": [step.to_dict() for step in self.steps],
        }


class ScenarioWorkflowComposer:
    """Convert a fine-grained intent into an auditable fixed Skill DAG."""

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
        terminal_step = SkillPlanStep(
            step_id="skill-1",
            skill_name=spec.name,
            skill_display_name=spec.display_name,
            operation=intent,
            tool_name=tool_name,
            tool_args=args,
            executor_family=spec.executor_family_for(intent),
            workflow=spec.workflow,
        )
        steps = [terminal_step]

        # Model-dependent scenarios share one deterministic pre-step. The
        # public intent remains unchanged; ensure_model is an internal Skill
        # and never becomes an LLM-selectable modeling shortcut.
        if intent in {
            "predict_performance",
            "optimize_formula",
            "recommend_next_experiments",
        }:
            ensure_spec = self.registry.get("ensure_model")
            ensure_args = dict(args)
            ensure_spec.validate_dispatch(
                intent="ensure_model",
                tool_name="list_artifacts",
                tool_args=ensure_args,
            )
            ensure_step = SkillPlanStep(
                step_id="skill-1",
                skill_name=ensure_spec.name,
                skill_display_name=ensure_spec.display_name,
                operation="ensure_model",
                tool_name="list_artifacts",
                tool_args=ensure_args,
                executor_family=ensure_spec.executor_family_for("ensure_model"),
                workflow=ensure_spec.workflow,
            )
            steps = [
                ensure_step,
                replace(terminal_step, step_id="skill-2"),
            ]

        return ScenarioPlan(
            scenario_name=f"{spec.name}.{intent}",
            primary_operation=intent,
            steps=tuple(steps),
            requires_approval=bool(spec.approval_points),
        )

