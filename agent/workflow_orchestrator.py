from __future__ import annotations

from typing import Any, Callable

from agent.scenario_composer import ScenarioPlan, SkillPlanStep
from runtime.progress import emit_progress
from schemas.user_context import UserContext


SkillStepExecutor = Callable[
    [SkillPlanStep, dict[str, Any], UserContext],
    dict[str, Any],
]


class ScenarioWorkflowOrchestrator:
    """Execute a declarative scenario plan as a fixed, ordered Skill DAG."""

    def execute(
        self,
        *,
        plan: ScenarioPlan,
        tool_args: dict[str, Any],
        ctx: UserContext,
        execute_skill: SkillStepExecutor,
    ) -> dict[str, Any]:
        args = dict(tool_args)
        step_results: list[dict[str, Any]] = []

        for step in plan.steps:
            emit_title = f"{plan.scenario_name}:{step.operation}"
            try:
                self._progress(
                    "running",
                    emit_title,
                    step,
                    f"开始执行 {step.skill_display_name}。",
                )
                result = execute_skill(step, dict(args), ctx)
            except Exception as exc:
                self._progress(
                    "failed",
                    emit_title,
                    step,
                    f"{step.skill_display_name}执行失败：{exc}",
                    result_status="ERROR",
                )
                raise
            status = str(result.get("status") or "OK")
            step_results.append({
                "step_id": step.step_id,
                "skill_name": step.skill_name,
                "operation": step.operation,
                "status": status,
            })

            if step.operation == "ensure_model" and status == "MODEL_REQUIRED":
                if step is not plan.terminal_step:
                    step_results.append({
                        "step_id": plan.terminal_step.step_id,
                        "skill_name": plan.terminal_step.skill_name,
                        "operation": plan.terminal_step.operation,
                        "status": "SKIPPED",
                        "reason": "MODEL_REQUIRED",
                    })
                self._annotate(result, plan, step_results)
                self._progress(
                    "completed",
                    emit_title,
                    step,
                    "未找到可用模型，后续业务 Skill 已跳过。",
                    result_status=status,
                )
                return result

            if step.operation == "ensure_model" and status == "OK":
                selected = self._selected_models(result)
                if len(selected) == 1:
                    args.setdefault("model_id", str(selected[0].get("model_id") or ""))
                    args.setdefault(
                        "model_version",
                        str(selected[0].get("version") or ""),
                    )

            self._progress(
                "completed",
                emit_title,
                step,
                f"{step.skill_display_name}已完成，状态 {status}。",
                result_status=status,
            )

        if not step_results:
            raise ValueError("场景 Workflow 不能为空")
        final_result = result
        if final_result is None:
            raise ValueError("场景 Workflow 没有产生结果")
        self._annotate(final_result, plan, step_results)
        return final_result

    @staticmethod
    def _selected_models(result: dict[str, Any]) -> list[dict[str, Any]]:
        payload = result.get("result")
        if not isinstance(payload, dict):
            return []
        selected = payload.get("selected_models")
        if not isinstance(selected, list):
            return []
        return [dict(item) for item in selected if isinstance(item, dict)]

    @staticmethod
    def _annotate(
        result: dict[str, Any],
        plan: ScenarioPlan,
        step_results: list[dict[str, Any]],
    ) -> None:
        execution = result.setdefault("workflow_execution", {})
        if not isinstance(execution, dict):
            execution = {}
            result["workflow_execution"] = execution
        execution.update({
            "version": "scenario-workflow-v1",
            "scenario_name": plan.scenario_name,
            "primary_operation": plan.primary_operation,
            "steps": step_results,
        })

    @staticmethod
    def _progress(
        progress_status: str,
        title: str,
        step: SkillPlanStep,
        message: str,
        *,
        result_status: str | None = None,
    ) -> None:
        details: dict[str, Any] = {
            "step_id": step.step_id,
            "skill_name": step.skill_name,
            "operation": step.operation,
        }
        if result_status is not None:
            details["result_status"] = result_status
        emit_progress(
            "workflow_orchestration",
            str(progress_status),
            title,
            message,
            **details,
        )
