from __future__ import annotations

import unittest
from typing import Any

from agent.core import AgentCore
from agent.scenario_composer import ScenarioWorkflowComposer
from schemas.user_context import UserContext
from skills.catalog import build_default_skill_registry


class FakeRegistry:
    def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"unexpected direct tool call: {name}")


class FakeEngineAdapter:
    def __init__(self, *, model_required: bool = False):
        self.model_required = model_required
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def execute(
        self,
        intent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
    ) -> dict[str, Any]:
        self.calls.append((intent, tool_name, dict(tool_args)))
        if intent == "ensure_model":
            if self.model_required:
                return {"status": "MODEL_REQUIRED", "result": {}, "answer": "缺模型"}
            return {
                "status": "OK",
                "result": {
                    "selected_models": [{
                        "model_id": "model_test",
                        "version": "v001",
                        "target_name": "performance.impact",
                    }]
                },
                "answer": "模型可用",
            }
        return {"status": "OK", "result": {"prediction_count": 1}, "answer": "完成"}


class ScenarioWorkflowOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = UserContext(
            user_id="user-1",
            company_id="company_a",
            project_ids=(1,),
            permission_source="test",
        )
        self.skill_registry = build_default_skill_registry()
        self.composer = ScenarioWorkflowComposer(self.skill_registry)

    def _core(self, adapter: FakeEngineAdapter) -> AgentCore:
        return AgentCore(
            registry=FakeRegistry(),
            llm=object(),
            llm_enabled=False,
            skill_registry=self.skill_registry,
            scenario_composer=self.composer,
            engine_workflow_adapter=adapter,
        )

    def test_prediction_dag_passes_selected_model_to_terminal_skill(self) -> None:
        adapter = FakeEngineAdapter()
        result = self._core(adapter).execute(
            "predict_performance",
            "predict_model",
            {"project_id": 1, "target_metric": "impact", "inputs": [{"A": 1.0}]},
            self.ctx,
        )
        self.assertEqual(result["status"], "OK")
        self.assertEqual(
            [item[0] for item in adapter.calls],
            ["ensure_model", "predict_performance"],
        )
        self.assertEqual(adapter.calls[1][2]["model_id"], "model_test")
        self.assertEqual(adapter.calls[1][2]["model_version"], "v001")
        self.assertEqual(
            [step["operation"] for step in result["workflow_execution"]["steps"]],
            ["ensure_model", "predict_performance"],
        )

    def test_missing_model_short_circuits_terminal_skill(self) -> None:
        adapter = FakeEngineAdapter(model_required=True)
        result = self._core(adapter).execute(
            "predict_performance",
            "predict_model",
            {"project_id": 1, "target_metric": "impact", "inputs": [{"A": 1.0}]},
            self.ctx,
        )
        self.assertEqual(result["status"], "MODEL_REQUIRED")
        self.assertEqual([item[0] for item in adapter.calls], ["ensure_model"])
        steps = result["workflow_execution"]["steps"]
        self.assertEqual(steps[-1]["operation"], "predict_performance")
        self.assertEqual(steps[-1]["status"], "SKIPPED")


if __name__ == "__main__":
    unittest.main()
