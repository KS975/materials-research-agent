from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent.engine_workflow_adapter import EngineWorkflowAdapter
from agent.scenario_composer import ScenarioWorkflowComposer
from schemas.user_context import UserContext
from skills.catalog import build_default_skill_registry
from skills.engine_workflow import EngineWorkflowSkill


def _ok(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "tool": tool, "status": "OK", "result": result}


def _sample() -> dict[str, Any]:
    return {
        "sample": {
            "id": 1,
            "name": "S1",
            "project_id": 1,
            "sample_type": "material",
            "create_time": "2026-01-01",
        },
        "formula": [{"name": "A", "value": 10.0, "unit": "phr", "resolved": True}],
        "process": [{"name": "temperature", "value": 180.0, "unit": "C", "resolved": True}],
        "performance": [{"name": "impact", "value": 35.0, "unit": "kJ/m2", "resolved": True}],
        "conditions": {"notch": "A"},
    }


class FakeRegistry:
    def __init__(self, *, gate_fail: bool = False, models: list[dict[str, Any]] | None = None):
        self.gate_fail = gate_fail
        self.models = models or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((name, dict(kwargs)))
        if name == "list_samples_for_analysis":
            ctx = kwargs["ctx"]
            if ctx.project_ids != (1,) or ctx.all_projects:
                raise AssertionError("engine snapshot did not narrow project scope")
            return {
                "status": "ok",
                "count": 1,
                "total_matches": 1,
                "scan_complete": True,
                "scan_truncated": False,
                "samples": [_sample()],
                "warnings": [],
            }
        if name == "get_sample_context":
            return {"status": "ok", **_sample()}
        if name == "list_artifacts":
            return _ok(name, {"datasets": [], "models": self.models})
        if name == "preprocess_dataset":
            decision = "FAIL" if self.gate_fail else "PASS"
            artifact = None if self.gate_fail else {
                "dataset_id": "dataset_test",
                "version": "v001",
                "artifact_dir": str(Path(kwargs["payload"]["output_dir"]) / "dataset_test" / "v001"),
            }
            return _ok(name, {
                "initial_gate": {"decision": decision},
                "final_gate": {"decision": decision},
                "dataset_artifact": artifact,
                "warnings": [],
                "stage_technical_summaries": [],
            })
        if name == "train_model":
            return _ok(name, {
                "training_run": {
                    "model_artifacts": [
                        {"model_id": "model_test", "version": "v001"}
                    ]
                }
            })
        if name == "predict_model":
            return _ok(name, {
                "model": {
                    "model_id": "model_test",
                    "version": "v001",
                    "target_name": "performance.impact",
                },
                "prediction_count": 1,
                "prediction_preview": [],
            })
        if name in {"optimize_formula", "recommend_next_experiments"}:
            return _ok(name, {
                "status": "COMPLETE",
                "selected_candidates": [{"candidate_id": "c1"}],
                "warnings": [],
                "artifact_ids": {
                    "optimization_result": str(
                        Path(kwargs["payload"]["output_dir"]) / "result.json"
                    )
                },
            })
        if name == "get_chart_data":
            return _ok(name, {"datasets": []})
        raise AssertionError(f"unexpected tool call: {name}")


class EngineWorkflowAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.ctx = UserContext(
            user_id="user-1",
            company_id="company_a",
            project_ids=(1,),
            permission_source="test",
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _adapter(self, registry: FakeRegistry) -> EngineWorkflowAdapter:
        return EngineWorkflowAdapter(
            registry,
            artifact_root=self.root,
            default_algorithms=["linear_regression"],
        )

    def test_prepare_discards_host_paths_and_scopes_artifacts(self) -> None:
        registry = FakeRegistry()
        result = self._adapter(registry).execute(
            "engine_prepare_dataset",
            "preprocess_dataset",
            {
                "project_id": 1,
                "target_metric": "impact",
                "input_uri": "Z:/llm-controlled.csv",
                "output_dir": "Z:/llm-output",
                "model_registry_path": "Z:/llm-registry.json",
                "_conversation_id": "conversation-1",
            },
            self.ctx,
        )
        self.assertEqual(result["status"], "OK")
        preprocess = dict(registry.calls[1][1]["payload"])
        project_root = self.root / "companies" / "company_a" / "projects" / "project_1"
        self.assertTrue(
            Path(preprocess["input_uri"]).is_relative_to(project_root / "sessions")
        )
        self.assertTrue(
            Path(preprocess["output_dir"]).is_relative_to(project_root / "sessions")
        )
        self.assertEqual(preprocess["config"]["target_fields"], ["performance.impact"])
        self.assertNotIn(
            "performance.impact", preprocess["config"]["feature_fields"]
        )
        self.assertIn("formula.A", preprocess["config"]["feature_fields"])
        self.assertEqual(preprocess["metadata"]["target_fields"], ["performance.impact"])
        self.assertNotIn("model_registry_path", preprocess)

    def test_training_is_blocked_when_modeling_gate_fails(self) -> None:
        registry = FakeRegistry(gate_fail=True)
        result = self._adapter(registry).execute(
            "automl_training",
            "train_model",
            {"project_id": 1, "target_metric": "impact"},
            self.ctx,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertNotIn("train_model", [name for name, _ in registry.calls])

    def test_prediction_returns_model_required_without_auto_training(self) -> None:
        registry = FakeRegistry(models=[])
        result = self._adapter(registry).execute(
            "predict_performance",
            "predict_model",
            {
                "project_id": 1,
                "target_metric": "impact",
                "inputs": [{"A": 10.0, "temperature": 180.0}],
            },
            self.ctx,
        )
        self.assertEqual(result["status"], "MODEL_REQUIRED")
        self.assertNotIn("train_model", [name for name, _ in registry.calls])
        self.assertNotIn("predict_model", [name for name, _ in registry.calls])

    def test_optimization_uses_project_registry_and_maps_business_fields(self) -> None:
        model = {
            "model_id": "model_test",
            "version": "v001",
            "target_name": "performance.impact",
            "dataset_artifact_id": "dataset_test",
            "status": "CANDIDATE",
            "feature_names": ["formula.A", "process.temperature"],
            "created_at": "2026-01-02T00:00:00Z",
        }
        registry = FakeRegistry(models=[model])
        result = self._adapter(registry).execute(
            "optimize_formula",
            "optimize_formula",
            {
                "project_id": 1,
                "objectives": [{
                    "target_name": "impact",
                    "operator": "greater_or_equal",
                    "value": 30.0,
                }],
                "variables": [{"name": "A", "type": "continuous", "lower": 0.0, "upper": 20.0}],
                "hard_constraints": [{
                    "name": "a_upper",
                    "kind": "bound",
                    "variables": ["A"],
                    "upper": 15.0,
                }],
            },
            self.ctx,
        )
        self.assertEqual(result["status"], "OK")
        optimize = dict(registry.calls[-2][1]["payload"])
        request = dict(optimize["request"])
        expected_registry = (
            self.root / "companies" / "company_a" / "projects" / "project_1"
            / "models" / "model-registry.json"
        )
        self.assertEqual(request["model_registry_path"], str(expected_registry))
        self.assertEqual(request["objectives"][0]["target_name"], "performance.impact")
        self.assertEqual(request["variables"][0]["name"], "formula.A")
        self.assertEqual(request["hard_constraints"][0]["variables"], ["formula.A"])
        self.assertEqual(
            request["model_selection"]["target_mappings"]["performance.impact"]["model_id"],
            "model_test",
        )

    def test_unauthorized_project_is_rejected(self) -> None:
        registry = FakeRegistry()
        with self.assertRaises(PermissionError):
            self._adapter(registry).execute(
                "predict_performance",
                "predict_model",
                {"project_id": 2, "target_metric": "impact", "inputs": [{"A": 1.0}]},
                self.ctx,
            )

    def test_engine_workflow_skill_dispatches_public_intent(self) -> None:
        registry = FakeRegistry(models=[])
        adapter = self._adapter(registry)
        skill = EngineWorkflowSkill(adapter)
        result = skill.execute_intent(
            "predict_performance",
            "predict_model",
            {"project_id": 1, "target_metric": "impact", "inputs": [{"A": 1.0}]},
            self.ctx,
        )
        self.assertEqual(result["status"], "MODEL_REQUIRED")

    def test_skill_contracts_bind_public_engine_tools(self) -> None:
        registry = build_default_skill_registry()
        composer = ScenarioWorkflowComposer(registry)
        expected = {
            "engine_prepare_dataset": ("auto_ml", "preprocess_dataset"),
            "automl_training": ("auto_ml", "train_model"),
            "predict_performance": ("prediction", "predict_model"),
            "optimize_formula": ("optimization", "optimize_formula"),
            "recommend_next_experiments": (
                "optimization",
                "recommend_next_experiments",
            ),
        }
        for intent, (skill_name, tool_name) in expected.items():
            args = {"objectives": [{}]} if intent in {
                "optimize_formula", "recommend_next_experiments"
            } else {"target_metric": "impact"}
            plan = composer.compose(
                intent=intent,
                tool_name=tool_name,
                tool_args=args,
            )
            self.assertEqual(plan.primary_skill, skill_name)
            self.assertEqual(plan.executor_family, "engine_workflow")


if __name__ == "__main__":
    unittest.main()
