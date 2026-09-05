from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from typing import Any

from agent.core import AgentCore
from agent.scenario_composer import ScenarioPlan, ScenarioWorkflowComposer
from runtime.engine_tasks import (
    EngineTaskManager,
    EngineTaskNotFoundError,
    EngineTaskRequest,
    EngineTaskStore,
)
from schemas.user_context import UserContext
from skills.catalog import build_default_skill_registry


class FakeRegistry:
    def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"unexpected direct tool call: {name}")


class ControlledEngineAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.ensure_started = threading.Event()
        self.release_ensure = threading.Event()
        self.prediction_started = threading.Event()
        self.release_prediction = threading.Event()

    def execute(
        self,
        intent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
    ) -> dict[str, Any]:
        self.calls.append((intent, dict(tool_args)))
        if intent == "ensure_model":
            self.ensure_started.set()
            if not self.release_ensure.wait(10):
                raise RuntimeError("ensure_model test barrier timed out")
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
        if intent == "predict_performance":
            self.prediction_started.set()
            if not self.release_prediction.wait(10):
                raise RuntimeError("predict_performance test barrier timed out")
            return {
                "status": "OK",
                "result": {"prediction_count": 1},
                "answer": "预测完成",
            }
        raise AssertionError(f"unexpected intent: {intent}")


def wait_for_status(
    manager: EngineTaskManager,
    task_id: str,
    ctx: UserContext,
    expected: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = manager.status(task_id, ctx)
        if last.get("status") == expected:
            return last
        time.sleep(0.02)
    raise AssertionError(f"task did not reach {expected}: {last}")


class EngineTaskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.ctx = UserContext(
            user_id="user-1",
            company_id="company_a",
            project_ids=(1,),
            permission_source="test",
        )
        self.adapter = ControlledEngineAdapter()
        self.adapter.release_prediction.set()
        self.skill_registry = build_default_skill_registry()
        self.composer = ScenarioWorkflowComposer(self.skill_registry)
        self.core = AgentCore(
            registry=FakeRegistry(),
            llm=object(),
            llm_enabled=False,
            skill_registry=self.skill_registry,
            scenario_composer=self.composer,
            engine_workflow_adapter=self.adapter,
        )
        self.store = EngineTaskStore(self.root / "tasks")
        self.manager = EngineTaskManager(core=self.core, store=self.store)

    def tearDown(self) -> None:
        self.adapter.release_ensure.set()
        self.adapter.release_prediction.set()
        self.manager.shutdown(wait=True)
        self._temporary.cleanup()

    def _request(self) -> EngineTaskRequest:
        return EngineTaskRequest(
            intent="predict_performance",
            tool_name="predict_model",
            tool_args={
                "project_id": 1,
                "target_metric": "impact",
                "inputs": [{"A": 1.0}],
            },
            conversation_id="conversation-1",
        )

    def test_task_completes_with_skill_checkpoints_and_final_report(self) -> None:
        self.adapter.release_ensure.set()
        created = self.manager.submit(ctx=self.ctx, request=self._request())
        task_id = str(created["task_id"])
        status = wait_for_status(self.manager, task_id, self.ctx, "SUCCEEDED")

        self.assertEqual(status["completed_steps"], ["skill-1", "skill-2"])
        self.assertEqual(status["answer"], "预测完成")
        self.assertEqual(status["result"]["workflow_execution"]["steps"][-1]["status"], "OK")
        prediction_args = self.adapter.calls[1][1]
        self.assertEqual(prediction_args["model_id"], "model_test")
        self.assertEqual(prediction_args["model_version"], "v001")
        event_types = {
            item["event_type"] for item in status["progress_events"]
        }
        self.assertIn("TASK_STARTED", event_types)
        self.assertIn("SKILL_CHECKPOINTED", event_types)
        self.assertIn("TASK_COMPLETED", event_types)

    def test_cancel_is_checked_at_skill_boundary(self) -> None:
        created = self.manager.submit(ctx=self.ctx, request=self._request())
        task_id = str(created["task_id"])
        self.assertTrue(self.adapter.ensure_started.wait(10))

        cancel_status = self.manager.cancel(task_id, self.ctx)
        self.assertEqual(cancel_status["status"], "CANCEL_REQUESTED")
        self.adapter.release_ensure.set()
        status = wait_for_status(self.manager, task_id, self.ctx, "CANCELLED")

        self.assertEqual(status["result"], None)
        self.assertFalse(self.adapter.prediction_started.is_set())

    def test_resume_reuses_completed_skill_checkpoint(self) -> None:
        self.adapter.release_prediction.clear()
        created = self.manager.submit(ctx=self.ctx, request=self._request())
        task_id = str(created["task_id"])
        self.assertTrue(self.adapter.ensure_started.wait(10))
        self.adapter.release_ensure.set()
        self.assertTrue(self.adapter.prediction_started.wait(10))
        self.manager.cancel(task_id, self.ctx)
        self.adapter.release_prediction.set()
        wait_for_status(self.manager, task_id, self.ctx, "CANCELLED")

        self.manager.resume(task_id, self.ctx)
        status = wait_for_status(self.manager, task_id, self.ctx, "SUCCEEDED")

        ensure_calls = [call for call in self.adapter.calls if call[0] == "ensure_model"]
        self.assertEqual(len(ensure_calls), 1)
        self.assertTrue(self.adapter.prediction_started.is_set())
        self.assertEqual(status["resume_count"], 1)
        self.assertEqual(status["answer"], "预测完成")

    def test_approval_task_does_not_execute_before_approval(self) -> None:
        self.adapter.release_ensure.set()
        base_plan = self.composer.compose(
            intent="predict_performance",
            tool_name="predict_model",
            tool_args=self._request().tool_args,
        )
        approval_plan = ScenarioPlan(
            scenario_name=base_plan.scenario_name,
            primary_operation=base_plan.primary_operation,
            steps=base_plan.steps,
            requires_approval=True,
        )
        created = self.store.create(
            ctx=self.ctx,
            request=self._request(),
            plan=approval_plan,
        )
        task_id = str(created["task_id"])
        self.assertEqual(created["status"], "AWAITING_APPROVAL")
        self.assertEqual(self.adapter.calls, [])

        approved = self.manager.approve(task_id, self.ctx, reason="test approval")
        self.assertIn(approved["status"], {"QUEUED", "RUNNING"})
        status = wait_for_status(self.manager, task_id, self.ctx, "SUCCEEDED")
        self.assertEqual(status["approval"]["approved_by"], "user-1")

    def test_interrupted_running_task_can_be_recovered(self) -> None:
        plan = self.composer.compose(
            intent="predict_performance",
            tool_name="predict_model",
            tool_args=self._request().tool_args,
        )
        created = self.store.create(
            ctx=self.ctx,
            request=self._request(),
            plan=plan,
        )
        task_id = str(created["task_id"])
        self.store.mark_running(task_id, self.ctx)
        path = self.store._path(self.ctx, task_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["updated_at"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(data), encoding="utf-8")

        recovered = self.manager.recover_interrupted()
        self.assertEqual(recovered, 1)
        self.assertEqual(self.manager.status(task_id, self.ctx)["status"], "INTERRUPTED")

        self.adapter.release_ensure.set()
        self.adapter.release_prediction.set()
        self.manager.resume(task_id, self.ctx)
        wait_for_status(self.manager, task_id, self.ctx, "SUCCEEDED")

    def test_task_is_scoped_to_company_and_user(self) -> None:
        created = self.manager.submit(ctx=self.ctx, request=self._request())
        task_id = str(created["task_id"])
        other_ctx = UserContext(
            user_id="other-user",
            company_id="company_a",
            project_ids=(1,),
            permission_source="test",
        )
        with self.assertRaises(EngineTaskNotFoundError):
            self.manager.status(task_id, other_ctx)


if __name__ == "__main__":
    unittest.main()
