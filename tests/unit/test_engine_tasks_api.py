from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.chat import resolve_user_context
from api.engine_tasks import get_container
from app.main import app
from schemas.user_context import UserContext


def _ctx() -> UserContext:
    return UserContext(
        user_id="api-user",
        company_id="company-api",
        project_ids=(115,),
        permission_source="test",
    )


def test_engine_task_api_create_get_and_cancel():
    status = {
        "schema_version": 1,
        "task_id": "task-api-1",
        "status": "RUNNING",
        "intent": "predict_performance",
        "tool_name": "predict_model",
        "scenario_plan": {"steps": []},
        "current_stage": "predicting",
        "progress_events": [],
        "completed_steps": [],
        "resume_count": 0,
        "approval": None,
        "answer": None,
        "result": None,
        "error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": None,
    }
    manager = SimpleNamespace(
        submit=lambda *, ctx, request: dict(status, status="QUEUED"),
        status=lambda task_id, ctx: dict(status),
        cancel=lambda task_id, ctx: dict(status, status="CANCEL_REQUESTED"),
    )
    app.dependency_overrides[resolve_user_context] = lambda: _ctx()
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        engine_task_manager=manager
    )
    client = TestClient(app)
    try:
        created = client.post(
            "/api/v1/engine-tasks",
            json={
                "intent": "predict_performance",
                "tool_name": "predict_model",
                "tool_args": {"project_id": 115, "target_metric": "impact"},
            },
        )
        assert created.status_code == 202
        assert created.json()["task_id"] == "task-api-1"

        fetched = client.get("/api/v1/engine-tasks/task-api-1")
        assert fetched.status_code == 200
        assert fetched.json()["answer"] is None

        cancelled = client.post("/api/v1/engine-tasks/task-api-1/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCEL_REQUESTED"
    finally:
        app.dependency_overrides.clear()
