from __future__ import annotations

from types import SimpleNamespace
import os
import time

import pytest

from orchestration.chat_ui_graph import (
    build_chat_ui_graph,
    invoke_chat_ui_graph,
)
from runtime.chat_ui_workflow import (
    ChatUIWorkflowConflictError,
    ChatUIWorkflowNotFoundError,
    ChatUIWorkflowStore,
)
from schemas.chat_ui import ChatUIRequest, ChatUIResponse
from schemas.user_context import UserContext


def _ctx(user_id: str = "user-v3", company_id: str = "company-v3") -> UserContext:
    return UserContext(
        user_id=user_id,
        company_id=company_id,
        project_ids=(115,),
        permission_source="test",
    )


def _graph(executor):
    return build_chat_ui_graph(
        executor,
        primary_classifier=lambda state: {"primary_family": "semantic"},
        semantic_classifier=lambda state: {
            "semantic_family": "general_conversation"
        },
    )


def test_v3_success_checkpoint_and_cached_resume_do_not_execute_twice(tmp_path):
    calls = []

    def executor(body, ctx, container):
        calls.append(body.message)
        return ChatUIResponse(
            answer=f"answer:{body.message}",
            intent="general_conversation",
        )

    store = ChatUIWorkflowStore(tmp_path / "workflows")
    container = SimpleNamespace(chat_ui_workflow_store=store)
    graph = _graph(executor)

    first = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(message="checkpoint me"),
        user_context=_ctx(),
        container=container,
    )
    workflow_id = first.routing["workflow"]["workflow_id"]
    assert calls == ["checkpoint me"]
    assert store.status(workflow_id, _ctx())["status"] == "SUCCEEDED"

    replay = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(
            message="checkpoint me",
            workflow_id=workflow_id,
            resume_workflow=True,
        ),
        user_context=_ctx(),
        container=container,
    )

    assert replay.answer == "answer:checkpoint me"
    assert calls == ["checkpoint me"]
    assert replay.routing["workflow"]["cached_response_replayed"] is True
    assert replay.routing["workflow"]["resume_count"] == 1


def test_v3_pause_before_business_execution_then_resume(tmp_path):
    calls = []

    def executor(body, ctx, container):
        calls.append(body.message)
        return ChatUIResponse(answer="continued", intent="general_conversation")

    store = ChatUIWorkflowStore(tmp_path / "workflows")
    container = SimpleNamespace(chat_ui_workflow_store=store)
    graph = _graph(executor)

    paused = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(
            message="pause safely",
            pause_after="classify_primary",
        ),
        user_context=_ctx(),
        container=container,
    )
    workflow_id = paused.routing["workflow"]["workflow_id"]
    assert paused.intent == "workflow_paused"
    assert calls == []
    assert store.status(workflow_id, _ctx())["status"] == "PAUSED"

    continued = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(
            message="pause safely",
            workflow_id=workflow_id,
            resume_workflow=True,
        ),
        user_context=_ctx(),
        container=container,
    )
    assert continued.answer == "continued"
    assert calls == ["pause safely"]
    assert store.status(workflow_id, _ctx())["status"] == "SUCCEEDED"


def test_v3_resume_rejects_changed_question_or_scope(tmp_path):
    store = ChatUIWorkflowStore(tmp_path / "workflows")
    session = store.begin(body=ChatUIRequest(message="original"), ctx=_ctx())

    with pytest.raises(ChatUIWorkflowConflictError, match="不一致"):
        store.begin(
            body=ChatUIRequest(
                message="changed",
                workflow_id=session["workflow_id"],
                resume_workflow=True,
            ),
            ctx=_ctx(),
        )

    with pytest.raises(ChatUIWorkflowNotFoundError):
        store.status(session["workflow_id"], _ctx(user_id="other-user"))


def test_v3_requires_workflow_id_for_resume(tmp_path):
    store = ChatUIWorkflowStore(tmp_path / "workflows")
    with pytest.raises(ChatUIWorkflowConflictError, match="必须提供"):
        store.begin(
            body=ChatUIRequest(message="resume", resume_workflow=True),
            ctx=_ctx(),
        )


def test_v3_active_workflow_lease_blocks_concurrent_resume(tmp_path):
    store = ChatUIWorkflowStore(tmp_path / "workflows", lease_seconds=120)
    session = store.begin(body=ChatUIRequest(message="still running"), ctx=_ctx())

    with pytest.raises(ChatUIWorkflowConflictError, match="拒绝并发恢复"):
        store.begin(
            body=ChatUIRequest(
                message="still running",
                workflow_id=session["workflow_id"],
                resume_workflow=True,
            ),
            ctx=_ctx(),
        )


def test_v3_failed_execution_is_checkpointed_without_automatic_retry(tmp_path):
    calls = []

    def executor(body, ctx, container):
        calls.append(body.message)
        raise RuntimeError("model unavailable")

    store = ChatUIWorkflowStore(tmp_path / "workflows")
    container = SimpleNamespace(chat_ui_workflow_store=store)
    workflow_id = "workflow-failure-v3"

    with pytest.raises(RuntimeError, match="model unavailable"):
        invoke_chat_ui_graph(
            _graph(executor),
            body=ChatUIRequest(message="fail once", workflow_id=workflow_id),
            user_context=_ctx(),
            container=container,
        )

    status = store.status(workflow_id, _ctx())
    assert calls == ["fail once"]
    assert status["status"] == "FAILED"
    assert status["last_error"]["type"] == "RuntimeError"


def test_v3_checkpoint_write_has_bounded_io_retry(tmp_path, monkeypatch):
    store = ChatUIWorkflowStore(
        tmp_path / "workflows",
        checkpoint_retries=3,
    )
    import runtime.chat_ui_workflow as workflow_module

    real_replace = workflow_module.os.replace
    attempts = {"count": 0}

    def flaky_replace(source, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("temporary filesystem error")
        return real_replace(source, target)

    monkeypatch.setattr(workflow_module.os, "replace", flaky_replace)
    session = store.begin(body=ChatUIRequest(message="retry io"), ctx=_ctx())

    assert attempts["count"] == 3
    assert store.status(session["workflow_id"], _ctx())["status"] == "RUNNING"


def test_v3_cleanup_only_removes_expired_scoped_workflow_json(tmp_path):
    store = ChatUIWorkflowStore(
        tmp_path / "workflows",
        ttl_hours=1,
        cleanup_interval_seconds=60,
    )
    session = store.begin(body=ChatUIRequest(message="expire me"), ctx=_ctx())
    checkpoint = next((tmp_path / "workflows").glob("*/*.json"))
    old = time.time() - 2 * 3600
    os.utime(checkpoint, (old, old))
    unrelated = checkpoint.parent / "invalid!name.json"
    unrelated.write_text("{}", encoding="utf-8")
    os.utime(unrelated, (old, old))
    store._last_cleanup_monotonic = 0.0

    store.begin(body=ChatUIRequest(message="trigger cleanup"), ctx=_ctx())

    with pytest.raises(ChatUIWorkflowNotFoundError):
        store.status(session["workflow_id"], _ctx())
    assert unrelated.exists()
