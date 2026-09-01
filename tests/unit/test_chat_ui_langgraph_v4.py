from __future__ import annotations

from types import SimpleNamespace

import pytest

from orchestration.chat_ui_graph import build_chat_ui_graph, invoke_chat_ui_graph
from runtime.chat_ui_workflow import ChatUIWorkflowStore
from schemas.chat_ui import ChatUIRequest, ChatUIResponse
from schemas.user_context import UserContext


SEMANTIC_FAMILIES = {
    "database_explorer",
    "rag",
    "current_attachment",
    "general_conversation",
    "material_tool",
}


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-v4",
        company_id="company-v4",
        project_ids=(115,),
        permission_source="test",
    )


def _native_graph(family: str, calls: list[str]):
    def legacy_executor(body, ctx, container):
        calls.append("legacy")
        return ChatUIResponse(answer="legacy", intent="legacy")

    def planner(state):
        calls.append("planner")
        return {
            "semantic_family": family,
            "intent": f"intent:{family}",
            "router_name": "deepseek",
            "tool_name": None,
            "tool_args": {},
            "routing_meta": {"primary_intent": f"intent:{family}"},
        }

    def build_executor(executor_family):
        def execute(state):
            calls.append(f"execute:{executor_family}")
            return ChatUIResponse(
                answer=f"answer:{executor_family}",
                intent=str(state["intent"]),
                router="deepseek",
                routing=dict(state["routing_meta"]),
            )

        return execute

    return build_chat_ui_graph(
        legacy_executor,
        primary_classifier=lambda state: {"primary_family": "semantic"},
        semantic_planner=planner,
        semantic_executors={
            name: build_executor(name)
            for name in SEMANTIC_FAMILIES
        },
    )


@pytest.mark.parametrize("family", sorted(SEMANTIC_FAMILIES))
def test_v4_semantic_plan_selects_exactly_one_native_executor(family):
    calls: list[str] = []
    response = invoke_chat_ui_graph(
        _native_graph(family, calls),
        body=ChatUIRequest(message=f"route:{family}"),
        user_context=_ctx(),
        container=SimpleNamespace(),
    )

    assert response.answer == f"answer:{family}"
    assert calls == ["planner", f"execute:{family}"]
    assert response.routing["workflow"]["version"] == "langgraph-chat-ui-v4"


@pytest.mark.parametrize("family", ["direct_attachment", "deterministic"])
def test_v4_protected_primary_families_stay_on_legacy_executor(family):
    calls: list[str] = []

    def legacy_executor(body, ctx, container):
        calls.append("legacy")
        return ChatUIResponse(answer="protected", intent="protected")

    graph = build_chat_ui_graph(
        legacy_executor,
        primary_classifier=lambda state: {"primary_family": family},
        semantic_planner=lambda state: (_ for _ in ()).throw(
            AssertionError("semantic planner must not run")
        ),
        semantic_executors={
            name: lambda state: (_ for _ in ()).throw(
                AssertionError("semantic executor must not run")
            )
            for name in SEMANTIC_FAMILIES
        },
    )
    response = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(message=f"protected:{family}"),
        user_context=_ctx(),
        container=SimpleNamespace(),
    )

    assert response.answer == "protected"
    assert calls == ["legacy"]


def test_v4_rejects_incomplete_native_executor_mapping():
    with pytest.raises(ValueError, match="映射不完整"):
        build_chat_ui_graph(
            lambda body, ctx, container: ChatUIResponse(answer="x", intent="x"),
            semantic_planner=lambda state: {"semantic_family": "rag"},
            semantic_executors={"rag": lambda state: ChatUIResponse(answer="x", intent="x")},
        )


def test_v4_planner_failure_does_not_execute_or_retry():
    calls: list[str] = []

    def planner(state):
        calls.append("planner")
        raise RuntimeError("routing unavailable")

    graph = build_chat_ui_graph(
        lambda body, ctx, container: ChatUIResponse(answer="legacy", intent="legacy"),
        primary_classifier=lambda state: {"primary_family": "semantic"},
        semantic_planner=planner,
        semantic_executors={
            name: lambda state: calls.append("unexpected")
            for name in SEMANTIC_FAMILIES
        },
    )
    with pytest.raises(RuntimeError, match="routing unavailable"):
        invoke_chat_ui_graph(
            graph,
            body=ChatUIRequest(message="fail once"),
            user_context=_ctx(),
            container=SimpleNamespace(),
        )
    assert calls == ["planner"]


def test_v4_checkpoint_records_safe_plan_and_cached_resume(tmp_path):
    calls: list[str] = []
    store = ChatUIWorkflowStore(tmp_path / "workflows")
    container = SimpleNamespace(chat_ui_workflow_store=store)
    graph = _native_graph("database_explorer", calls)

    first = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(message="unknown database question"),
        user_context=_ctx(),
        container=container,
    )
    workflow_id = first.routing["workflow"]["workflow_id"]
    status = store.status(workflow_id, _ctx())
    assert status["schema_version"] == 4
    assert status["semantic_family"] == "database_explorer"
    assert status["semantic_intent"] == "intent:database_explorer"
    assert status["semantic_router"] == "deepseek"

    replay = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(
            message="unknown database question",
            workflow_id=workflow_id,
            resume_workflow=True,
        ),
        user_context=_ctx(),
        container=container,
    )
    assert replay.answer == "answer:database_explorer"
    assert calls == ["planner", "execute:database_explorer"]
    assert replay.routing["workflow"]["cached_response_replayed"] is True
