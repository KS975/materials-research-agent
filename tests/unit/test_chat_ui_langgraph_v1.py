from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.chat_ui as chat_ui_module
from orchestration.chat_ui_graph import (
    build_chat_ui_graph,
    invoke_chat_ui_graph,
)
from schemas.chat_ui import ChatUIRequest, ChatUIResponse
from schemas.user_context import UserContext


def _ctx(user_id: str = "user-a") -> UserContext:
    return UserContext(
        user_id=user_id,
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


def _response(answer: str) -> ChatUIResponse:
    return ChatUIResponse(
        answer=answer,
        intent="test_intent",
        router="langgraph_v1_test",
    )


def test_v1_graph_runs_production_executor_once_and_preserves_response():
    calls = []

    def executor(body, ctx, container):
        calls.append((body.message, ctx.user_id, container.name))
        return _response(f"answer:{body.message}")

    graph = build_chat_ui_graph(executor)
    response = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(message="测试生产图"),
        user_context=_ctx(),
        container=SimpleNamespace(name="container-a"),
    )

    assert response.answer == "answer:测试生产图"
    assert response.intent == "test_intent"
    assert calls == [("测试生产图", "user-a", "container-a")]


def test_v1_graph_preserves_http_exception_status_and_detail():
    def executor(body, ctx, container):
        raise HTTPException(status_code=403, detail="无权访问当前项目")

    graph = build_chat_ui_graph(executor)
    with pytest.raises(HTTPException) as exc_info:
        invoke_chat_ui_graph(
            graph,
            body=ChatUIRequest(message="访问项目"),
            user_context=_ctx(),
            container=SimpleNamespace(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "无权访问当前项目"


def test_v1_graph_request_state_is_isolated_between_concurrent_calls():
    def executor(body, ctx, container):
        return _response(f"{ctx.user_id}:{body.message}")

    graph = build_chat_ui_graph(executor)

    def run(index: int) -> str:
        response = invoke_chat_ui_graph(
            graph,
            body=ChatUIRequest(message=f"message-{index}"),
            user_context=_ctx(f"user-{index}"),
            container=SimpleNamespace(),
        )
        return response.answer

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(run, range(8)))

    assert answers == [f"user-{i}:message-{i}" for i in range(8)]


def test_public_chat_ui_endpoint_invokes_compiled_v1_graph(monkeypatch):
    expected = _response("graph endpoint answer")

    class FakeGraph:
        def __init__(self):
            self.state = None

        def invoke(self, state):
            self.state = state
            return {**state, "response": expected}

    graph = FakeGraph()
    monkeypatch.setattr(chat_ui_module, "_chat_ui_graph", graph)
    container = SimpleNamespace(name="production-container")
    body = ChatUIRequest(message="从接口进入")

    response = chat_ui_module.chat_ui(body, _ctx(), container)

    assert response is expected
    assert graph.state["body"] is body
    assert graph.state["user_context"].company_id == "company-a"
    assert graph.state["container"] is container
