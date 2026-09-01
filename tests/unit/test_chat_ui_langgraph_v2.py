from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.router import RuleIntentRouter
from api.chat_ui import (
    ChatUIRequest,
    _classify_chat_ui_primary_family,
    _classify_chat_ui_semantic_family,
)
from orchestration.chat_ui_graph import (
    build_chat_ui_graph,
    invoke_chat_ui_graph,
)
from schemas.chat_ui import ChatUIResponse
from schemas.user_context import UserContext


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-v2",
        company_id="company-v2",
        project_ids=(115,),
        permission_source="test",
    )


def _container():
    core = SimpleNamespace(rule_router=RuleIntentRouter())
    return SimpleNamespace(core=core)


def _state(message: str, **request_kwargs):
    return {
        "body": ChatUIRequest(message=message, **request_kwargs),
        "user_context": _ctx(),
        "container": _container(),
    }


def test_v2_primary_classifier_preserves_attachment_first_priority():
    update = _classify_chat_ui_primary_family(_state(
        "Project 9016：冲击强度 >= 43，推荐5组方案",
        attachment_ids=["attachment-a"],
        attachment_reference_mode=True,
    ))
    assert update == {
        "primary_family": "direct_attachment",
        "deterministic_kind": "deepseek_attachment_passthrough",
    }


def test_v2_primary_classifier_routes_high_precision_mysql_deterministically():
    update = _classify_chat_ui_primary_family(
        _state("找和3811最像的5个样品")
    )
    assert update["primary_family"] == "deterministic"
    assert update["deterministic_kind"] == "similar_samples"


def test_v2_primary_classifier_routes_unmatched_question_to_semantic_path():
    update = _classify_chat_ui_primary_family(
        _state("为什么尼龙吸水后尺寸会变化？")
    )
    assert update == {
        "primary_family": "semantic",
        "deterministic_kind": "",
    }


@pytest.mark.parametrize(
    ("intent", "router", "expected"),
    [
        ("database_explorer", "deepseek_database_explorer", "database_explorer"),
        ("sample_historical_similarity", "deepseek", "rag"),
        ("joint_mysql_knowledge_analysis", "deepseek", "rag"),
        ("search_historical_knowledge", "deepseek", "rag"),
        ("ask_current_attachment", "deepseek", "current_attachment"),
        ("general_conversation", "deepseek_general_answer", "general_conversation"),
        ("clarification_required", "deepseek", "general_conversation"),
        ("get_sample_context", "deepseek", "material_tool"),
    ],
)
def test_v2_semantic_response_uses_actual_intent_family(intent, router, expected):
    response = ChatUIResponse(answer="ok", intent=intent, router=router)
    update = _classify_chat_ui_semantic_family({"response": response})
    assert update == {"semantic_family": expected}


@pytest.mark.parametrize(
    "primary_family",
    ["direct_attachment", "deterministic", "semantic"],
)
def test_v2_graph_executes_each_primary_branch_once(primary_family):
    calls = []

    def executor(body, ctx, container):
        calls.append(body.message)
        return ChatUIResponse(
            answer="ok",
            intent=(
                "general_conversation"
                if primary_family == "semantic"
                else "deterministic_result"
            ),
        )

    graph = build_chat_ui_graph(
        executor,
        primary_classifier=lambda state: {"primary_family": primary_family},
        semantic_classifier=lambda state: {
            "semantic_family": "general_conversation"
        },
    )
    response = invoke_chat_ui_graph(
        graph,
        body=ChatUIRequest(message=f"branch:{primary_family}"),
        user_context=_ctx(),
        container=SimpleNamespace(),
    )

    assert response.answer == "ok"
    assert calls == [f"branch:{primary_family}"]


def test_v2_graph_rejects_unknown_primary_family_before_execution():
    calls = []

    def executor(body, ctx, container):
        calls.append(body.message)
        return ChatUIResponse(answer="unexpected", intent="unexpected")

    graph = build_chat_ui_graph(
        executor,
        primary_classifier=lambda state: {"primary_family": "unknown"},
    )
    with pytest.raises(RuntimeError, match="未知一级分支"):
        invoke_chat_ui_graph(
            graph,
            body=ChatUIRequest(message="bad branch"),
            user_context=_ctx(),
            container=SimpleNamespace(),
        )
    assert calls == []
