from __future__ import annotations

from agent.router import RuleIntentRouter
from api.chat_ui import (
    ChatUIRequest,
    _route_round2a2_database_intent,
    chat_ui,
)
from schemas.user_context import UserContext


class FakeCore:
    def __init__(self):
        self.rule_router = RuleIntentRouter()
        self.calls = []

    def execute(self, intent, tool_name, tool_args, ctx):
        self.calls.append((intent, tool_name, tool_args, ctx))
        return {
            "status": "ok",
            "analysis_type": intent,
            "ranking": [],
            "evidence": [{"source": "eln_sample", "record_id": 3811}],
            "warnings": [],
        }

    def answer(self, message, intent, result):
        return f"executed:{intent}"


class FakeContainer:
    def __init__(self):
        self.core = FakeCore()


def ctx():
    return UserContext(
        user_id="u1",
        company_id="c1",
        project_ids=(115,),
        permission_source="test",
        all_projects=False,
    )


def test_plain_rank_question_is_reserved_for_business_mysql():
    decision = _route_round2a2_database_intent(
        "冲击强度最高的前5个样品",
        RuleIntentRouter(),
    )
    assert decision is not None
    assert decision.intent == "performance_rank"
    assert decision.tool_name == "list_samples_for_analysis"


def test_explicit_company_data_question_is_not_preempted():
    decision = _route_round2a2_database_intent(
        "单位真实数据里冲击强度最高的前5个样品",
        RuleIntentRouter(),
    )
    assert decision is None


def test_chat_ui_rank_response_proves_mysql_route():
    container = FakeContainer()
    response = chat_ui(
        ChatUIRequest(message="哪几个样品冲击强度最好？"),
        ctx(),
        container,
    )
    assert response.intent == "performance_rank"
    assert response.tool_name == "list_samples_for_analysis"
    assert response.router == "materials_round2a2_mysql"
    assert response.routing["version"] == "2A-2.3"
    assert response.routing["scope"]["data_source"] == "business_mysql"
    assert container.core.calls[0][0] == "performance_rank"


def test_chat_ui_series_response_proves_mysql_route():
    response = chat_ui(
        ChatUIRequest(message="N20260305这一组在研究什么？"),
        ctx(),
        FakeContainer(),
    )
    assert response.intent == "experiment_series_analysis"
    assert response.tool_args["keyword"] == "N20260305"
    assert response.router == "materials_round2a2_mysql"


def test_chat_ui_quality_response_proves_mysql_route():
    response = chat_ui(
        ChatUIRequest(message="帮我看看这些数据有没有问题"),
        ctx(),
        FakeContainer(),
    )
    assert response.intent == "data_quality_check"
    assert response.router == "materials_round2a2_mysql"
