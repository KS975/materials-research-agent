from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system: str, user: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def test_v2_payload_exposes_domain_entities_scope_and_safe_plan():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "retrieve",
        "primary_intent": "get_formula",
        "secondary_intents": ["formula_summary"],
        "entities": {"sample_type": "sample"},
        "scope": {"company": "current", "projects": "all_authorized"},
        "constraints": {},
        "context_reference": {"action": "new_request"},
        "tool_name": "get_formula",
        "tool_args": {"identifier": "ABS-051"},
        "tool_plan": [{"name": "evil_unknown_tool"}],
        "needs_clarification": False,
        "reasoning_summary": "查询样品配方",
    }))

    decision = router.route("ABS-051 的配方是什么？")

    assert decision.intent == "get_formula"
    assert decision.domain == "retrieve"
    assert decision.tool_args["identifier"] == "ABS-051"
    assert decision.scope["company"] == "current"
    assert decision.scope["projects"] == "all_authorized"
    # LLM-provided arbitrary plan is ignored; backend derives the executable plan.
    assert len(decision.tool_plan) == 1
    assert decision.tool_plan[0].name == "get_formula"


def test_legacy_query_sample_data_alias_no_longer_breaks_chat():
    router = DeepSeekIntentRouter(FakeLLM({
        "intent": "query_sample_data",
        "tool_name": "query_sample_data",
        "tool_args": {"identifier": 3073},
        "reasoning_summary": "查询样品",
    }))

    decision = router.route("帮我看看 3073")

    assert decision.intent == "get_sample_context"
    assert decision.tool_name == "get_sample_context"
    assert decision.tool_args["identifier"] == 3073


def test_unknown_intent_with_valid_tool_is_normalized_to_tool_intent():
    router = DeepSeekIntentRouter(FakeLLM({
        "intent": "some_new_sample_query_name",
        "tool_name": "get_performance",
        "tool_args": {"identifier": 3811},
    }))

    decision = router.route("3811 的性能数据")
    assert decision.intent == "get_performance"
    assert decision.tool_name == "get_performance"


def test_follow_up_only_look_formula_reuses_unique_previous_sample():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "retrieve",
        "primary_intent": "get_formula",
        "tool_name": "get_formula",
        "tool_args": {},
        "context_reference": {
            "action": "refine_previous",
            "use_previous_sample": True,
        },
    }))

    decision = router.route(
        "只看配方",
        history=[
            {"role": "user", "content": "帮我看看 3073"},
            {"role": "assistant", "content": "已查询样品 ABS-051。"},
        ],
    )

    assert decision.tool_args["identifier"] == "3073"
    assert decision.needs_clarification is False
    assert decision.context_reference["use_previous_sample"] is True
    assert "refine_previous" in decision.secondary_intents


def test_user_correction_replaces_previous_identifier():
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "get_formula",
        "tool_name": "get_formula",
        "tool_args": {"identifier": "3811"},
        "context_reference": {"action": "user_correction"},
    }))

    decision = router.route(
        "不是3811，是3812",
        history=[{"role": "user", "content": "查询3811的配方"}],
    )

    assert decision.tool_args["identifier"] == "3812"
    assert decision.context_reference["action"] == "user_correction"


def test_missing_sample_turns_into_clarification_instead_of_bad_tool_call():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "retrieve",
        "primary_intent": "get_formula",
        "tool_name": "get_formula",
        "tool_args": {},
        "needs_clarification": False,
    }))

    decision = router.route("帮我看看配方")

    assert decision.intent == "get_formula"
    assert decision.needs_clarification is True
    assert "样品" in decision.clarification_question


def test_unmentioned_historical_project_id_is_removed():
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "search_historical_knowledge",
        "tool_name": None,
        "tool_args": {"project_id": 115},
    }))

    decision = router.route("历史上有没有类似的冲击强度异常？")

    assert "project_id" not in decision.tool_args
    assert decision.scope["projects"] == "all_authorized"


def test_explicit_historical_project_id_is_kept():
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "search_historical_knowledge",
        "tool_name": None,
        "tool_args": {"project_id": 115},
    }))

    decision = router.route("Project 115 历史上有没有类似问题？")

    assert decision.tool_args["project_id"] == 115
    assert decision.scope["projects"] == [115]


def test_joint_plan_is_backend_derived_and_multistep():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "diagnosis",
        "primary_intent": "joint_mysql_knowledge_analysis",
        "tool_name": None,
        "tool_args": {
            "left_identifier": 3811,
            "right_identifier": 3809,
            "target_metric": "冲击强度",
            "direction_claim": "更低",
        },
        "secondary_intents": [
            "compare_samples",
            "historical_similar_case",
        ],
    }))

    decision = router.route(
        "3811 的冲击强度比 3809 低很多，历史上有没有类似问题？结合数据库数据和历史资料分析一下。"
    )

    assert decision.needs_clarification is False
    assert [step.name for step in decision.tool_plan] == [
        "compare_samples",
        "historical_knowledge",
        "joint_mysql_knowledge",
    ]
