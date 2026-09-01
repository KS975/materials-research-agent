
from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        return json.dumps(
            self.payload,
            ensure_ascii=False,
        )


def test_compare_numeric_identifiers_are_normalized_to_strings():
    router = DeepSeekIntentRouter(
        FakeLLM({
            "intent": "analyze_performance_difference",
            "tool_name": "compare_samples",
            "tool_args": {
                "left_identifier": 3811,
                "right_identifier": 3809,
                "target_metric": "冲击强度",
                "direction_claim": "低",
            },
            "reasoning_summary": "test",
        })
    )

    decision = router.route(
        "为什么 3811 的冲击强度比 3809 低？"
    )

    assert decision.tool_args["left_identifier"] == "3811"
    assert decision.tool_args["right_identifier"] == "3809"
    assert isinstance(
        decision.tool_args["left_identifier"], str
    )
    assert isinstance(
        decision.tool_args["right_identifier"], str
    )


def test_single_numeric_identifier_is_normalized_to_string():
    router = DeepSeekIntentRouter(
        FakeLLM({
            "intent": "get_sample_context",
            "tool_name": "get_sample_context",
            "tool_args": {
                "identifier": 3811,
            },
            "reasoning_summary": "test",
        })
    )

    decision = router.route("查看 3811")

    assert decision.tool_args["identifier"] == "3811"
    assert isinstance(
        decision.tool_args["identifier"], str
    )


def test_joint_mysql_rag_identifiers_are_strings_but_project_stays_int():
    router = DeepSeekIntentRouter(
        FakeLLM({
            "intent": "joint_mysql_knowledge_analysis",
            "tool_name": None,
            "tool_args": {
                "left_identifier": 3811,
                "right_identifier": 3809,
                "target_metric": "冲击强度",
                "direction_claim": "更低",
                "project_id": 115,
            },
            "reasoning_summary": "test",
        })
    )

    decision = router.route(
        "3811 比 3809 低，结合数据库和历史报告分析"
    )

    assert decision.tool_args["left_identifier"] == "3811"
    assert decision.tool_args["right_identifier"] == "3809"
    assert decision.tool_args["project_id"] == 115
    assert isinstance(
        decision.tool_args["project_id"], int
    )


def test_existing_string_identifier_is_preserved():
    router = DeepSeekIntentRouter(
        FakeLLM({
            "intent": "get_sample_context",
            "tool_name": "get_sample_context",
            "tool_args": {
                "identifier": "trial_10",
            },
            "reasoning_summary": "test",
        })
    )

    decision = router.route("查看 trial_10")
    assert decision.tool_args["identifier"] == "trial_10"
