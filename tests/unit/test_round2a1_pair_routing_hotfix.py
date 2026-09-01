from __future__ import annotations

from agent.router import RuleIntentRouter


def _route(message: str):
    decision = RuleIntentRouter().route(message)
    assert decision is not None
    return decision


def test_exact_reported_formula_pair_question_is_not_collapsed_into_one_identifier():
    decision = _route("3811和3809的配方差在哪?")
    assert decision.intent == "formula_difference"
    assert decision.tool_name == "compare_samples"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_formula_pair_with_compare_prefix_is_specialized_before_legacy_compare_regex():
    decision = _route("比较3811和3809的配方差异")
    assert decision.intent == "formula_difference"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_formula_pair_with_query_prefix_is_still_a_pair():
    decision = _route("查看3811和3809的配方差异")
    assert decision.intent == "formula_difference"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_process_pair_question_routes_to_process_difference():
    decision = _route("3811和3809工艺有哪些不同？")
    assert decision.intent == "process_difference"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_comparability_pair_keeps_metric_without_merging_identifiers():
    decision = _route("3811和3809的冲击强度能直接比较吗？")
    assert decision.intent == "comparability_check"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
        "target_metric": "冲击强度",
    }


def test_generic_pair_difference_routes_to_compare_samples():
    decision = _route("3811和3809有什么区别")
    assert decision.intent == "compare_samples"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_single_sample_fallback_behavior_is_preserved():
    assert _route("查看样品3811具体信息").tool_args == {"identifier": "3811"}
    assert _route("查看ABS-051具体信息").tool_args == {"identifier": "ABS-051"}
