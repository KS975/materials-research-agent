from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter
from agent.tool_registry import ToolRegistry
from schemas.user_context import UserContext
from skills.comparison import ComparisonSkill


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system: str, user: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _ctx():
    return UserContext(
        user_id="u1",
        company_id="c1",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )


def test_comparison_skill_drops_analysis_only_args_before_db_tool():
    registry = ToolRegistry()
    calls = []

    # Deliberately strict signature matching MaterialsTools.compare_samples.
    def strict_compare(*, left_identifier, right_identifier, ctx):
        calls.append((left_identifier, right_identifier, ctx.user_id))
        return {"status": "ok"}

    registry.register("compare_samples", "compare", strict_compare)
    skill = ComparisonSkill(registry)

    result = skill.execute(
        "compare_samples",
        {
            "left_identifier": "3811",
            "right_identifier": "3809",
            "target_metric": "冲击强度",
            "direction": "低",
            "project_id": 115,
        },
        _ctx(),
    )

    assert result == {"status": "ok"}
    assert calls == [("3811", "3809", "u1")]


def test_router_exact_reported_formula_question_drops_hallucinated_target_metric():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "compare",
        "primary_intent": "compare_samples",
        "tool_name": "compare_samples",
        "tool_args": {
            "left_identifier": "3811",
            "right_identifier": "3809",
            "target_metric": "冲击强度",
        },
    }))

    decision = router.route("3811和3809的配方差在哪?")

    assert decision.intent == "formula_difference"
    assert decision.tool_name == "compare_samples"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_router_generic_compare_never_forwards_target_metric_to_compare_tool():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "compare",
        "primary_intent": "compare_samples",
        "tool_name": "compare_samples",
        "tool_args": {
            "left_identifier": "3811",
            "right_identifier": "3809",
            "target_metric": "冲击强度",
        },
    }))

    decision = router.route("比较3811和3809")

    assert decision.intent == "compare_samples"
    assert decision.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
    }


def test_comparability_keeps_target_metric_because_skill_consumes_it():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "validate",
        "primary_intent": "comparability_check",
        "tool_name": "compare_samples",
        "tool_args": {
            "left_identifier": "3811",
            "right_identifier": "3809",
            "target_metric": "冲击强度",
        },
    }))

    decision = router.route("3811和3809的冲击强度能直接比较吗？")

    assert decision.intent == "comparability_check"
    assert decision.tool_args["target_metric"] == "冲击强度"
