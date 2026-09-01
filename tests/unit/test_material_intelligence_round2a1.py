from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter
from agent.core import AgentCore
from agent.tool_registry import ToolRegistry
from agent.tools import MaterialsTools
from schemas.user_context import UserContext
from skills.material_intelligence import MaterialIntelligenceSkill


class FakeRegistry:
    def __init__(self, *, sample=None, comparison=None):
        self.sample = sample
        self.comparison = comparison
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "get_sample_context":
            return self.sample
        if name == "compare_samples":
            return self.comparison
        raise AssertionError(name)


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


def _comparison(*, condition_status="missing_both", left_unit="kJ/m²", right_unit="kJ/m²"):
    return {
        "status": "ok",
        "left_sample": {"id": 3811, "name": "trial_6", "sample_type": None},
        "right_sample": {"id": 3809, "name": "trial_4", "sample_type": None},
        "formula_diff": {
            "changed": [
                {
                    "field": "ABS",
                    "left": "18.5",
                    "right": "33.2",
                    "unit": "%",
                    "left_unit": "%",
                    "right_unit": "%",
                    "unit_match": True,
                    "left_present": True,
                    "right_present": True,
                },
                {
                    "field": "PC",
                    "left": "59.7",
                    "right": "43.1",
                    "unit": "%",
                    "left_unit": "%",
                    "right_unit": "%",
                    "unit_match": True,
                    "left_present": True,
                    "right_present": True,
                },
            ],
            "same": [
                {
                    "field": "阻燃剂",
                    "left": "3",
                    "right": "3",
                    "unit": "%",
                    "left_unit": "%",
                    "right_unit": "%",
                    "unit_match": True,
                    "left_present": True,
                    "right_present": True,
                }
            ],
        },
        "process_diff": {
            "changed": [
                {
                    "field": "挤出温度",
                    "left": "250",
                    "right": "240",
                    "unit": "℃",
                    "left_unit": "℃",
                    "right_unit": "℃",
                    "unit_match": True,
                    "left_present": True,
                    "right_present": True,
                }
            ],
            "same": [],
        },
        "performance_diff": {
            "changed": [
                {
                    "field": "冲击强度",
                    "left": "24",
                    "right": "54",
                    "unit": left_unit or right_unit,
                    "left_unit": left_unit,
                    "right_unit": right_unit,
                    "unit_match": left_unit == right_unit,
                    "left_present": True,
                    "right_present": True,
                }
            ],
            "same": [],
        },
        "service_performance_diff": {"changed": [], "same": []},
        "test_conditions": {
            "status": condition_status,
            "same": True if condition_status == "same" else False if condition_status == "different" else None,
            "comparable": condition_status == "same",
            "left": {} if condition_status == "missing_both" else {"standard": "ISO"},
            "right": {} if condition_status == "missing_both" else {"standard": "ISO" if condition_status == "same" else "ASTM"},
        },
        "evidence": [{"source": "eln_sample", "record_id": 3811}],
        "warnings": [],
    }


def test_formula_difference_uses_decimal_backend_calculation():
    skill = MaterialIntelligenceSkill(FakeRegistry(comparison=_comparison()))
    result = skill.execute_intent(
        "formula_difference",
        "compare_samples",
        {"left_identifier": 3811, "right_identifier": 3809},
        _ctx(),
    )

    assert result["analysis_type"] == "formula_difference"
    abs_change = next(x for x in result["changed_fields"] if x["field"] == "ABS")
    assert abs_change["numeric_delta"]["left_minus_right"] == "-14.7"
    assert abs_change["numeric_delta"]["relative_to_right_percent"] == "-44.28"
    assert result["raw_numeric_totals"]["left_raw_numeric_sum"] == "81.2"
    assert result["raw_numeric_totals"]["right_raw_numeric_sum"] == "79.3"


def test_process_difference_is_scoped_to_process_fields():
    skill = MaterialIntelligenceSkill(FakeRegistry(comparison=_comparison()))
    result = skill.execute_intent(
        "process_difference",
        "compare_samples",
        {"left_identifier": 3811, "right_identifier": 3809},
        _ctx(),
    )

    assert result["analysis_type"] == "process_difference"
    assert [x["field"] for x in result["changed_fields"]] == ["挤出温度"]
    assert result["changed_fields"][0]["numeric_delta"]["left_minus_right"] == "10"
    assert "raw_numeric_totals" not in result


def test_comparability_missing_conditions_is_partial_not_same():
    skill = MaterialIntelligenceSkill(FakeRegistry(comparison=_comparison()))
    result = skill.execute_intent(
        "comparability_check",
        "compare_samples",
        {
            "left_identifier": 3811,
            "right_identifier": 3809,
            "target_metric": "冲击强度",
        },
        _ctx(),
    )

    assert result["assessment"]["grade"] == "PARTIALLY_COMPARABLE"
    assert result["assessment"]["strict_comparable_on_recorded_evidence"] is False
    assert any("测试条件均未记录" in x for x in result["assessment"]["evidence_gaps"])


def test_comparability_unit_mismatch_blocks_direct_comparison():
    comparison = _comparison(left_unit="kJ/m²", right_unit="J/m²")
    skill = MaterialIntelligenceSkill(FakeRegistry(comparison=comparison))
    result = skill.execute_intent(
        "comparability_check",
        "compare_samples",
        {
            "left_identifier": 3811,
            "right_identifier": 3809,
            "target_metric": "冲击强度",
        },
        _ctx(),
    )
    assert result["assessment"]["grade"] == "NOT_DIRECTLY_COMPARABLE"
    assert any("单位不一致" in x for x in result["assessment"]["blockers"])


def test_full_profile_adds_deterministic_coverage_metadata():
    sample = {
        "status": "ok",
        "sample": {"id": 3811, "name": "trial_6"},
        "formula": [{"name": "ABS", "value": 18}],
        "process": [{"name": "温度", "value": 250}],
        "performance": [{"name": "冲击强度", "value": 24}],
        "service_performance": [],
        "conditions": {},
        "recipe_batches": {},
        "craft_detail": None,
        "synthesis_records": [],
        "verify_items": [],
        "evidence": [],
        "warnings": [],
    }
    skill = MaterialIntelligenceSkill(FakeRegistry(sample=sample))
    result = skill.execute_intent(
        "sample_full_profile",
        "get_sample_context",
        {"identifier": 3811},
        _ctx(),
    )
    assert result["coverage"]["formula_fields"] == 1
    assert result["coverage"]["has_test_conditions"] is False


def test_router_specializes_generic_compare_to_formula_difference():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "compare",
        "primary_intent": "compare_samples",
        "tool_name": "compare_samples",
        "tool_args": {"left_identifier": 3811, "right_identifier": 3809},
    }))
    decision = router.route("3811和3809的配方差在哪？")
    assert decision.intent == "formula_difference"
    assert decision.tool_name == "compare_samples"
    assert decision.domain == "compare"


def test_router_specializes_generic_compare_to_process_difference():
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "compare_samples",
        "tool_name": "compare_samples",
        "tool_args": {"left_identifier": 3811, "right_identifier": 3809},
    }))
    decision = router.route("3811和3809工艺有哪些不同？")
    assert decision.intent == "process_difference"
    assert decision.tool_name == "compare_samples"


def test_router_specializes_direct_comparability_question():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "retrieve",
        "primary_intent": "compare_samples",
        "tool_name": "compare_samples",
        "tool_args": {"left_identifier": 3811, "right_identifier": 3809},
    }))
    decision = router.route("3811和3809的冲击强度能直接比较吗？")
    assert decision.intent == "comparability_check"
    assert decision.domain == "validate"
    assert decision.tool_args["target_metric"] == "冲击强度"


def test_router_full_profile_is_distinct_business_intent():
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "get_sample_context",
        "tool_name": "get_sample_context",
        "tool_args": {"identifier": 3811},
    }))
    decision = router.route("完整看看3811的所有信息")
    assert decision.intent == "sample_full_profile"
    assert decision.tool_name == "get_sample_context"
    assert decision.tool_args["identifier"] == 3811


def test_diff_fields_preserves_both_units_for_comparability_checks():
    left = [{"raw_key": "P1", "name": "冲击强度", "value": 24, "unit": "kJ/m²"}]
    right = [{"raw_key": "P1", "name": "冲击强度", "value": 24000, "unit": "J/m²"}]
    diff = MaterialsTools._diff_fields(left, right)
    item = diff["changed"][0]
    assert item["left_unit"] == "kJ/m²"
    assert item["right_unit"] == "J/m²"
    assert item["unit_match"] is False


def test_agent_core_executes_round2a_material_skill():
    registry = ToolRegistry()
    registry.register("compare_samples", "compare", lambda **kwargs: _comparison())
    registry.register("get_sample_context", "sample", lambda **kwargs: {"status": "ok"})

    core = AgentCore(registry=registry, llm=FakeLLM({}), llm_enabled=False)
    result = core.execute(
        "formula_difference",
        "compare_samples",
        {"left_identifier": 3811, "right_identifier": 3809},
        _ctx(),
    )
    assert result["analysis_type"] == "formula_difference"
    assert result["summary"]["changed_count"] == 2
