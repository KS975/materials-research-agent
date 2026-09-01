from __future__ import annotations

import json

from agent.core import AgentCore
from agent.deepseek_intent_router import DeepSeekIntentRouter
from agent.router import RuleIntentRouter
from schemas.user_context import UserContext
from runtime.company_data_conversation import company_data_has_priority
from skills.material_intelligence import MaterialIntelligenceSkill


class FakeRegistry:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        assert name == "list_samples_for_analysis"
        return self.payload


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        return json.dumps(self.payload, ensure_ascii=False)


def ctx():
    return UserContext(
        user_id="u1",
        company_id="c1",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )


def field(name, value, unit=""):
    return {"name": name, "value": value, "unit": unit, "resolved": True}


def source():
    rows = [
        {
            "sample": {"id": 3814, "name": "trial_9", "project_id": 115, "sample_type": "A"},
            "formula": [field("PC", "60", "%"), field("ABS", "40", "%")],
            "process": [field("注塑温度", "80", "℃")],
            "performance": [
                field("冲击强度", "54", "kJ/m²"),
                field("成本", "25", "元/kg"),
            ],
            "conditions": {"测试标准": "ISO 180"},
        },
        {
            "sample": {"id": 3813, "name": "trial_8", "project_id": 115, "sample_type": "A"},
            "formula": [field("PC", "55", "%"), field("ABS", "45", "%")],
            "process": [field("注塑温度", "65", "℃")],
            "performance": [
                field("冲击强度", "45", "kJ/m²"),
                field("成本", "32", "元/kg"),
            ],
            "conditions": {},
        },
        {
            "sample": {"id": 3812, "name": "trial_7", "project_id": 116, "sample_type": None},
            "formula": [field("PC", "40", "%"), field("ABS", "60", "%")],
            "process": [field("注塑温度", "75", "℃")],
            "performance": [
                field("冲击强度", "bad", "kJ/m²"),
                field("成本", "20", "元/kg"),
            ],
            "conditions": {"测试标准": "ASTM"},
        },
    ]
    return {
        "status": "ok",
        "keyword": "",
        "count": 3,
        "total_matches": 3,
        "scan_page_size": 500,
        "scan_page_count": 1,
        "scan_complete": True,
        "scan_truncated": False,
        "samples": rows,
        "evidence": [
            {"source": "eln_sample", "record_id": row["sample"]["id"]}
            for row in rows
        ],
        "warnings": [],
    }


def run_filters(filters, *, logic="and", result_limit=50, payload=None):
    registry = FakeRegistry(payload or source())
    result = MaterialIntelligenceSkill(registry).execute_intent(
        "find_samples_multi_condition",
        "list_samples_for_analysis",
        {
            "filters": filters,
            "logic": logic,
            "result_limit": result_limit,
        },
        ctx(),
    )
    return result, registry


def test_and_filter_is_deterministic_across_performance_fields():
    result, _ = run_filters([
        {"section": "performance", "field": "冲击强度", "operator": "gt", "value": 40},
        {"section": "performance", "field": "成本", "operator": "lt", "value": 30},
    ])
    assert result["status"] == "ok"
    assert result["matched_sample_count"] == 1
    assert result["matched_samples"][0]["sample"]["id"] == 3814
    assert result["excluded_sample_count"] == 2


def test_formula_process_and_project_conditions_can_be_combined():
    result, _ = run_filters([
        {"section": "sample", "field": "project_id", "operator": "eq", "value": 115},
        {"section": "formula", "field": "PC", "operator": "gt", "value": 50, "unit": "%"},
        {"section": "process", "field": "注塑温度", "operator": "gte", "value": 70, "unit": "℃"},
    ])
    assert [row["sample"]["id"] for row in result["matched_samples"]] == [3814]


def test_or_between_and_condition_missing_are_supported():
    ranged, _ = run_filters([
        {"section": "performance", "field": "冲击强度", "operator": "between", "values": [44, 50]},
    ])
    assert [row["sample"]["id"] for row in ranged["matched_samples"]] == [3813]

    missing, _ = run_filters([
        {"section": "conditions", "field": "*", "operator": "missing"},
    ])
    assert [row["sample"]["id"] for row in missing["matched_samples"]] == [3813]

    either, _ = run_filters([
        {"section": "performance", "field": "冲击强度", "operator": "gt", "value": 50},
        {"section": "performance", "field": "成本", "operator": "lt", "value": 22},
    ], logic="or")
    assert [row["sample"]["id"] for row in either["matched_samples"]] == [3814, 3812]


def test_mixed_units_without_explicit_unit_fail_closed():
    payload = source()
    payload["samples"][1]["performance"][0]["unit"] = "J/m²"
    result, _ = run_filters([
        {"section": "performance", "field": "冲击强度", "operator": "gt", "value": 40},
    ], payload=payload)
    assert result["status"] == "unit_ambiguity"
    assert result["matched_samples"] == []


def test_explicit_unit_only_matches_same_unit_records():
    payload = source()
    payload["samples"][1]["performance"][0]["unit"] = "J/m²"
    result, _ = run_filters([
        {
            "section": "performance",
            "field": "冲击强度",
            "operator": "gt",
            "value": 40,
            "unit": "kJ/m²",
        },
    ], payload=payload)
    assert result["status"] == "ok"
    assert [row["sample"]["id"] for row in result["matched_samples"]] == [3814]
    assert result["filter_diagnostics"][0]["outcomes"]["unit_mismatch"] == 1


def test_duplicate_and_non_numeric_fields_are_excluded_with_diagnostics():
    payload = source()
    payload["samples"][0]["performance"].append(field("冲击强度", "53", "kJ/m²"))
    result, _ = run_filters([
        {"section": "performance", "field": "冲击强度", "operator": "gt", "value": 40},
    ], payload=payload)
    assert [row["sample"]["id"] for row in result["matched_samples"]] == [3813]
    outcomes = result["filter_diagnostics"][0]["outcomes"]
    assert outcomes["ambiguous"] == 1
    assert outcomes["non_numeric"] == 1


def test_result_limit_does_not_change_total_match_count_or_scan_count():
    result, _ = run_filters([
        {"section": "performance", "field": "成本", "operator": "gt", "value": 0},
    ], result_limit=2)
    assert result["matched_sample_count"] == 3
    assert result["returned_sample_count"] == 2
    assert result["results_truncated"] is True
    assert result["scanned_sample_count"] == 3


def test_invalid_filter_set_is_rejected_before_database_read():
    result, registry = run_filters([
        {"section": "performance", "field": "成本", "operator": "gt", "value": 0},
        {"section": "sql", "field": "DROP TABLE", "operator": "eq", "value": 1},
    ])
    assert result["status"] == "invalid_filters"
    assert registry.calls == []


def test_unknown_field_does_not_silently_return_zero_matches():
    result, _ = run_filters([
        {"section": "performance", "field": "不存在的性能", "operator": "gt", "value": 1},
    ])
    assert result["status"] == "field_not_found"
    assert result["unknown_filter_fields"][0]["field"] == "不存在的性能"


def test_deepseek_router_sanitizes_filters_and_strips_unstated_units():
    decision = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "find_samples_multi_condition",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {
            "filters": [{
                "section": "性能",
                "field": "冲击强度",
                "operator": "高于",
                "value": 40,
                "unit": "kJ/m²",
                "sql": "DROP TABLE eln_sample",
            }],
            "logic": "AND",
            "evil": "ignored",
        },
    })).route("找冲击强度大于40的样品")
    assert decision.intent == "find_samples_multi_condition"
    assert decision.tool_name == "list_samples_for_analysis"
    assert decision.tool_args["filters"] == [{
        "section": "performance",
        "field": "冲击强度",
        "operator": "gt",
        "value": 40,
    }]
    assert decision.scope["data_source"] == "business_mysql"
    assert decision.constraints["arbitrary_sql"] is False


def test_explicit_user_unit_is_preserved_by_router():
    decision = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "find_samples_multi_condition",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {
            "filters": [{
                "section": "performance",
                "field": "冲击强度",
                "operator": "gt",
                "value": 40,
                "unit": "kJ/m²",
            }],
        },
    })).route("找冲击强度大于40 kJ/m²的样品")
    assert decision.tool_args["filters"][0]["unit"] == "kJ/m²"


def test_one_invalid_filter_causes_clarification_not_partial_execution():
    decision = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "find_samples_multi_condition",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {
            "filters": [
                {"section": "performance", "field": "成本", "operator": "lt", "value": 30},
                {"section": "database", "field": "password", "operator": "eq", "value": "x"},
            ],
        },
    })).route("找成本低于30的样品")
    assert decision.needs_clarification is True
    assert decision.tool_args["filters"] == []
    assert "安全解析" in decision.clarification_question


def test_router_rejects_invented_project_scope_and_unstated_keyword():
    decision = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "find_samples_multi_condition",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {
            "keyword": "N20260305",
            "filters": [
                {"section": "sample", "field": "project_id", "operator": "eq", "value": 115},
                {"section": "performance", "field": "成本", "operator": "lt", "value": 30},
            ],
        },
    })).route("找成本低于30的样品")
    assert decision.needs_clarification is True
    assert decision.tool_args["filters"] == []
    assert decision.tool_args["keyword"] == ""


def test_router_keeps_explicit_project_filter_and_series_keyword():
    decision = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "find_samples_multi_condition",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {
            "keyword": "N20260305",
            "filters": [
                {"section": "sample", "field": "project_id", "operator": "eq", "value": 115},
                {"section": "performance", "field": "成本", "operator": "lt", "value": 30},
            ],
        },
    })).route("找项目115中N20260305系列成本低于30的样品")
    assert decision.needs_clarification is False
    assert decision.tool_args["keyword"] == "N20260305"
    assert decision.tool_args["filters"][0]["field"] == "project_id"


def test_rule_fallback_never_turns_condition_sentence_into_name_like_search():
    router = RuleIntentRouter()
    assert router.route("找冲击强度大于40、成本低于30的样品") is None
    simple = router.route("找 trial_9")
    assert simple.intent == "find_samples"


def test_implicit_haike_global_route_cannot_preempt_business_mysql_filtering():
    assert company_data_has_priority({
        "route": True,
        "explicit_company_scope": False,
        "conversation_scope": {
            "product_type": None,
            "source": "GLOBAL_DEFAULT",
        },
    }) is False
    assert company_data_has_priority({
        "route": True,
        "explicit_company_scope": True,
        "conversation_scope": {
            "product_type": None,
            "source": "GLOBAL_DEFAULT",
        },
    }) is True


def test_deterministic_answer_reports_backend_counts_only():
    result, _ = run_filters([
        {"section": "performance", "field": "成本", "operator": "lt", "value": 30},
    ])
    answer = AgentCore._deterministic_answer("find_samples_multi_condition", result)
    assert "共找到 2 个" in answer
    assert "3814" in answer and "3812" in answer
    assert "已扫描 3 / 3" in answer
