from __future__ import annotations

import json
from types import SimpleNamespace

from agent.deepseek_intent_router import DeepSeekIntentRouter
from agent.field_catalog import build_material_field_catalog
from agent.router import RuleIntentRouter
from schemas.user_context import UserContext
from skills.material_intelligence import MaterialIntelligenceSkill


class FakeRegistry:
    def __init__(self, source):
        self.source = source
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        assert name == "list_samples_for_analysis"
        return self.source


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        return json.dumps(self.payload, ensure_ascii=False)


class FakeAnalysisSamples:
    def __init__(self):
        self.calls = []

    def list_for_analysis(self, keyword, context, limit, before_id=None):
        self.calls.append({"limit": limit, "before_id": before_id})
        rows = []
        for sample_id in range(575, 0, -1):
            if before_id is not None and sample_id >= before_id:
                continue
            rows.append({
                "id": sample_id,
                "name": f"A-{sample_id}",
                "project_id": 115,
                "sample_type": None,
                "create_time": None,
                "recipes": {"R3-10": "50"},
                "craft_param": {"S20": "230"},
                "performances": {"P30": "24"},
                "conditions": {},
            })
        return rows[:limit]

    def count_for_analysis(self, keyword, context):
        return 575


class FakeDefinitions:
    def __init__(self, definitions):
        self.definitions = definitions
        self.calls = []

    def get_sample_materials(self, ids, company_id):
        self.calls.append((set(ids), company_id))
        return self.definitions

    def get_by_ids(self, ids, company_id):
        self.calls.append((set(ids), company_id))
        return self.definitions


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
    return {
        "status": "ok",
        "keyword": "N20260305",
        "count": 3,
        "samples": [
            {
                "sample": {"id": 1, "name": "N20260305-1", "project_id": 115},
                "formula": [field("ABS", "50", "%"), field("PC", "50", "%")],
                "process": [field("温度", "230", "℃")],
                "performance": [field("冲击强度", "24", "kJ/m²")],
                "conditions": {"standard": "ISO"},
            },
            {
                "sample": {"id": 2, "name": "N20260305-2", "project_id": 115},
                "formula": [field("ABS", "60", "%"), field("PC", "40", "%")],
                "process": [field("温度", "230", "℃")],
                "performance": [field("冲击强度", "54", "kJ/m²")],
                "conditions": {},
            },
            {
                "sample": {"id": 3, "name": "N20260305-2", "project_id": 115},
                "formula": [field("ABS", "70", "%"), field("PC", "40", "%")],
                "process": [],
                "performance": [field("冲击强度", "bad", "kJ/m²")],
                "conditions": {},
            },
        ],
        "unresolved_dynamic_fields": [],
        "evidence": [{"source": "eln_sample", "record_id": x} for x in (1, 2, 3)],
        "warnings": [],
    }


def n20260305_series_source():
    temperatures = ("55", "58", "61", "64")
    bet_values = ("25.7", "26.32", "26.36", "26.37")
    tg_values = ("4.01", "4.04", "4.07", "4.13")
    rows = []
    for project_id, first_id in ((-1540, 2485), (-1539, 2426)):
        for index, (temperature, bet, tg) in enumerate(
            zip(temperatures, bet_values, tg_values),
            1,
        ):
            rows.append({
                "sample": {
                    "id": first_id + index - 1,
                    "name": f"N20260305-{index}",
                    "project_id": project_id,
                },
                "formula": [],
                "process": [
                    field("处理温度", temperature, "℃"),
                    field("处理时间", "45", "min"),
                    field("处理转速", "2600", "r_min"),
                    field(
                        "表面处理工艺流程",
                        "3.9%（65%1840+35%CBO+15%CS）+0.4%YS50，2600rpm/min处理45min",
                    ),
                ],
                "performance": [
                    field("BET", bet, "m²"),
                    field("TG", tg, "%"),
                    field("全组未填写性能", None, "MPa"),
                ],
                "conditions": {},
            })
    return {
        "status": "ok",
        "keyword": "N20260305",
        "count": 8,
        "total_matches": 8,
        "samples": rows,
        "similar_names": [],
        "unresolved_dynamic_fields": [],
        "evidence": [
            {"source": "eln_sample", "record_id": row["sample"]["id"]}
            for row in rows
        ],
        "warnings": [],
    }


def test_performance_rank_is_decimal_sorted_and_excludes_non_numeric():
    result = MaterialIntelligenceSkill(FakeRegistry(source())).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "冲击强度", "keyword": "", "top_n": 5, "order": "desc"},
        ctx(),
    )
    assert [row["sample"]["id"] for row in result["ranking"]] == [2, 1]
    assert result["ranking"][0]["value"] == "54"
    assert result["excluded_samples"][0]["sample"]["id"] == 3


def test_performance_rank_refuses_mixed_units():
    payload = source()
    payload["samples"][1]["performance"][0]["unit"] = "J/m²"
    result = MaterialIntelligenceSkill(FakeRegistry(payload)).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "冲击强度"},
        ctx(),
    )
    assert result["status"] == "unit_mismatch"
    assert result["ranking"] == []


def test_experiment_series_reports_constant_variable_and_missing():
    result = MaterialIntelligenceSkill(FakeRegistry(source())).execute_intent(
        "experiment_series_analysis",
        "list_samples_for_analysis",
        {"keyword": "N20260305"},
        ctx(),
    )
    constant_names = {x["field"] for x in result["constant_fields"]}
    variable_names = {x["field"] for x in result["variable_fields"]}
    assert "温度" not in constant_names
    assert {"ABS", "PC", "温度", "冲击强度"}.issubset(variable_names)
    assert result["sample_count"] == 3


def test_series_analysis_separates_missing_projects_and_design_inference():
    result = MaterialIntelligenceSkill(
        FakeRegistry(n20260305_series_source())
    ).execute_intent(
        "experiment_series_analysis",
        "list_samples_for_analysis",
        {"keyword": "N20260305"},
        ctx(),
    )

    constant_names = {item["field"] for item in result["constant_fields"]}
    missing_names = {item["field"] for item in result["missing_fields"]}
    assert "全组未填写性能" in missing_names
    assert "全组未填写性能" not in constant_names

    groups = result["project_groups"]
    assert {group["project_id"] for group in groups} == {-1540, -1539}
    assert {group["sample_count"] for group in groups} == {4}
    assessment = result["cross_project_assessment"]
    assert assessment["requires_separate_analysis"] is True
    assert assessment["same_sample_name_pattern"] is True
    assert assessment["same_recorded_design"] is True

    inference = result["purpose_inference"]
    assert inference["confidence"] == "medium_high"
    assert [
        item["field"] for item in inference["candidate_independent_factors"]
    ] == ["处理温度"]
    assert {item["field"] for item in inference["controlled_factors"]} >= {
        "处理时间", "处理转速", "表面处理工艺流程",
    }
    assert {item["field"] for item in inference["response_metrics"]} == {
        "BET", "TG",
    }
    assert "55/58/61/64 ℃" in inference["summary"]
    assert "不能直接合并" in inference["summary"]
    assert "不证明" in inference["causality_limit"]


def test_data_quality_counts_are_deterministic():
    result = MaterialIntelligenceSkill(FakeRegistry(source())).execute_intent(
        "data_quality_check",
        "list_samples_for_analysis",
        {"keyword": "N20260305"},
        ctx(),
    )
    assert result["summary"]["duplicate_name_count"] == 1
    assert result["summary"]["missing_condition_count"] == 2
    assert result["summary"]["missing_condition_percent"] == "66.67"
    assert result["summary"]["non_numeric_performance_count"] == 1
    assert result["summary"]["formula_total_warning_count"] == 1


def test_rule_router_routes_round2a2_examples():
    router = RuleIntentRouter()
    assert router.route("哪几个样品冲击强度最好").intent == "performance_rank"
    assert router.route("N20260305这一组在研究什么").intent == "experiment_series_analysis"
    assert router.route("N20260305这一组实验在研究什么？").intent == "experiment_series_analysis"
    assert router.route("帮我看看这些数据有没有问题").intent == "data_quality_check"


def test_rank_router_strips_request_fillers_and_keeps_top_n():
    router = RuleIntentRouter()
    for message in (
        "给我冲击强度最高的前5个样品",
        "请给我冲击强度最高的前5个样品",
        "帮我找冲击强度最高的前5个样品",
        "麻烦给我冲击强度最高的前5个样品",
        "找冲击强度最高的前5个样品",
        "哪个样品冲击强度最高的前5个",
        "所有样品中冲击强度最高的前5个",
    ):
        decision = router.route(message)
        assert decision.intent == "performance_rank"
        assert decision.tool_args["target_metric"] == "冲击强度"
        assert decision.tool_args["top_n"] == 5


def test_collection_analysis_defaults_to_500_row_page_size():
    registry = FakeRegistry(source())
    MaterialIntelligenceSkill(registry).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "冲击强度"},
        ctx(),
    )
    assert registry.calls[0][1]["limit"] == 500


def test_empty_series_preserves_deterministic_name_suggestions():
    payload = {
        "status": "ok",
        "keyword": "N20260305",
        "count": 0,
        "total_matches": 0,
        "samples": [],
        "similar_names": [{"id": 9, "name": "N2026-OTHER", "project_id": 115}],
        "unresolved_dynamic_fields": [],
        "evidence": [],
        "warnings": [],
    }
    result = MaterialIntelligenceSkill(FakeRegistry(payload)).execute_intent(
        "experiment_series_analysis",
        "list_samples_for_analysis",
        {"keyword": "N20260305"},
        ctx(),
    )
    assert result["sample_count"] == 0
    assert result["similar_names"][0]["name"] == "N2026-OTHER"


def test_deepseek_router_normalizes_round2a2_tool_and_args():
    decision = DeepSeekIntentRouter(FakeLLM({
        "domain": "analyze",
        "primary_intent": "performance_rank",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {"target_metric": "冲击强度", "top_n": 5, "evil": "drop"},
        "needs_clarification": False,
    })).route("冲击强度最高的前5个样品")
    assert decision.intent == "performance_rank"
    assert decision.tool_name == "list_samples_for_analysis"
    assert decision.tool_args["target_metric"] == "冲击强度"
    assert "evil" not in decision.tool_args


def test_deepseek_router_fallback_uses_same_rank_metric_normalization():
    decision = DeepSeekIntentRouter(FakeLLM({
        "domain": "analyze",
        "primary_intent": "performance_rank",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {},
        "needs_clarification": False,
    })).route("给我冲击强度最高的前5个样品")
    assert decision.intent == "performance_rank"
    assert decision.tool_args["target_metric"] == "冲击强度"
    assert decision.tool_args["top_n"] == 5


def test_deepseek_rank_metric_cannot_keep_dashboard_request_verb():
    decision = DeepSeekIntentRouter(FakeLLM({
        "domain": "analyze",
        "primary_intent": "performance_rank",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {"target_metric": "找拉伸强度"},
        "needs_clarification": False,
    })).route("找拉伸强度最高的样品")
    assert decision.intent == "performance_rank"
    assert decision.tool_args["target_metric"] == "拉伸强度"


def test_deepseek_rank_drops_model_invented_section_and_keeps_explicit_user_section():
    router = DeepSeekIntentRouter(FakeLLM({
        "domain": "analyze",
        "primary_intent": "performance_rank",
        "tool_name": "list_samples_for_analysis",
        "tool_args": {"target_metric": "增韧剂含量", "target_section": "performance"},
        "needs_clarification": False,
    }))
    inferred = router.route("找增韧剂含量最高的样品")
    assert "target_section" not in inferred.tool_args

    explicit = router.route("找配方增韧剂最高的样品")
    assert explicit.tool_args["target_section"] == "formula"


def test_rank_binds_canonical_metric_and_reports_quality_reasons():
    payload = {
        "status": "ok",
        "count": 5,
        "total_matches": 5,
        "samples": [
            {"sample": {"id": 1, "name": "A"}, "performance": [field("拉伸强度", "52", "MPa")]},
            {"sample": {"id": 2, "name": "B"}, "performance": [field("拉伸强度", "", "MPa")]},
            {"sample": {"id": 3, "name": "C"}, "performance": [field("拉伸强度", "待复测", "MPa")]},
            {"sample": {"id": 4, "name": "D"}, "performance": [field("拉伸强度", "45", "MPa"), field("拉伸强度", "46", "MPa")]},
            {"sample": {"id": 5, "name": "E"}, "performance": [field("冲击强度", "24", "kJ/m²")]},
        ],
        "warnings": [],
        "evidence": [],
    }
    result = MaterialIntelligenceSkill(FakeRegistry(payload)).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "找性能拉伸强度指标最高", "top_n": 5},
        ctx(),
    )

    assert result["status"] == "ok"
    assert result["requested_target_metric"] == "找性能拉伸强度指标最高"
    assert result["target_metric"] == "拉伸强度"
    assert result["field_binding"]["status"] == "ok"
    assert [row["sample"]["id"] for row in result["ranking"]] == [1]
    assert result["numeric_sample_count"] == 1
    assert result["field_absent_sample_count"] == 1
    assert result["empty_value_sample_count"] == 1
    assert result["non_numeric_sample_count"] == 1
    assert result["ambiguous_sample_count"] == 1

    catalog = build_material_field_catalog(payload)
    tensile = next(
        item for item in catalog["sections"]["performance"]
        if item["name"] == "拉伸强度"
    )
    assert tensile["observed_sample_count"] == 4
    assert tensile["numeric_sample_count"] == 1
    assert tensile["empty_value_sample_count"] == 1
    assert tensile["non_numeric_sample_count"] == 1
    assert tensile["ambiguous_sample_count"] == 1


def test_rank_binds_formula_quantity_suffix_to_canonical_formula_field():
    payload = {
        "status": "ok",
        "count": 3,
        "total_matches": 3,
        "samples": [
            {"sample": {"id": 1, "name": "A"}, "formula": [field("增韧剂", "12", "%")]},
            {"sample": {"id": 2, "name": "B"}, "formula": [field("增韧剂", "28", "%")]},
            {"sample": {"id": 3, "name": "C"}, "formula": [field("PC", "60", "%")]},
        ],
        "warnings": [],
        "evidence": [],
    }
    result = MaterialIntelligenceSkill(FakeRegistry(payload)).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "增韧剂含量", "top_n": 10},
        ctx(),
    )

    assert result["status"] == "ok"
    assert result["target_section"] == "formula"
    assert result["target_section_label"] == "配方"
    assert result["target_metric"] == "增韧剂"
    assert [row["sample"]["id"] for row in result["ranking"]] == [2, 1]
    assert result["field_absent_sample_count"] == 1

    catalog = build_material_field_catalog(payload)
    additive = next(
        item for item in catalog["sections"]["formula"]
        if item["name"] == "增韧剂"
    )
    assert additive["numeric_sample_count"] == 2


def test_rank_and_average_bind_process_and_formula_fields():
    payload = {
        "status": "ok",
        "count": 2,
        "total_matches": 2,
        "samples": [
            {
                "sample": {"id": 1, "name": "A"},
                "formula": [field("PC", "40", "%")],
                "process": [field("注塑温度", "220", "℃")],
            },
            {
                "sample": {"id": 2, "name": "B"},
                "formula": [field("PC", "60", "%")],
                "process": [field("注塑温度", "240", "℃")],
            },
        ],
        "warnings": [],
        "evidence": [],
    }
    skill = MaterialIntelligenceSkill(FakeRegistry(payload))
    ranking = skill.execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "工艺注塑温度", "target_section": "process"},
        ctx(),
    )
    mean = skill.execute_intent(
        "performance_statistics",
        "list_samples_for_analysis",
        {"target_metric": "PC含量"},
        ctx(),
    )

    assert ranking["target_section"] == "process"
    assert ranking["target_metric"] == "注塑温度"
    assert [row["sample"]["id"] for row in ranking["ranking"]] == [2, 1]
    assert mean["status"] == "ok"
    assert mean["target_section"] == "formula"
    assert mean["target_metric"] == "PC"
    assert mean["statistics"]["mean_display"] == "50"


def test_cross_section_same_name_requires_explicit_category():
    payload = {
        "status": "ok",
        "count": 1,
        "total_matches": 1,
        "samples": [{
            "sample": {"id": 1, "name": "A"},
            "formula": [field("指数", "1", "%")],
            "performance": [field("指数", "2", "MPa")],
        }],
        "warnings": [],
        "evidence": [],
    }
    result = MaterialIntelligenceSkill(FakeRegistry(payload)).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "指数"},
        ctx(),
    )
    assert result["status"] == "ambiguous_field"
    assert result["field_binding"]["candidates"] == ["配方.指数", "性能.指数"]


def test_exact_performance_name_wins_before_formula_quantity_suffix_hint():
    payload = {
        "status": "ok",
        "count": 1,
        "total_matches": 1,
        "samples": [{
            "sample": {"id": 1, "name": "A"},
            "formula": [field("氧", "1", "%")],
            "performance": [field("氧含量", "2", "%")],
        }],
        "warnings": [],
        "evidence": [],
    }
    result = MaterialIntelligenceSkill(FakeRegistry(payload)).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "氧含量"},
        ctx(),
    )
    assert result["status"] == "ok"
    assert result["target_section"] == "performance"
    assert result["target_metric"] == "氧含量"


def test_collection_tool_reads_all_pages_and_prefetches_definitions_once():
    from agent.tools import MaterialsTools

    samples = FakeAnalysisSamples()
    materials = FakeDefinitions({10: {"name": "ABS", "unit": "%"}})
    columns = FakeDefinitions({
        20: {"name": "温度", "unit": "℃"},
        30: {"name": "冲击强度", "unit": "kJ/m²"},
    })
    tools = MaterialsTools(
        samples=samples,
        projects=None,
        archives=None,
        experiments=None,
        resolver=SimpleNamespace(materials=materials, columns=columns),
    )
    result = tools.list_samples_for_analysis("", ctx(), limit=9999)
    assert samples.calls == [
        {"limit": 500, "before_id": None},
        {"limit": 500, "before_id": 76},
    ]
    assert result["scan_limit"] is None
    assert result["scan_page_size"] == 500
    assert result["scan_page_count"] == 2
    assert result["scan_complete"] is True
    assert result["scan_truncated"] is False
    assert result["count"] == 575
    assert result["total_matches"] == 575
    assert materials.calls == [({10}, "c1")]
    assert columns.calls == [({20, 30}, "c1")]
    assert result["samples"][-1]["performance"][0]["name"] == "冲击强度"

    analysis = MaterialIntelligenceSkill(FakeRegistry(result)).execute_intent(
        "performance_rank",
        "list_samples_for_analysis",
        {"target_metric": "冲击强度", "top_n": 5},
        ctx(),
    )
    assert analysis["scanned_sample_count"] == 575
    assert analysis["total_matching_sample_count"] == 575
    assert analysis["scan_page_count"] == 2
    assert analysis["scan_truncated"] is False
    assert len(analysis["ranking"]) == 5
