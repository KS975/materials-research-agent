from __future__ import annotations

from types import SimpleNamespace

from agent.research_analysis import (
    analyze_key_variables,
    build_evidence_dataset,
    discover_process_window,
    run_research_analysis,
)
from agent.research_scenarios import resolve_research_scenario
from schemas.user_context import UserContext
from skills.hybrid_research_qa import HybridResearchQASkill


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-1",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


def _source() -> dict:
    pc_values = (10, 20, 30, 40, 50, 60)
    temperature_values = (180, 181, 182, 183, 184, 185)
    impact_values = (32, 34, 36, 38, 40, 42)
    mfr_values = (20, 19, 18, 17, 16, 15)
    samples = []
    for index in range(6):
        samples.append(
            {
                "sample": {
                    "id": 128 + index,
                    "name": f"EXP-{128 + index}",
                    "project_id": 115,
                    "create_time": f"2026-01-0{index + 1}",
                },
                "formula": [
                    {"name": "PC", "raw_key": f"R3-{index + 1}", "value": pc_values[index], "unit": "%"}
                ],
                "process": [
                    {"name": "加工温度", "raw_key": f"S{index + 1}", "value": temperature_values[index], "unit": "°C"}
                ],
                "performance": [
                    {"name": "冲击强度", "raw_key": f"P{index + 1}", "value": impact_values[index], "unit": "kJ/m²"},
                    {"name": "MFR", "raw_key": f"P{index + 9}", "value": mfr_values[index], "unit": "g/10min"},
                ],
            }
        )
    return {
        "status": "ok",
        "count": len(samples),
        "total_matches": len(samples),
        "scan_complete": True,
        "samples": samples,
        "evidence": [
            {"source": "eln_sample", "record_id": item["sample"]["id"]}
            for item in samples
        ],
        "warnings": [],
    }


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "list_samples_for_analysis":
            return _source()
        if name == "search_vector_knowledge":
            return {
                "status": "ok",
                "provider": "external_vector_api",
                "hit_count": 1,
                "hits": [
                    {
                        "point_id": "point-1",
                        "score": 0.8,
                        "text": "阶段项目资料。",
                        "metadata": {"company_id": "company-a", "project_id": 115},
                    }
                ],
                "warnings": [],
            }
        raise AssertionError(f"unexpected tool: {name}")


class FakeLLM:
    def complete(self, system, user):
        return "综合结论。"


def _skill():
    registry = FakeRegistry()
    return HybridResearchQASkill(
        registry=registry,
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *_args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )


def test_stage4_scenarios_are_supported_read_only_workflows():
    for scenario_id in (8, 9, 10, 11, 18):
        plan = resolve_research_scenario("开放分析问题", {"scenario_id": scenario_id})
        assert plan["execution_status"] == "SUPPORTED"
        assert plan["read_only"] is True
        assert plan["write_policy"] == "NO_WRITE"

    assert resolve_research_scenario("哪些变量影响冲击强度？")["scenario_id"] == 8
    assert resolve_research_scenario("找冲击强度和MFR的稳定工艺窗口")["scenario_id"] == 9
    assert resolve_research_scenario("冲击强度和MFR为什么冲突？")["scenario_id"] == 10
    assert resolve_research_scenario("分析正常批次和异常批次的差异")["scenario_id"] == 11
    assert resolve_research_scenario("生成阶段总结报告")["scenario_id"] == 18


def test_evidence_dataset_and_key_variable_analysis_are_traceable():
    dataset = build_evidence_dataset(_source())
    result = analyze_key_variables(
        dataset,
        {"target_metrics": ["冲击强度"]},
    )

    assert result["status"] == "ok"
    assert result["target"] == "performance.冲击强度"
    assert result["target_unit"] == "kJ/m²"
    assert result["variables"][0]["field"] == "formula.PC"
    assert result["variables"][0]["correlation"] == 1.0
    assert len(result["evidence_refs"]) == 6
    assert "相关性" in result["conclusion_limit"]


def test_key_variable_hybrid_result_adds_derived_evidence_and_chart_data():
    result = _skill().answer(
        message="影响冲击强度的关键变量有哪些？结合历史资料分析。",
        tool_args={"project_id": 115},
        ctx=_ctx(),
    )

    assert result["research_workflow"]["scenario_id"] == 8
    assert result["structured_strategy"]["strategy"] == "authorized_evidence_dataset_scan"
    assert result["analysis_result"]["status"] == "ok"
    assert result["chart_data"][0]["chart_type"] == "bar"
    assert result["evidence_frame"]["source_summary"]["derived"] == 1
    assert result["synthesis"]["mode"] == "deterministic_analysis_report"
    assert result["synthesis"]["citation_validation"]["valid_count"] >= 1
    assert "derived-" in result["answer"]


def test_conflict_batch_window_and_stage_report_analysis():
    source = _source()
    conflict = run_research_analysis(
        scenario_id=10,
        source=source,
        args={"target_metrics": ["冲击强度", "MFR"]},
        message="冲击强度和MFR冲突",
    )
    assert conflict["status"] == "ok"
    assert conflict["target_correlation"] == -1.0
    assert conflict["conflicts"][0]["field"] == "formula.PC"

    batch = run_research_analysis(
        scenario_id=11,
        source=source,
        args={
            "batch_groups": {
                "normal": [128, 129, 130],
                "abnormal": [133, 134, 133],
            }
        },
        message="批次差异",
    )
    assert batch["status"] == "ok"
    assert batch["compared_groups"] == ["abnormal", "normal"]
    assert batch["differences"]

    window = discover_process_window(
        build_evidence_dataset(source),
        {
            "filters": [
                {"section": "performance", "field": "冲击强度", "operator": "gt", "value": 35},
                {"section": "performance", "field": "MFR", "operator": "lt", "value": 18},
            ]
        },
        message="稳定窗口",
    )
    assert window["status"] == "ok"
    assert window["feasible_sample_count"] == 3
    assert window["windows"]
    assert window["windows"][0]["recommended_min"] <= window["windows"][0]["recommended_max"]

    report = run_research_analysis(
        scenario_id=18,
        source=source,
        args={},
        message="阶段总结",
    )
    assert report["status"] == "ok"
    assert report["scope"]["sample_count"] == 6
    assert report["target_statistics"][0]["sample_count"] == 6


def test_process_window_does_not_extrapolate_without_feasible_samples():
    result = discover_process_window(
        build_evidence_dataset(_source()),
        {
            "filters": [
                {"section": "performance", "field": "冲击强度", "operator": "gt", "value": 100}
            ]
        },
    )
    assert result["status"] == "no_feasible_samples"
    assert any("不外推" in item for item in result["warnings"])


def test_evidence_dataset_excludes_mixed_or_missing_units():
    source = _source()
    source["samples"][0]["formula"][0]["unit"] = None
    source["samples"][1]["performance"][0]["unit"] = "MPa"
    dataset = build_evidence_dataset(source)

    assert dataset["rows"][0]["values"].pop("formula.PC", None) is None
    assert dataset["rows"][0]["values"].pop("performance.冲击强度", None) is None
    pc_profile = next(
        item for item in dataset["field_catalog"] if item["key"] == "formula.PC"
    )
    impact_profile = next(
        item
        for item in dataset["field_catalog"]
        if item["key"] == "performance.冲击强度"
    )
    assert pc_profile["excluded_reason"] == "unit_mismatch"
    assert impact_profile["excluded_reason"] == "unit_mismatch"
    assert any("单位缺失或单位混杂" in item for item in dataset["warnings"])
