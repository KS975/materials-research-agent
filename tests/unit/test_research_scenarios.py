from __future__ import annotations

from types import SimpleNamespace

from agent.deepseek_intent_router import DeepSeekIntentRouter
from agent.research_scenarios import (
    RESEARCH_WORKFLOWS,
    resolve_research_scenario,
)
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
    return {
        "status": "ok",
        "count": 2,
        "total_matches": 2,
        "scan_complete": True,
        "samples": [
            {
                "sample": {"id": 128, "name": "EXP-128", "project_id": 115},
                "formula": [
                    {"name": "PC", "raw_key": "R3-1", "value": 65, "unit": "%"}
                ],
                "process": [],
                "performance": [
                    {"name": "冲击强度", "value": 36, "unit": "kJ/m²"}
                ],
            },
            {
                "sample": {"id": 129, "name": "EXP-129", "project_id": 115},
                "formula": [
                    {"name": "ABS", "raw_key": "R3-2", "value": 35, "unit": "%"}
                ],
                "process": [],
                "performance": [],
            },
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
                        "score": 0.9,
                        "text": "PC 使用记录。",
                        "metadata": {"company_id": "company-a", "project_id": 115},
                    }
                ],
                "warnings": [],
            }
        raise AssertionError(f"unexpected tool: {name}")


class FakeLLM:
    def complete(self, system, user):
        return "综合结论。"


def test_all_twenty_scenarios_map_to_eight_fixed_workflows() -> None:
    expected = {tuple(spec["scenario_ids"]) for spec in RESEARCH_WORKFLOWS.values()}
    assert expected == {
        (1, 2, 3, 5, 6, 7, 12, 13),
        (4, 17),
        (14,),
        (15, 16),
        (8, 10, 11),
        (9,),
        (18,),
        (19, 20),
    }

    for scenario_id in range(1, 21):
        plan = resolve_research_scenario("开放研究问题", {"scenario_id": scenario_id})
        assert plan["scenario_id"] == scenario_id
        assert plan["workflow_id"] in RESEARCH_WORKFLOWS
        assert scenario_id in plan["covered_scenario_ids"]


def test_natural_similarity_phrases_map_to_similarity_workflow() -> None:
    for message in (
        "查找与 EXP-128 相似的配方，并结合历史案例。",
        "找与 EXP-128 类似的样品。",
        "查与 EXP-128 相近的实验。",
    ):
        plan = resolve_research_scenario(message)
        assert plan["workflow_id"] == "hybrid_search_rank"
        assert plan["scenario_id"] in {1, 2}


def test_filters_take_precedence_over_numeric_identifier_in_scenario_3() -> None:
    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *_args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    strategy = skill._structured_strategy(
        {
            "identifier": "100",
            "filters": [
                {
                    "section": "performance",
                    "field": "密度差",
                    "operator": "gt",
                    "value": 100,
                }
            ],
        },
        "查找密度差大于100的历史样品。",
        3,
    )
    assert strategy["strategy"] == "structured_multi_condition_filter"
    assert strategy["tool_name"] == "list_samples_for_analysis"


def test_natural_similarity_phrase_reuses_structured_similarity_strategy() -> None:
    calls = []

    def execute_intent(intent, tool_name, args, ctx):
        calls.append((intent, tool_name, dict(args)))
        return {"status": "ok", "ranking": [], "warnings": []}

    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=execute_intent),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    strategy = skill._structured_strategy(
        {"identifier": "EXP-128", "top_n": 5},
        "查找与 EXP-128 相似的配方，并结合历史案例。",
        1,
    )
    strategy["executor"](_ctx())

    assert strategy["strategy"] == "structured_similarity"
    assert calls[0][:2] == ("similar_samples", "list_samples_for_analysis")
    assert calls[0][2]["similarity_scope"] == "formula"


def test_material_usage_effect_uses_bounded_structured_scan_and_vector_evidence() -> None:
    registry = FakeRegistry()
    skill = HybridResearchQASkill(
        registry=registry,
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *_args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )

    result = skill.answer(
        message="查 PC 的原料使用效果，并结合历史案例。",
        tool_args={"material_name": "PC", "project_id": 115},
        ctx=_ctx(),
    )

    assert result["research_workflow"]["scenario_id"] == 5
    assert result["research_workflow"]["workflow_id"] == "hybrid_search_rank"
    assert result["structured_strategy"]["strategy"] == "material_usage_effect_scan"
    assert result["mysql_result"]["count"] == 1
    assert result["mysql_result"]["matched_samples"][0]["sample"]["id"] == 128
    assert [item[0] for item in registry.calls] == [
        "list_samples_for_analysis",
        "search_vector_knowledge",
    ]
    assert registry.calls[-1][1]["query"] == "PC 使用效果"
    assert result["evidence_frame"]["source_summary"]["mysql"] > 0
    assert result["evidence_frame"]["source_summary"]["vector_api"] == 1


def test_material_substitution_remains_co_occurrence_not_proof() -> None:
    registry = FakeRegistry()
    skill = HybridResearchQASkill(
        registry=registry,
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *_args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )

    result = skill.answer(
        message="查 PC 替代 ABS 的历史效果。",
        tool_args={
            "original_material": "ABS",
            "replacement_material": "PC",
            "project_id": 115,
        },
        ctx=_ctx(),
    )

    assert result["research_workflow"]["scenario_id"] == 6
    assert result["mysql_result"]["analysis_type"] == "material_substitution_history"
    assert result["mysql_result"]["count"] == 0
    assert any("不能自动证明发生过替代" in item for item in result["warnings"])


def test_hybrid_similarity_is_not_downgraded_by_deterministic_similarity_router() -> None:
    class RouterLLM:
        def complete(self, system, user):
            return """{
                "domain":"knowledge",
                "primary_intent":"hybrid_research_qa",
                "tool_name":null,
                "tool_args":{"identifier":"EXP-128","similarity_scope":"formula"},
                "reasoning_summary":"综合数据库与历史资料"
            }"""

    decision = DeepSeekIntentRouter(RouterLLM()).route(
        "查找与 EXP-128 相似的配方，并结合历史案例。"
    )
    assert decision.intent == "hybrid_research_qa"
    assert decision.tool_name is None
    assert decision.tool_args["identifier"] == "EXP-128"
