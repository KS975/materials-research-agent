from __future__ import annotations

from types import SimpleNamespace

from schemas.user_context import UserContext
from skills.hybrid_research_qa import HybridResearchQASkill


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-1",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "get_sample_context":
            return {
                "status": "ok",
                "sample": {
                    "id": 128,
                    "name": "EXP-128",
                    "project_id": 115,
                },
                "formula": [
                    {"name": "PC", "value": 65, "unit": "%", "resolved": True}
                ],
                "evidence": [{"source": "eln_sample", "record_id": 128}],
                "warnings": [],
            }
        if name == "search_vector_knowledge":
            assert kwargs["project_ids"] == [115]
            assert kwargs["ctx"].project_ids == (115,)
            return {
                "status": "ok",
                "provider": "external_vector_api",
                "hit_count": 1,
                "hits": [
                    {
                        "point_id": "point-1",
                        "score": 0.9,
                        "text": "历史项目使用相近配方。",
                        "metadata": {
                            "company_id": "company-a",
                            "project_id": 115,
                        },
                        "source_uri": "vector://external/point-1",
                    }
                ],
                "warnings": [],
            }
        raise AssertionError(f"unexpected tool: {name}")


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return "综合结论：数据库与历史资料均支持该配方方向。"


def test_hybrid_research_unifies_sources_before_synthesis():
    registry = FakeRegistry()
    llm = FakeLLM()
    skill = HybridResearchQASkill(
        registry=registry,
        llm=llm,
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )

    result = skill.answer(
        message="查 EXP-128 的研发上下文，并结合历史资料综合判断。",
        tool_args={"project_id": 115, "identifier": "EXP-128"},
        ctx=_ctx(),
    )

    assert result["status"] == "ok"
    assert result["analysis_type"] == "hybrid_research_qa"
    assert result["structured_strategy"]["tool_name"] == "get_sample_context"
    assert [item[0] for item in registry.calls] == [
        "get_sample_context",
        "search_vector_knowledge",
    ]
    frame = result["evidence_frame"]
    assert frame["source_summary"]["mysql"] > 0
    assert frame["source_summary"]["vector_api"] == 1
    assert frame["source_summary"]["dialog"] == 1
    assert "STRUCTURED EVIDENCE SUMMARY" in llm.calls[0][1]
    assert result["synthesis"]["status"] == "ok"
    assert result["synthesis"]["context_chars"] > 0
    assert result["evidence"][0]["record_id"]


def test_hybrid_research_degrades_vector_failure_to_structured_evidence():
    class FailingVectorRegistry(FakeRegistry):
        def execute(self, name, **kwargs):
            if name == "search_vector_knowledge":
                raise RuntimeError("vector unavailable")
            return super().execute(name, **kwargs)

    registry = FailingVectorRegistry()
    skill = HybridResearchQASkill(
        registry=registry,
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    result = skill.answer(
        message="查 EXP-128，并结合历史资料。",
        tool_args={"project_id": 115, "identifier": "EXP-128"},
        ctx=_ctx(),
    )

    assert result["status"] == "ok"
    assert result["vector_result"]["status"] == "vector_unavailable"
    assert result["evidence_frame"]["source_summary"].get("vector_api", 0) == 0
    assert any("知识资料源暂不可用" in item for item in result["warnings"])


def test_hybrid_research_retries_with_smaller_context_before_degradation():
    class TimeoutThenSuccessLLM:
        def __init__(self):
            self.prompts = []

        def complete(self, system, user):
            self.prompts.append(user)
            if len(self.prompts) == 1:
                raise TimeoutError("read timeout")
            return "综合结论：第二次有界上下文生成成功。"

    llm = TimeoutThenSuccessLLM()
    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=llm,
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    result = skill.answer(
        message="查 EXP-128，并结合历史资料。",
        tool_args={"project_id": 115, "identifier": "EXP-128"},
        ctx=_ctx(),
    )

    assert result["synthesis"]["status"] == "ok"
    assert [item["status"] for item in result["synthesis"]["attempts"]] == [
        "failed",
        "ok",
    ]
    assert result["synthesis"]["mode"] == "llm_structured_summary"


def test_hybrid_research_deterministic_fallback_keeps_structured_gap():
    class FailingLLM:
        def complete(self, system, user):
            raise TimeoutError("read timeout")

    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=FailingLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    result = skill.answer(
        message="查 EXP-128，并结合历史资料。",
        tool_args={"project_id": 115, "identifier": "EXP-128"},
        ctx=_ctx(),
    )

    assert result["status"] == "ok"
    assert result["synthesis"]["status"] == "degraded"
    assert result["synthesis"]["mode"] == "deterministic_evidence_summary"
    assert "LLM 综合未完成" in result["answer"]
    assert any("LLM 综合失败" in item for item in result["warnings"])


def test_sanitize_answer_strips_internal_references_and_adds_guardrails():
    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    raw = (
        "我将先查数据库。结论：EXP-128 密度差为 183 "
        "[mysql-fd4ad83404371c4fc836]，来源 eln_sample 表。"
    )
    cleaned = skill._sanitize_answer(raw, scenario_id=1)
    assert "mysql-fd4ad83404371c4fc836" not in cleaned
    assert "eln_sample" not in cleaned
    assert "我将先查数据库" not in cleaned
    assert "EXP-128 密度差为 183" in cleaned
    assert "单位" in cleaned


def test_sanitize_answer_adds_causal_caveat_for_analysis_scenarios():
    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    cleaned = skill._sanitize_answer("关键变量是温度。", scenario_id=8)
    assert "不能据此直接判定因果" in cleaned


def test_sanitize_answer_hides_plain_citation_ids_and_redundant_evidence_section():
    skill = HybridResearchQASkill(
        registry=FakeRegistry(),
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=lambda *args: {}),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    raw = (
        "### 结论\n相关性成立。\n\n"
        "### 证据依据\n"
        "该结论由 mysql-fd4ad83404371c4fc836 和 vector-1234567890abcdef1234 支撑。"
    )
    cleaned = skill._sanitize_answer(raw, scenario_id=10)
    assert "mysql-fd4ad83404371c4fc836" not in cleaned
    assert "vector-1234567890abcdef1234" not in cleaned
    assert "证据依据" not in cleaned
    assert "相关性成立" in cleaned


def test_vector_query_extracts_keywords_instead_of_full_sentence():
    vq = HybridResearchQASkill._vector_query
    spectrum = vq("查 EXP-128 的 DSC 图谱特征", {}, 16)
    assert "EXP-128" in spectrum
    assert "图谱" in spectrum
    assert "DSC" in spectrum
    assert "查" not in spectrum
    assert vq("关键变量有哪些？结合历史资料。", {}, 8) == "关键变量"
    assert vq("按项目分析批次差异，并结合历史资料。", {}, 11) == "批次差异"
    assert vq("哪些项目的数据可以跨项目复用？", {}, 20) == "跨项目 复用"
    # No keyword match falls back to canonical scenario keywords, not the raw question.
    fallback = vq("请帮我分析一下这个问题", {}, 10)
    assert fallback == "性能冲突 冲突分析 权衡"


def test_hybrid_research_reuses_structured_similarity_workflow():
    calls = []

    def execute_intent(intent, tool_name, args, ctx):
        calls.append((intent, tool_name, dict(args)))
        return {
            "status": "ok",
            "ranking": [],
            "warnings": [],
        }

    registry = FakeRegistry()
    skill = HybridResearchQASkill(
        registry=registry,
        llm=FakeLLM(),
        material_intelligence=SimpleNamespace(execute_intent=execute_intent),
        attachment_store=SimpleNamespace(get=lambda *_args, **_kwargs: None),
    )
    result = skill.answer(
        message="查找与 EXP-128 相似的配方和历史案例。",
        tool_args={
            "project_id": 115,
            "identifier": "EXP-128",
            "similarity_scope": "formula",
            "top_n": 5,
        },
        ctx=_ctx(),
    )

    assert result["structured_strategy"] == {
        "strategy": "structured_similarity",
        "tool_name": "list_samples_for_analysis",
    }
    assert calls[0][:2] == (
        "similar_samples",
        "list_samples_for_analysis",
    )
    assert calls[0][2]["similarity_scope"] == "formula"
    # The vector tool is still recalled through the governed registry.
    assert registry.calls[-1][0] == "search_vector_knowledge"
