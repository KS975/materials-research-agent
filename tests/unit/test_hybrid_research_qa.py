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
    assert "EVIDENCE FRAME" in llm.calls[0][1]
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
    assert any("向量证据源不可用" in item for item in result["warnings"])
