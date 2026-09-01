from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from agent.tool_registry import ToolRegistry
from schemas.user_context import UserContext
from skills.sample_historical_similarity import SampleHistoricalSimilaritySkill


@dataclass
class Chunk:
    filename: str = "历史冲击强度异常报告.docx"
    project_id: int = 115
    source_id: str = "manual-upload:test"
    text: str = "历史样品A出现冲击强度明显下降，建议核查配方、工艺和测试条件。"
    document_id: str = "doc1"
    chunk_index: int = 0
    page_number: int | None = None
    paragraph_start: int | None = 1
    paragraph_end: int | None = 3
    locator_type: str = "paragraph"


@dataclass
class Hit:
    score: float = 0.88
    chunk: Chunk = None

    def __post_init__(self):
        if self.chunk is None:
            self.chunk = Chunk()


class FakeRepo:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [Hit()]


class FakeLLM:
    def complete(self, system: str, user: str) -> str:
        assert "DATABASE FACTS" in user
        assert "HISTORICAL KNOWLEDGE SOURCES" in user
        return "检索到类似历史案例，但不能据此认定根因相同。"


def test_single_sample_history_skill_keeps_sample_scope_and_history_scope_separate():
    registry = ToolRegistry()
    registry.register(
        "get_sample_context",
        "test",
        lambda identifier, ctx: {
            "status": "ok",
            "sample": {"id": 3811, "name": "trial_6", "project_id": 999},
            "formula": [],
            "process": [],
            "performance": [
                {"name": "冲击强度", "value": 24, "unit": "kJ/m²"}
            ],
            "conditions": {},
            "evidence": [{"source": "eln_sample", "record_id": 3811}],
            "warnings": [],
        },
    )
    repo = FakeRepo()

    @contextmanager
    def opener():
        yield repo

    skill = SampleHistoricalSimilaritySkill(registry, opener, FakeLLM())
    ctx = UserContext(
        user_id="u",
        company_id="company-a",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )

    result = skill.answer(
        message="Project 115呢？",
        history_query="历史上有没有和3811类似的冲击强度异常？",
        identifier=3811,
        target_metric="冲击强度",
        project_id=115,
        ctx=ctx,
    )

    assert result["status"] == "ok"
    assert result["analysis_scope"]["project_ids"] == [115]
    # Sample's own project=999 does not prevent using Project115 as history scope.
    assert result["database_result"]["sample"]["project_id"] == 999
    assert repo.calls[0]["company_id"] == "company-a"
    assert repo.calls[0]["project_ids"] == [115]
    assert repo.calls[0]["all_projects"] is False
    assert "3811" in repo.calls[0]["query"]
    assert "冲击强度" in repo.calls[0]["query"]
    assert any(x.get("evidence_type") == "mysql" for x in result["evidence"])
    assert any(x.get("evidence_type") == "historical_knowledge" for x in result["evidence"])
