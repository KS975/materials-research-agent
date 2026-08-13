from contextlib import contextmanager

import pytest

from knowledge.models import KnowledgeChunk
from knowledge.repository import KnowledgeSearchHit
from schemas.user_context import UserContext
from skills.joint_mysql_knowledge import JointMySQLKnowledgeAnalysisSkill


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        assert name == "compare_samples"
        ctx = kwargs["ctx"]
        assert ctx.project_ids == (115,)
        return {
            "status": "ok",
            "left_sample": {"id": 3811, "name": "trial_6", "project_id": 115},
            "right_sample": {"id": 3809, "name": "trial_4", "project_id": 115},
            "formula_diff": {
                "changed": [{"field": "ABS", "left": 18.49, "right": 33.24, "unit": "%"}],
                "same": [],
            },
            "process_diff": {
                "changed": [{"field": "保温时间", "left": 81.28, "right": 58.57, "unit": "min"}],
                "same": [],
            },
            "performance_diff": {
                "changed": [
                    {"field": "冲击强度", "left": 24, "right": 54, "unit": "kJ/m²"}
                ],
                "same": [],
            },
            "service_performance_diff": {"changed": [], "same": []},
            "test_conditions": {
                "left": {},
                "right": {},
                "status": "missing_both",
                "same": None,
                "comparable": False,
            },
            "evidence": [
                {"source": "eln_sample", "record_id": 3811},
                {"source": "eln_sample", "record_id": 3809},
            ],
            "warnings": [],
        }


class FakeRepo:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.hits


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return (
            "【数据库事实】3811 的冲击强度低于 3809。"
            "\n【历史资料】检索到类似下降记录。"
            "\n【联合判断】现象相似，但不能证明原因相同。"
            "\n【假设】配方和工艺变化值得进一步验证。"
            "\n【证据缺口】测试条件缺失。"
            "\n【结论边界】当前证据不能确认因果关系。"
        )


def _ctx(projects=(115,)):
    return UserContext(
        user_id="local-test",
        company_id="company-a",
        project_ids=projects,
        permission_source="development_header",
    )


def _history_hit():
    chunk = KnowledgeChunk(
        document_id="hist-001",
        company_id="company-a",
        project_id=115,
        filename="历史冲击强度异常报告.docx",
        source_type="manual_index",
        source_id="manual:hist-001",
        chunk_index=0,
        text="历史样品曾出现冲击强度下降，但证据不足以确定单一原因。",
        locator_type="paragraph",
        paragraph_start=1,
        paragraph_end=5,
    )
    return KnowledgeSearchHit(
        point_id=chunk.point_id,
        score=0.61,
        chunk=chunk,
    )


def test_t07_combines_mysql_and_historical_evidence_with_same_project():
    registry = FakeRegistry()
    repo = FakeRepo([_history_hit()])
    llm = FakeLLM()

    @contextmanager
    def open_repo():
        yield repo

    skill = JointMySQLKnowledgeAnalysisSkill(
        registry,
        open_repo,
        llm,
        score_threshold=0.42,
        max_hits=5,
    )

    result = skill.answer(
        message="3811 比 3809 的冲击强度低，结合历史报告分析。",
        project_id=115,
        left_identifier=3811,
        right_identifier=3809,
        target_metric="冲击强度",
        direction_claim="更低",
        ctx=_ctx(),
    )

    assert result["status"] == "ok"
    assert result["knowledge_hit_count"] == 1
    assert result["database_result"]["facts"]["target_performance"]["left"] == 24
    assert result["database_result"]["facts"]["target_performance"]["right"] == 54

    evidence_types = {item["evidence_type"] for item in result["evidence"]}
    assert "mysql" in evidence_types
    assert "knowledge_index" in evidence_types

    assert repo.calls[0]["company_id"] == "company-a"
    assert repo.calls[0]["project_ids"] == [115]
    assert repo.calls[0]["score_threshold"] == 0.42
    assert llm.calls
    assert "MYSQL FACTS" in llm.calls[0][1]
    assert "HISTORICAL KNOWLEDGE" in llm.calls[0][1]


def test_t07_no_history_still_preserves_database_facts_and_boundary():
    registry = FakeRegistry()
    repo = FakeRepo([])
    llm = FakeLLM()

    @contextmanager
    def open_repo():
        yield repo

    skill = JointMySQLKnowledgeAnalysisSkill(registry, open_repo, llm)

    result = skill.answer(
        message="结合历史分析 3811 和 3809。",
        project_id=115,
        left_identifier=3811,
        right_identifier=3809,
        target_metric="冲击强度",
        direction_claim="更低",
        ctx=_ctx(),
    )

    assert result["status"] == "ok"
    assert result["knowledge_hit_count"] == 0
    assert any("当前已索引历史资料" in warning for warning in result["warnings"])
    assert all(
        item["evidence_type"] == "mysql"
        for item in result["evidence"]
    )
    assert "NO_RELEVANT_HISTORY_HITS" in llm.calls[0][1]


def test_t07_rejects_unauthorized_project_before_mysql_or_qdrant():
    registry = FakeRegistry()
    repo = FakeRepo([])
    llm = FakeLLM()

    @contextmanager
    def open_repo():
        yield repo

    skill = JointMySQLKnowledgeAnalysisSkill(registry, open_repo, llm)

    with pytest.raises(PermissionError):
        skill.answer(
            message="项目120联合分析。",
            project_id=120,
            left_identifier=3811,
            right_identifier=3809,
            target_metric="冲击强度",
            direction_claim="更低",
            ctx=_ctx(projects=(115,)),
        )

    assert registry.calls == []
    assert repo.calls == []
    assert llm.calls == []
