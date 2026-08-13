from contextlib import contextmanager

from knowledge.models import KnowledgeChunk
from knowledge.repository import KnowledgeSearchHit
from schemas.user_context import UserContext
from skills.historical_knowledge import HistoricalKnowledgeRAGSkill


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return (
            "检索到类似历史记录：历史报告记录过冲击强度下降。"
            "\n\n【历史资料依据】SOURCE 1"
        )


class FakeRepo:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.hits


def _ctx(projects=(115,)):
    return UserContext(
        user_id="local-test",
        company_id="company-a",
        project_ids=projects,
        permission_source="development_header",
    )


def _hit(score=0.71):
    chunk = KnowledgeChunk(
        document_id="doc-001",
        company_id="company-a",
        project_id=115,
        filename="历史异常报告.docx",
        source_type="manual_index",
        source_id="manual:001",
        chunk_index=0,
        text="历史实验记录：某样品的冲击强度出现明显下降。",
        locator_type="paragraph",
        paragraph_start=10,
        paragraph_end=12,
    )
    return KnowledgeSearchHit(
        point_id=chunk.point_id,
        score=score,
        chunk=chunk,
    )


def test_historical_rag_uses_project_scoped_search_and_evidence():
    repo = FakeRepo([_hit()])
    llm = FakeLLM()

    @contextmanager
    def open_repo():
        yield repo

    skill = HistoricalKnowledgeRAGSkill(
        open_repo,
        llm,
        score_threshold=0.42,
        max_hits=5,
    )

    result = skill.answer(
        message="历史有没有类似的冲击强度下降问题？",
        project_id=115,
        ctx=_ctx(),
    )

    assert result["status"] == "ok"
    assert result["hit_count"] == 1
    assert result["evidence"][0]["source"] == "knowledge_index"
    assert result["evidence"][0]["project_id"] == 115
    assert result["evidence"][0]["filename"] == "历史异常报告.docx"

    call = repo.calls[0]
    assert call["company_id"] == "company-a"
    assert call["project_ids"] == [115]
    assert call["score_threshold"] == 0.42
    assert llm.calls


def test_historical_rag_no_hits_does_not_call_llm():
    repo = FakeRepo([])
    llm = FakeLLM()

    @contextmanager
    def open_repo():
        yield repo

    skill = HistoricalKnowledgeRAGSkill(open_repo, llm)
    result = skill.answer(
        message="历史有没有类似问题？",
        project_id=115,
        ctx=_ctx(),
    )

    assert result["status"] == "no_relevant_history"
    assert result["evidence"] == []
    assert llm.calls == []


def test_historical_rag_rejects_unauthorized_project_before_search():
    repo = FakeRepo([_hit()])
    llm = FakeLLM()

    @contextmanager
    def open_repo():
        yield repo

    skill = HistoricalKnowledgeRAGSkill(open_repo, llm)

    try:
        skill.answer(
            message="项目120历史有没有类似问题？",
            project_id=120,
            ctx=_ctx(projects=(115,)),
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError")

    assert repo.calls == []
    assert llm.calls == []
