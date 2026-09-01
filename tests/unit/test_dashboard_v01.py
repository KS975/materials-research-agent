from __future__ import annotations

from data.mysql.repositories.dashboard_repository import DashboardRepository
from schemas.user_context import UserContext


class FakeDB:
    def __init__(self):
        self.calls = []

    def query_one(self, sql, params=None):
        self.calls.append(("one", " ".join(sql.split()), list(params or [])))
        if "FROM mat_project" in sql:
            return {"project_count": 3, "count": 3}
        return {"sample_count": 7, "count": 7}

    def query_all(self, sql, params=None):
        self.calls.append(("all", " ".join(sql.split()), list(params or [])))
        return []


def context(*, company="company-a", projects=(), all_projects=True):
    return UserContext(
        user_id="user-a",
        company_id=company,
        project_ids=tuple(projects),
        permission_source="test",
        all_projects=all_projects,
    )


def test_summary_is_always_bounded_by_current_company():
    db = FakeDB()
    result = DashboardRepository(db).summary(context())

    assert result["project_count"] == 3
    assert result["sample_count"] == 7
    assert len(db.calls) == 2
    for _, sql, params in db.calls:
        assert "company = %s" in sql
        assert params == ["company-a"]


def test_project_scope_is_added_on_top_of_company_scope():
    db = FakeDB()
    DashboardRepository(db).list_projects(
        context(projects=(115, 9010), all_projects=False),
        query="研发",
    )

    assert len(db.calls) == 2
    for _, sql, params in db.calls:
        assert "p.company = %s" in sql
        assert "p.id IN (%s, %s)" in sql
        assert "company-a" in params
        assert 115 in params
        assert 9010 in params


def test_sample_browser_never_accepts_or_queries_another_company():
    db = FakeDB()
    DashboardRepository(db).list_samples(
        context(company="company-a"),
        query="trial",
        project_id=115,
    )

    assert len(db.calls) == 2
    for _, sql, params in db.calls:
        assert "s.company = %s" in sql
        assert "s.project_id = %s" in sql
        assert "company-a" in params
        assert "company-b" not in params
        assert 115 in params
