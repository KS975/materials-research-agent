from data.mysql.repositories.common import project_scope_clause
from schemas.user_context import UserContext


def test_project_scope_clause_is_fail_closed_for_empty_ordinary_scope():
    sql, params = project_scope_clause(())
    assert sql == "1 = 0"
    assert params == []


def test_project_scope_clause_all_projects_requires_explicit_flag():
    sql, params = project_scope_clause((), allow_all=True)
    assert sql == "1 = 1"
    assert params == []


def test_all_projects_still_has_explicit_company_identity():
    ctx = UserContext(
        user_id="u1",
        company_id="company-1",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )
    assert ctx.company_id == "company-1"
    assert ctx.can_access_project(115)
    assert ctx.can_access_project(-1606)
