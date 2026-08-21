import pytest
from fastapi import HTTPException

from api.chat_ui import _resolve_historical_project_id
from schemas.user_context import UserContext


def _ctx(projects=(), *, all_projects=False):
    return UserContext(
        user_id="local-test",
        company_id="company-a",
        project_ids=tuple(projects),
        permission_source="development_header",
        all_projects=all_projects,
    )


def test_no_explicit_project_keeps_scope_open_for_downstream_authorized_search():
    assert _resolve_historical_project_id({}, _ctx([115])) is None
    assert _resolve_historical_project_id({}, _ctx([115, 120])) is None
    assert _resolve_historical_project_id({}, _ctx(all_projects=True)) is None


def test_explicit_authorized_project_is_preserved():
    assert _resolve_historical_project_id({"project_id": 115}, _ctx([115])) == 115


def test_explicit_unauthorized_project_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _resolve_historical_project_id({"project_id": 120}, _ctx([115]))
    assert exc.value.status_code == 403
