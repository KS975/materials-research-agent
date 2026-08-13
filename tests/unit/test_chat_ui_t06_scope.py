import pytest
from fastapi import HTTPException

from api.chat_ui import _resolve_historical_project_id
from schemas.user_context import UserContext


def _ctx(projects):
    return UserContext(
        user_id="local-test",
        company_id="company-a",
        project_ids=tuple(projects),
        permission_source="development_header",
    )


def test_single_project_scope_is_inferred():
    assert _resolve_historical_project_id({}, _ctx([115])) == 115


def test_explicit_unauthorized_project_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _resolve_historical_project_id({"project_id": 120}, _ctx([115]))
    assert exc.value.status_code == 403


def test_multiple_projects_require_explicit_project():
    with pytest.raises(HTTPException) as exc:
        _resolve_historical_project_id({}, _ctx([115, 120]))
    assert exc.value.status_code == 400
