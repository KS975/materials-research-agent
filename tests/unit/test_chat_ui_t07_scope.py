import pytest
from fastapi import HTTPException

from api.chat_ui import _resolve_joint_args
from schemas.user_context import UserContext


def _ctx(projects=(), *, all_projects=False):
    return UserContext(
        user_id="local-test",
        company_id="company-a",
        project_ids=tuple(projects),
        permission_source="development_header",
        all_projects=all_projects,
    )


def test_t07_no_explicit_project_uses_downstream_authorized_scope():
    result = _resolve_joint_args(
        {
            "left_identifier": 3811,
            "right_identifier": 3809,
            "target_metric": "冲击强度",
            "direction_claim": "更低",
        },
        _ctx([115, 120]),
    )
    assert result["project_id"] is None


def test_t07_no_explicit_project_supports_company_all_projects():
    result = _resolve_joint_args(
        {
            "left_identifier": 3811,
            "right_identifier": 3809,
            "target_metric": "冲击强度",
        },
        _ctx(all_projects=True),
    )
    assert result["project_id"] is None


def test_t07_missing_required_argument_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _resolve_joint_args(
            {
                "left_identifier": 3811,
                "right_identifier": 3809,
            },
            _ctx([115]),
        )
    assert exc.value.status_code == 400


def test_t07_explicit_unauthorized_project_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _resolve_joint_args(
            {
                "project_id": 120,
                "left_identifier": 3811,
                "right_identifier": 3809,
                "target_metric": "冲击强度",
            },
            _ctx([115]),
        )
    assert exc.value.status_code == 403
