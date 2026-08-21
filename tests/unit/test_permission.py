from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from connectors.permission.development_header import DevelopmentHeaderPermissionAdapter


def _app():
    app = FastAPI()
    adapter = DevelopmentHeaderPermissionAdapter()

    @app.get("/")
    def route(request: Request):
        ctx = adapter.resolve(request)
        return {
            "user_id": ctx.user_id,
            "company_id": ctx.company_id,
            "project_ids": list(ctx.project_ids),
            "all_projects": ctx.all_projects,
            "can_access_negative_project": ctx.can_access_project(-1606),
        }

    return app


def test_development_header_permission_is_fail_closed_without_scope():
    client = TestClient(_app())
    response = client.get("/")
    assert response.status_code == 401


def test_development_header_permission_parses_scope():
    client = TestClient(_app())
    response = client.get(
        "/",
        headers={
            "X-User-Id": "u1",
            "X-Company-Id": "c1",
            "X-Project-Ids": "120,140",
        },
    )
    assert response.status_code == 200
    assert response.json()["project_ids"] == [120, 140]
    assert response.json()["all_projects"] is False
    assert response.json()["can_access_negative_project"] is False


def test_development_header_permission_all_company_projects_wildcard():
    client = TestClient(_app())
    response = client.get(
        "/",
        headers={
            "X-User-Id": "u1",
            "X-Company-Id": "c1",
            "X-Project-Ids": "*",
        },
    )
    assert response.status_code == 200
    assert response.json()["project_ids"] == []
    assert response.json()["all_projects"] is True
    assert response.json()["can_access_negative_project"] is True
