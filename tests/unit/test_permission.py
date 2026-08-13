from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from connectors.permission.development_header import DevelopmentHeaderPermissionAdapter


def test_development_header_permission_is_fail_closed_without_scope():
    app = FastAPI()
    adapter = DevelopmentHeaderPermissionAdapter()

    @app.get("/")
    def route(request: Request):
        ctx = adapter.resolve(request)
        return {
            "user_id": ctx.user_id,
            "company_id": ctx.company_id,
            "project_ids": list(ctx.project_ids),
        }

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 401


def test_development_header_permission_parses_scope():
    app = FastAPI()
    adapter = DevelopmentHeaderPermissionAdapter()

    @app.get("/")
    def route(request: Request):
        ctx = adapter.resolve(request)
        return {
            "user_id": ctx.user_id,
            "company_id": ctx.company_id,
            "project_ids": list(ctx.project_ids),
        }

    client = TestClient(app)
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
