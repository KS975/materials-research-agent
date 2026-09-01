import base64
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.chat import router as chat_router
from app.config import Settings, get_settings
from connectors.permission.development_header import DevelopmentHeaderPermissionAdapter
from connectors.permission.platform import PlatformPermissionAdapter


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


def _jwt(payload):
    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'RS256', 'typ': 'JWT'})}.{encode(payload)}.test-signature"


def _platform_app(*, trusted=True):
    app = FastAPI()
    settings = Settings(
        _env_file=None,
        permission_mode="platform",
        platform_trust_forwarded_headers=trusted,
        llm_enabled=False,
    )
    adapter = PlatformPermissionAdapter(settings)

    @app.get("/")
    def route(request: Request):
        ctx = adapter.resolve(request)
        return {
            "user_id": ctx.user_id,
            "company_id": ctx.company_id,
            "organization_id": ctx.organization_id,
            "organization_level": ctx.organization_level,
            "all_projects": ctx.all_projects,
        }

    return app


def _platform_headers(payload=None):
    return {
        "authorization": f"Bearer {_jwt(payload or {'userId': '2090369875129171970'})}",
        "company-id": "6a4b19f62d0e000027001eb8",
        "organization-id": "6a6b2090f1a6586e94f958bb",
        "organization-level": "1",
    }


def test_platform_permission_resolves_four_forwarded_headers_and_user_claim():
    response = TestClient(_platform_app()).get("/", headers=_platform_headers())

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "2090369875129171970",
        "company_id": "6a4b19f62d0e000027001eb8",
        "organization_id": "6a6b2090f1a6586e94f958bb",
        "organization_level": "1",
        "all_projects": True,
    }


def test_platform_permission_accepts_nested_user_claim():
    response = TestClient(_platform_app()).get(
        "/",
        headers=_platform_headers({"data": {"user": {"userId": "nested-user"}}}),
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == "nested-user"


def test_platform_permission_fails_closed_when_header_missing():
    headers = _platform_headers()
    headers.pop("organization-id")
    response = TestClient(_platform_app()).get("/", headers=headers)
    assert response.status_code == 401
    assert "organization-id" in response.json()["detail"]


def test_platform_permission_requires_explicit_gateway_trust():
    response = TestClient(_platform_app(trusted=False)).get(
        "/", headers=_platform_headers()
    )
    assert response.status_code == 503
    assert "PLATFORM_TRUST_FORWARDED_HEADERS" in response.json()["detail"]


def test_platform_permission_rejects_token_without_stable_user_id():
    response = TestClient(_platform_app()).get(
        "/", headers=_platform_headers({"scope": "materials.read"})
    )
    assert response.status_code == 401
    assert "稳定用户标识" in response.json()["detail"]


def test_session_context_returns_safe_identity_without_token():
    settings = Settings(
        _env_file=None,
        permission_mode="platform",
        platform_trust_forwarded_headers=True,
        llm_enabled=False,
    )
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_settings] = lambda: settings

    response = TestClient(app).get(
        "/api/v1/session-context", headers=_platform_headers()
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == "2090369875129171970"
    assert response.json()["project_mode"] == "company_all_projects"
    assert "authorization" not in response.text.lower()
    assert "test-signature" not in response.text
