from fastapi.testclient import TestClient
from types import SimpleNamespace

from app.main import app
from api.health import get_container


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_business_db_health_is_sanitized():
    class FakeDB:
        def ping(self):
            return {"session_read_only": 1}

    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        settings=SimpleNamespace(
            business_db_host="db-host",
            business_db_user="db-user",
            business_db_password=SimpleNamespace(
                get_secret_value=lambda: "secret"
            ),
            business_db_name="materials",
        ),
        db=FakeDB(),
    )
    try:
        response = TestClient(app).get("/api/v1/health/business-db")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "configured": True,
        "connected": True,
        "database": "materials",
        "session_read_only": True,
        "error_class": None,
    }
