from app.main import app


def test_knowledge_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert "/api/v1/knowledge/index-upload" in paths
    assert "/api/v1/knowledge/search" in paths
