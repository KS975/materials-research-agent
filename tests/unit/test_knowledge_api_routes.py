from api.knowledge import KnowledgeSearchRequest
from app.main import app


def test_knowledge_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert "/api/v1/knowledge/index-upload" in paths
    assert "/api/v1/knowledge/search" in paths


def test_knowledge_search_project_id_is_optional():
    body = KnowledgeSearchRequest(query="历史上有没有类似问题？")
    assert body.project_id is None
