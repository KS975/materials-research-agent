from __future__ import annotations

import json

import httpx
import pytest

from agent.vector_search import ExternalVectorAPIGateway, VectorSearchService
from agent.vector_tool_registration import register_vector_tools
from agent.vector_search import VectorSearchTool
from agent.tool_registry import ToolRegistry, ToolValidationError
from schemas.user_context import UserContext


class FakeGateway:
    def __init__(self, rows):
        self.rows = rows
        self.requests = []

    def name(self) -> str:
        return "external_vector_api"

    def search(self, request):
        self.requests.append(dict(request))
        return self.rows


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-1",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


def test_vector_service_drops_unauthorized_hits_and_keeps_audit_metadata():
    gateway = FakeGateway(
        [
            {
                "point_id": "ok",
                "score": 0.9,
                "text": "authorized",
                "metadata": {"company_id": "company-a", "project_id": 115},
            },
            {
                "point_id": "wrong-project",
                "score": 0.9,
                "text": "unauthorized",
                "metadata": {"company_id": "company-a", "project_id": 999},
            },
        ]
    )
    service = VectorSearchService(gateway=gateway)
    result = service.search(query="相近配方", ctx=_ctx(), limit=5)

    assert result["status"] == "ok"
    assert result["hit_count"] == 1
    assert result["hits"][0]["point_id"] == "ok"
    assert "越权" in result["warnings"][0]
    assert gateway.requests[0]["company_id"] == "company-a"
    assert gateway.requests[0]["project_ids"] == [115]
    assert gateway.requests[0]["acting_user_id"] == "user-1"


def test_vector_tool_is_schema_validated_and_permission_scoped():
    registry = ToolRegistry()
    register_vector_tools(
        registry,
        VectorSearchTool(
            VectorSearchService(gateway=FakeGateway([]), default_score_threshold=0.4)
        ),
    )
    result = registry.execute(
        "search_vector_knowledge",
        query="history",
        limit=3,
        ctx=_ctx(),
    )
    assert result["status"] == "ok"

    with pytest.raises(ToolValidationError):
        registry.execute(
            "search_vector_knowledge",
            query="history",
            unexpected=True,
            ctx=_ctx(),
        )


def test_external_gateway_sends_service_credential_and_normalizes_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "chunk-1",
                        "similarity": 88,
                        "content": "historical fact",
                        "metadata": {
                            "company_id": "company-a",
                            "project_id": 115,
                        },
                    }
                ]
            },
        )

    gateway = ExternalVectorAPIGateway(
        endpoint="https://vector.example/api/search",
        api_key="secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    hits = gateway.search({"query": "history", "company_id": "company-a"})

    assert captured["authorization"] == "Bearer secret-token"
    assert captured["body"]["company_id"] == "company-a"
    assert hits[0]["point_id"] == "chunk-1"
    assert hits[0]["score"] == 0.88
    assert hits[0]["text"] == "historical fact"
