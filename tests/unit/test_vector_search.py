from __future__ import annotations

import httpx
import pytest

from agent.vector_search import (
    ExternalVectorAPIError,
    ExternalVectorAPIGateway,
    VectorSearchService,
)
from agent.vector_tool_registration import register_vector_tools
from agent.vector_search import VectorSearchTool
from agent.tool_registry import ToolRegistry, ToolValidationError
from runtime.request_credentials import request_authorization_scope
from schemas.user_context import UserContext


class FakeGateway:
    def __init__(self, rows, *, provider="legacy_qdrant"):
        self.rows = rows
        self.provider = provider
        self.requests = []

    def name(self) -> str:
        return self.provider

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
        ],
        provider="legacy_qdrant",
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


def test_external_gateway_uses_platform_token_and_get_contract():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        captured["company_id"] = request.headers.get("Company-Id")
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "messageZh": "成功",
                "data": [
                    {
                        "id": "chunk-1",
                        "score": 0.88,
                        "payload": {
                            "companyId": "company-a",
                            "organizationId": "org-a",
                            "userId": "user-1",
                            "fileName": "history.docx",
                            "content": "historical fact",
                        },
                    }
                ]
            },
        )

    gateway = ExternalVectorAPIGateway(
        endpoint="https://vector.example/api/search",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with request_authorization_scope("Bearer platform-token"):
        hits = gateway.search(
            {
                "query": "history",
                "company_id": "company-a",
                "header_company_id": "platform-company-a",
                "organization_id": "org-a",
                "limit": 3,
            }
        )

    assert captured["authorization"] == "Bearer platform-token"
    assert captured["company_id"] == "platform-company-a"
    assert captured["params"]["queryText"] == "history"
    assert captured["params"]["companyId"] == "company-a"
    assert captured["params"]["organizationId"] == "org-a"
    assert captured["params"]["pageRanges"] == "1-3"
    assert "limit" not in captured["params"]
    assert "content" not in captured["params"]
    assert hits[0]["point_id"] == "chunk-1"
    assert hits[0]["score"] == 0.88
    assert hits[0]["text"] == "historical fact"
    assert hits[0]["metadata"]["company_id"] == "company-a"
    assert hits[0]["metadata"]["organization_id"] == "org-a"
    assert hits[0]["metadata"]["filename"] == "history.docx"


def test_external_gateway_rejects_business_error_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": 121, "messageZh": "未登录", "data": None},
        )

    gateway = ExternalVectorAPIGateway(
        endpoint="https://vector.example/api/search",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ExternalVectorAPIError) as exc_info:
        gateway.search({"query": "history", "company_id": "company-a"})

    assert exc_info.value.code == 121
    assert "未登录" in str(exc_info.value)


def test_external_vector_all_query_keeps_zero_score_hits():
    gateway = FakeGateway(
        [
            {
                "point_id": "file-1",
                "score": 0.0,
                "text": "all documents",
                "metadata": {"company_id": "company-a"},
            }
        ],
        provider="external_vector_api",
    )
    service = VectorSearchService(gateway=gateway)
    result = service.search(query="", ctx=_ctx(), limit=5)

    assert result["status"] == "ok"
    assert result["hit_count"] == 1
    assert result["hits"][0]["score"] == 0.0


def test_external_gateway_always_sends_required_query_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"code": 0, "data": []})

    gateway = ExternalVectorAPIGateway(
        endpoint="https://vector.example/api/search",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    gateway.search({"query": "", "company_id": "company-a", "limit": 5})

    assert captured["params"]["queryText"] == ""
    assert captured["params"]["companyId"] == "company-a"
    assert captured["params"]["pageRanges"] == "1-5"


def test_external_vector_results_are_truncated_to_local_limit():
    rows = [
        {
            "point_id": f"file-{index}",
            "score": 0.9,
            "text": f"document {index}",
            "metadata": {"company_id": "company-a"},
        }
        for index in range(1, 5)
    ]
    service = VectorSearchService(
        gateway=FakeGateway(rows, provider="external_vector_api")
    )
    result = service.search(query="history", ctx=_ctx(), limit=2)

    assert result["hit_count"] == 2
    assert [item["point_id"] for item in result["hits"]] == ["file-1", "file-2"]
    assert any("本地" in warning and "截断" in warning for warning in result["warnings"])
