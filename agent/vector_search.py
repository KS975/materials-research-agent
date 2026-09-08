from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

import httpx

from schemas.user_context import UserContext


class VectorSearchGateway(Protocol):
    def name(self) -> str: ...
    def search(self, request: Mapping[str, Any]) -> list[dict[str, Any]]: ...


class QdrantVectorSearchGateway:
    """Compatibility gateway for the already validated local/server Qdrant path."""

    def __init__(
        self,
        repository_opener: Callable[[], AbstractContextManager],
        *,
        provider_name: str = "legacy_qdrant",
    ) -> None:
        self.repository_opener = repository_opener
        self.provider_name = provider_name

    def name(self) -> str:
        return self.provider_name

    def search(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        with self.repository_opener() as repository:
            hits = repository.search(
                query=str(request["query"]),
                company_id=str(request["company_id"]),
                project_ids=list(request.get("project_ids") or []),
                all_projects=bool(request.get("all_projects", False)),
                limit=int(request.get("limit") or 5),
                score_threshold=request.get("score_threshold"),
            )
        output: list[dict[str, Any]] = []
        for hit in hits:
            chunk = hit.chunk
            metadata = {
                "company_id": chunk.company_id,
                "project_id": chunk.project_id,
                "document_id": chunk.document_id,
                "source_id": chunk.source_id,
                "filename": chunk.filename,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page_number,
                "paragraph_start": chunk.paragraph_start,
                "paragraph_end": chunk.paragraph_end,
                "locator_type": chunk.locator_type,
            }
            output.append(
                {
                    "point_id": str(hit.point_id),
                    "score": float(hit.score),
                    "text": chunk.text,
                    "metadata": metadata,
                    "source_uri": (
                        f"vector://{self.provider_name}/{chunk.document_id}"
                        f"/{chunk.chunk_index}"
                    ),
                }
            )
        return output


class ExternalVectorAPIGateway:
    """Server-side adapter for an HTTP vector-search API.

    This is the internal stable contract. Deployment can map these fields to
    the external API path/header names without exposing credentials to the
    browser or to the LLM.
    """

    _ID_KEYS = ("point_id", "chunk_id", "id")
    _SCORE_KEYS = ("score", "similarity")
    _TEXT_KEYS = ("text", "content", "chunk", "passage")

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    def name(self) -> str:
        return "external_vector_api"

    def search(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = dict(request)
        payload["trace_id"] = str(uuid4())

        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        owns_client = self._client is None
        try:
            response = client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        finally:
            if owns_client:
                client.close()

        rows = self._rows(body)
        return [self._normalize_row(row) for row in rows if isinstance(row, Mapping)]

    @staticmethod
    def _rows(body: Any) -> list[Any]:
        if isinstance(body, list):
            return body
        if not isinstance(body, Mapping):
            return []
        for key in ("hits", "results", "items", "data"):
            value = body.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, Mapping):
                nested = ExternalVectorAPIGateway._rows(value)
                if nested:
                    return nested
        return []

    def _normalize_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        metadata = dict(row.get("metadata") or row.get("payload") or {})
        point_id = next(
            (str(row[key]) for key in self._ID_KEYS if row.get(key) is not None),
            str(uuid4()),
        )
        score = next(
            (row[key] for key in self._SCORE_KEYS if row.get(key) is not None),
            1.0,
        )
        text = next(
            (str(row[key]) for key in self._TEXT_KEYS if row.get(key) is not None),
            "",
        )
        for key in ("project_id", "company_id", "document_id", "filename"):
            if row.get(key) is not None:
                metadata.setdefault(key, row[key])
        return {
            "point_id": point_id,
            "score": self._score(score),
            "text": text,
            "metadata": metadata,
            "source_uri": str(row.get("source_uri") or f"vector://external/{point_id}"),
        }

    @staticmethod
    def _score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 1.0
        if 0 <= score <= 1:
            return score
        if 0 <= score <= 100:
            return score / 100
        return 1.0


@dataclass(frozen=True)
class VectorSearchService:
    gateway: VectorSearchGateway
    default_limit: int = 5
    default_score_threshold: float = 0.42

    def search(
        self,
        *,
        query: str,
        ctx: UserContext,
        project_ids: list[int] | None = None,
        all_projects: bool = False,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("向量检索 query 不能为空")
        normalized_limit = max(1, min(int(limit), 50))
        threshold = (
            self.default_score_threshold
            if score_threshold is None
            else max(0.0, min(float(score_threshold), 1.0))
        )

        projects = [int(item) for item in (project_ids or [])]
        for project_id in projects:
            if not ctx.can_access_project(project_id):
                raise PermissionError(f"当前用户无权检索 Project {project_id}")
        if all_projects:
            if not ctx.all_projects:
                raise PermissionError("当前用户没有全项目向量检索权限")
            projects = []
        elif not projects:
            if ctx.all_projects:
                all_projects = True
            elif ctx.project_ids:
                projects = sorted({int(item) for item in ctx.project_ids})
            else:
                raise PermissionError("当前用户没有可用于向量检索的项目权限")

        request = {
            "query": normalized_query,
            "company_id": ctx.company_id,
            "project_ids": projects,
            "all_projects": all_projects,
            "limit": normalized_limit,
            "score_threshold": threshold,
            "acting_user_id": ctx.user_id,
        }
        hits = self.gateway.search(request)
        authorized_hits: list[dict[str, Any]] = []
        rejected = 0
        low_score = 0
        for hit in hits:
            metadata = dict(hit.get("metadata") or {})
            company_id = metadata.get("company_id")
            project_id = metadata.get("project_id")
            try:
                score = float(hit.get("score"))
                valid_score = 0.0 <= score <= 1.0
            except (TypeError, ValueError):
                score = -1.0
                valid_score = False
            if not valid_score or score < threshold:
                low_score += 1
                continue
            if company_id is None or str(company_id) != str(ctx.company_id):
                rejected += 1
                continue
            if not all_projects:
                try:
                    project_allowed = int(project_id) in projects
                except (TypeError, ValueError):
                    project_allowed = False
                if not project_allowed:
                    rejected += 1
                    continue
            authorized_hits.append(hit)

        warnings = []
        if rejected:
            warnings.append(
                f"外部向量 API 返回 {rejected} 条越权结果，已全部丢弃。"
            )
        if low_score:
            warnings.append(f"向量检索有 {low_score} 条结果低于阈值，已过滤。")
        return {
            "status": "ok",
            "provider": self.gateway.name(),
            "query": normalized_query,
            "project_ids": projects,
            "all_projects": all_projects,
            "score_threshold": threshold,
            "hit_count": len(authorized_hits),
            "hits": authorized_hits,
            "warnings": warnings,
        }


class VectorSearchTool:
    def __init__(self, service: VectorSearchService) -> None:
        self.service = service

    def __call__(
        self,
        *,
        query: str,
        ctx: UserContext,
        project_ids: list[int] | None = None,
        all_projects: bool = False,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.service.search(
            query=query,
            ctx=ctx,
            project_ids=project_ids,
            all_projects=all_projects,
            limit=limit,
            score_threshold=score_threshold,
        )


def looks_like_vector_provider(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_.:-]+", str(value or "").casefold()))
