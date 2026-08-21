from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from qdrant_client import QdrantClient, models

from .embeddings import EmbeddingProvider
from .models import KnowledgeChunk


@dataclass(frozen=True)
class KnowledgeSearchHit:
    point_id: str
    score: float
    chunk: KnowledgeChunk


class QdrantKnowledgeRepository:
    """Qdrant-backed repository for long-term knowledge chunks.

    Permission rule:
      every vector query MUST include company_id. Project filtering is either:
      - one or more explicitly authorized project_ids; or
      - an explicit all_projects=True grant meaning every project inside that company.

    Omitting project_ids without all_projects=True is fail-closed.

    This repository works with both:
      - Qdrant Local Mode (current V0.1.2 development)
      - Qdrant Server Mode (future deployment)

    The upper indexing/RAG layers do not need to know which mode is used.
    """

    def __init__(
        self,
        *,
        client: QdrantClient,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "materials_knowledge",
        owns_client: bool = False,
    ) -> None:
        self.client = client
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self._owns_client = owns_client

    @classmethod
    def local(
        cls,
        *,
        path: str | Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "materials_knowledge",
    ) -> "QdrantKnowledgeRepository":
        client = QdrantClient(path=str(path))
        return cls(
            client=client,
            embedding_provider=embedding_provider,
            collection_name=collection_name,
            owns_client=True,
        )

    @classmethod
    def server(
        cls,
        *,
        url: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "materials_knowledge",
        api_key: str | None = None,
    ) -> "QdrantKnowledgeRepository":
        client = QdrantClient(url=url, api_key=api_key)
        return cls(
            client=client,
            embedding_provider=embedding_provider,
            collection_name=collection_name,
            owns_client=True,
        )

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_provider.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: Sequence[KnowledgeChunk]) -> int:
        if not chunks:
            return 0

        self.ensure_collection()

        vectors = self.embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )

        if len(vectors) != len(chunks):
            raise RuntimeError(
                "embedding provider returned a different number of vectors"
            )

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.embedding_provider.dimension:
                raise RuntimeError(
                    f"embedding dimension mismatch: expected "
                    f"{self.embedding_provider.dimension}, got {len(vector)}"
                )

            points.append(
                models.PointStruct(
                    id=chunk.point_id,
                    vector=vector,
                    payload=chunk.qdrant_payload(),
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    @staticmethod
    def _permission_filter(
        *,
        company_id: str,
        project_ids: Sequence[int] | None = None,
        all_projects: bool = False,
    ) -> models.Filter:
        if not company_id:
            raise ValueError("company_id is required")

        normalized_projects = sorted(
            {int(project_id) for project_id in (project_ids or ())}
        )

        company_condition = models.FieldCondition(
            key="company_id",
            match=models.MatchValue(value=company_id),
        )

        if all_projects:
            if normalized_projects:
                raise ValueError(
                    "all_projects=True cannot be combined with project_ids"
                )
            # Explicit wildcard means every project inside this company only.
            # The company predicate is always retained.
            return models.Filter(must=[company_condition])

        if not normalized_projects:
            raise ValueError(
                "project_ids must contain at least one authorized project "
                "unless all_projects=True"
            )

        return models.Filter(
            must=[
                company_condition,
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchAny(any=normalized_projects),
                ),
            ]
        )

    def search(
        self,
        *,
        query: str,
        company_id: str,
        project_ids: Sequence[int] | None = None,
        all_projects: bool = False,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[KnowledgeSearchHit]:
        if limit <= 0:
            raise ValueError("limit must be > 0")

        if not self.client.collection_exists(self.collection_name):
            return []

        query_filter = self._permission_filter(
            company_id=company_id,
            project_ids=project_ids,
            all_projects=all_projects,
        )
        query_vector = self.embedding_provider.embed_query(query)

        if len(query_vector) != self.embedding_provider.dimension:
            raise RuntimeError(
                f"query embedding dimension mismatch: expected "
                f"{self.embedding_provider.dimension}, got {len(query_vector)}"
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
            score_threshold=score_threshold,
        )

        hits: list[KnowledgeSearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            chunk = KnowledgeChunk.model_validate(payload)
            hits.append(
                KnowledgeSearchHit(
                    point_id=str(point.id),
                    score=float(point.score),
                    chunk=chunk,
                )
            )

        return hits


    def delete_document(
        self,
        *,
        company_id: str,
        project_id: int,
        document_id: str,
    ) -> None:
        """Delete all points for one document inside an explicit permission scope."""
        if not company_id:
            raise ValueError("company_id is required")
        if not document_id:
            raise ValueError("document_id is required")
        if not self.client.collection_exists(self.collection_name):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="company_id",
                            match=models.MatchValue(value=company_id),
                        ),
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=int(project_id)),
                        ),
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "QdrantKnowledgeRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
