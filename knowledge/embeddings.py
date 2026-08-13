from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx


class EmbeddingProvider(ABC):
    """Embedding abstraction used by the knowledge repository."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free deterministic embedding for local tests only.

    This is NOT the semantic embedding used by production RAG.
    """

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _features(text: str) -> list[str]:
        normalized = "".join(text.lower().split())
        if not normalized:
            return ["<empty>"]

        chars = list(normalized)
        features = chars.copy()
        features.extend(
            normalized[i : i + 2]
            for i in range(max(0, len(normalized) - 1))
        )
        return features

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension

        for feature in self._features(text):
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimension
            sign = 1.0 if (digest[8] & 1) == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Semantic embedding provider for OpenAI-compatible /embeddings APIs.

    Designed for Alibaba Cloud Model Studio as the current V0.1.2 provider,
    while keeping the implementation vendor-neutral.

    Recommended current Model Studio defaults:
      model: qwen3.7-text-embedding
      dimension: 1024

    text-embedding-v4 can be selected through configuration without code
    changes.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "qwen3.7-text-embedding",
        dimension: int = 1024,
        batch_size: int = 10,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.model = model
        self._dimension = int(dimension)
        self.batch_size = int(batch_size)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _request_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.post(
            "embeddings",
            json={
                "model": self.model,
                "input": list(texts),
                "dimensions": self._dimension,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("embedding response missing data list")

        rows = sorted(rows, key=lambda item: int(item.get("index", 0)))
        vectors: list[list[float]] = []

        for row in rows:
            vector = row.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("embedding response row missing embedding")
            if len(vector) != self._dimension:
                raise RuntimeError(
                    f"embedding dimension mismatch: expected "
                    f"{self._dimension}, got {len(vector)}"
                )
            vectors.append([float(value) for value in vector])

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: expected {len(texts)}, "
                f"got {len(vectors)}"
            )

        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        result: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            result.extend(
                self._request_batch(texts[start : start + self.batch_size])
            )

        return result

    def embed_query(self, text: str) -> list[float]:
        vectors = self._request_batch([text])
        return vectors[0]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAICompatibleEmbeddingProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
