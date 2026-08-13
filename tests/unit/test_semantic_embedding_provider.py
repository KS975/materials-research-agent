import httpx
import pytest

from knowledge.embeddings import OpenAICompatibleEmbeddingProvider


def test_openai_compatible_embedding_provider_batches_and_preserves_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content.decode("utf-8"))
        requests.append(body)

        rows = []
        for index, text in enumerate(body["input"]):
            base = float(len(text))
            rows.append(
                {
                    "index": index,
                    "embedding": [base, base + 1.0, base + 2.0, base + 3.0],
                }
            )

        return httpx.Response(200, json={"data": rows})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport,
        base_url="https://example.test/compatible-mode/v1/",
    )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://example.test/compatible-mode/v1",
        api_key="test-key",
        model="qwen3.7-text-embedding",
        dimension=4,
        batch_size=2,
        client=client,
    )

    vectors = provider.embed_documents(["甲", "乙乙", "丙丙丙"])

    assert len(vectors) == 3
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 2.0
    assert vectors[2][0] == 3.0
    assert len(requests) == 2
    assert all(request["dimensions"] == 4 for request in requests)
    assert all(
        request["model"] == "qwen3.7-text-embedding"
        for request in requests
    )

    client.close()


def test_openai_compatible_embedding_provider_rejects_wrong_dimension():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://example.test/compatible-mode/v1/",
    )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://example.test/compatible-mode/v1",
        api_key="test-key",
        dimension=4,
        client=client,
    )

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        provider.embed_query("测试")

    client.close()
