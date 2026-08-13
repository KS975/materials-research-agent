import math
import os
import sys

from dotenv import load_dotenv

from knowledge.embeddings import OpenAICompatibleEmbeddingProvider

load_dotenv()


def cosine(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def main() -> int:
    base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
    api_key = (
        os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    model = os.getenv(
        "EMBEDDING_MODEL",
        "qwen3.7-text-embedding",
    ).strip()
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    if not base_url:
        print("ERROR: EMBEDDING_BASE_URL is not configured")
        return 2
    if not api_key:
        print("ERROR: EMBEDDING_API_KEY or DASHSCOPE_API_KEY is not configured")
        return 2

    with OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dimension=dimension,
    ) as provider:
        query = provider.embed_query("历史上有没有冲击强度下降的类似问题？")
        related = provider.embed_query("实验报告记录样品冲击强度明显下降。")
        unrelated = provider.embed_query("数据库权限采用 development_header 模式。")

    print("model:", model)
    print("dimension:", len(query))
    print("related_similarity:", round(cosine(query, related), 6))
    print("unrelated_similarity:", round(cosine(query, unrelated), 6))
    print("SEMANTIC EMBEDDING API PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
