from __future__ import annotations

import os
from pathlib import Path

from knowledge.embeddings import OpenAICompatibleEmbeddingProvider
from knowledge.indexer import KnowledgeIndexer
from knowledge.repository import QdrantKnowledgeRepository
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
    api_key = (
        os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    model = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    if not base_url or not api_key:
        print("ERROR: configure EMBEDDING_BASE_URL and EMBEDDING_API_KEY/DASHSCOPE_API_KEY")
        return 2

    provider = OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dimension=dimension,
    )

    try:
        with QdrantKnowledgeRepository.local(
            path=Path(".runtime/qdrant_b04_smoke"),
            embedding_provider=provider,
            collection_name=f"b04_semantic_{dimension}",
        ) as repo:
            indexer = KnowledgeIndexer(repository=repo)

            indexer.index_text(
                document_id="b04-history-impact",
                company_id="company-a",
                project_id=115,
                filename="历史冲击强度异常报告.docx",
                source_type="manual_index",
                text=(
                    "历史实验记录：某样品的冲击强度出现明显下降。"
                    "报告要求进一步核查配方变化、工艺变化和测试条件，"
                    "但现有证据不足以确定单一因果原因。"
                ),
            )
            indexer.index_text(
                document_id="b04-permission",
                company_id="company-a",
                project_id=120,
                filename="项目120报告.docx",
                source_type="manual_index",
                text="项目120也存在冲击强度异常记录。",
            )
            indexer.index_text(
                document_id="b04-unrelated",
                company_id="company-a",
                project_id=115,
                filename="数据库权限说明.docx",
                source_type="manual_index",
                text="系统使用 development_header 模拟登录上下文。",
            )

            hits = repo.search(
                query="历史有没有类似的冲击强度下降问题？",
                company_id="company-a",
                project_ids=[115],
                limit=3,
            )

            print("hits:", len(hits))
            for index, hit in enumerate(hits, 1):
                print(
                    f"{index}. score={hit.score:.6f} "
                    f"project={hit.chunk.project_id} "
                    f"file={hit.chunk.filename}"
                )
                print("   ", hit.chunk.text)

            if not hits:
                print("FAIL: no result")
                return 1

            if hits[0].chunk.filename != "历史冲击强度异常报告.docx":
                print("FAIL: semantic top hit was not the expected historical report")
                return 1

            if any(hit.chunk.project_id != 115 for hit in hits):
                print("FAIL: project permission filter leaked another project")
                return 1

            print("SEMANTIC QDRANT KNOWLEDGE PASS")
            return 0
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
