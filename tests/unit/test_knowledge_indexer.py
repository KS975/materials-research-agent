from pathlib import Path

from knowledge.embeddings import HashEmbeddingProvider
from knowledge.indexer import (
    KnowledgeChunker,
    KnowledgeIndexer,
    KnowledgeSourceSegment,
)
from knowledge.repository import QdrantKnowledgeRepository


def test_chunker_prefers_paragraph_boundaries():
    chunker = KnowledgeChunker(max_chars=100, overlap_chars=10)

    text = (
        "第一段内容。" * 5
        + "\n"
        + "第二段内容。" * 5
        + "\n"
        + "第三段内容。" * 5
    )

    chunks = chunker.split(text)

    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "第一段内容" in chunks[0]


def test_indexer_indexes_text_and_can_search_with_scope(tmp_path: Path):
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="indexer_test",
    ) as repo:
        indexer = KnowledgeIndexer(
            repository=repo,
            chunker=KnowledgeChunker(max_chars=120, overlap_chars=10),
        )

        result = indexer.index_text(
            document_id="history-001",
            company_id="company-a",
            project_id=115,
            filename="历史异常报告.docx",
            source_type="manual_index",
            text=(
                "实验记录显示样品冲击强度出现明显下降。\n"
                "工程师随后检查了配方记录与工艺记录。\n"
                "本报告不对下降原因作确定性因果判断。"
            ),
            parser="python-docx",
            uploaded_by_user_id="local-test",
        )

        assert result.chunks_indexed >= 1

        hits = repo.search(
            query="冲击强度下降",
            company_id="company-a",
            project_ids=[115],
            limit=5,
        )

        assert hits
        assert all(hit.chunk.company_id == "company-a" for hit in hits)
        assert all(hit.chunk.project_id == 115 for hit in hits)
        assert any(
            "冲击强度" in hit.chunk.text
            for hit in hits
        )


def test_indexer_preserves_parser_evidence_locations(tmp_path: Path):
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="evidence_test",
    ) as repo:
        indexer = KnowledgeIndexer(repository=repo)

        result = indexer.index_segments(
            document_id="pdf-001",
            company_id="company-a",
            project_id=115,
            filename="历史报告.pdf",
            source_type="manual_index",
            segments=[
                KnowledgeSourceSegment(
                    text="第 3 页记录了冲击强度下降现象。",
                    locator_type="page",
                    page_number=3,
                ),
                KnowledgeSourceSegment(
                    text="第 5 页给出了后续复测结果。",
                    locator_type="page",
                    page_number=5,
                ),
            ],
            parser="pypdf",
        )

        assert result.chunks_indexed == 2

        hits = repo.search(
            query="冲击强度下降",
            company_id="company-a",
            project_ids=[115],
            limit=5,
        )

        assert hits
        matching = [
            hit for hit in hits if "第 3 页" in hit.chunk.text
        ]
        assert matching
        assert matching[0].chunk.page_number == 3
        assert matching[0].chunk.locator_type == "page"
