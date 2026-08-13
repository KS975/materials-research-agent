from pathlib import Path

import pytest

from knowledge.embeddings import HashEmbeddingProvider
from knowledge.models import KnowledgeChunk, KnowledgeDocument
from knowledge.repository import QdrantKnowledgeRepository


def _document(
    *,
    document_id: str,
    company_id: str,
    project_id: int,
    filename: str,
    text: str,
) -> KnowledgeDocument:
    return KnowledgeDocument.from_text(
        document_id=document_id,
        company_id=company_id,
        project_id=project_id,
        filename=filename,
        source_type="manual_index",
        text=text,
        uploaded_by_user_id="local-test",
    )


def _chunk(
    document: KnowledgeDocument,
    *,
    chunk_index: int,
    text: str,
) -> KnowledgeChunk:
    return KnowledgeChunk.from_document(
        document,
        chunk_index=chunk_index,
        text=text,
        locator_type="paragraph",
        paragraph_start=chunk_index,
        paragraph_end=chunk_index,
    )


def test_hash_embedding_is_deterministic_and_normalized():
    provider = HashEmbeddingProvider(dimension=64)

    a = provider.embed_query("样品冲击强度下降")
    b = provider.embed_query("样品冲击强度下降")

    assert a == b
    assert len(a) == 64
    assert sum(value * value for value in a) == pytest.approx(1.0)


def test_qdrant_upsert_search_and_chinese_round_trip(tmp_path: Path):
    provider = HashEmbeddingProvider(dimension=64)

    doc_115 = _document(
        document_id="doc-115",
        company_id="company-a",
        project_id=115,
        filename="历史冲击强度报告.docx",
        text="样品冲击强度下降问题分析",
    )
    doc_120 = _document(
        document_id="doc-120",
        company_id="company-a",
        project_id=120,
        filename="另一个项目报告.pdf",
        text="样品冲击强度下降问题分析",
    )
    doc_other_company = _document(
        document_id="doc-company-b",
        company_id="company-b",
        project_id=115,
        filename="其他公司的报告.pdf",
        text="样品冲击强度下降问题分析",
    )

    chunks = [
        _chunk(
            doc_115,
            chunk_index=0,
            text="样品 3811 出现冲击强度下降，需要检查配方和工艺记录。",
        ),
        _chunk(
            doc_120,
            chunk_index=0,
            text="项目 120 也记录过冲击强度下降。",
        ),
        _chunk(
            doc_other_company,
            chunk_index=0,
            text="其他公司存在冲击强度下降记录。",
        ),
    ]

    path = tmp_path / "qdrant"

    with QdrantKnowledgeRepository.local(
        path=path,
        embedding_provider=provider,
        collection_name="knowledge_test",
    ) as repo:
        assert repo.upsert_chunks(chunks) == 3
        assert repo.count() == 3

        hits = repo.search(
            query="3811 冲击强度下降",
            company_id="company-a",
            project_ids=[115],
            limit=10,
        )

        assert hits
        assert all(hit.chunk.company_id == "company-a" for hit in hits)
        assert all(hit.chunk.project_id == 115 for hit in hits)
        assert hits[0].chunk.filename == "历史冲击强度报告.docx"
        assert "冲击强度" in hits[0].chunk.text
        assert "样品 3811" in hits[0].chunk.text


def test_permission_filter_is_fail_closed_and_project_scoped(tmp_path: Path):
    provider = HashEmbeddingProvider(dimension=64)
    path = tmp_path / "qdrant"

    doc_115 = _document(
        document_id="doc-115",
        company_id="company-a",
        project_id=115,
        filename="p115.docx",
        text="相同历史问题",
    )
    doc_120 = _document(
        document_id="doc-120",
        company_id="company-a",
        project_id=120,
        filename="p120.docx",
        text="相同历史问题",
    )

    with QdrantKnowledgeRepository.local(
        path=path,
        embedding_provider=provider,
        collection_name="permission_test",
    ) as repo:
        repo.upsert_chunks(
            [
                _chunk(doc_115, chunk_index=0, text="相同历史问题"),
                _chunk(doc_120, chunk_index=0, text="相同历史问题"),
            ]
        )

        hits_115 = repo.search(
            query="相同历史问题",
            company_id="company-a",
            project_ids=[115],
            limit=10,
        )
        assert {hit.chunk.project_id for hit in hits_115} == {115}

        hits_120 = repo.search(
            query="相同历史问题",
            company_id="company-a",
            project_ids=[120],
            limit=10,
        )
        assert {hit.chunk.project_id for hit in hits_120} == {120}

        with pytest.raises(ValueError):
            repo.search(
                query="相同历史问题",
                company_id="company-a",
                project_ids=[],
            )


def test_upsert_is_idempotent_for_same_chunk(tmp_path: Path):
    provider = HashEmbeddingProvider(dimension=64)
    path = tmp_path / "qdrant"

    doc = _document(
        document_id="doc-001",
        company_id="company-a",
        project_id=115,
        filename="report.docx",
        text="历史材料",
    )
    chunk = _chunk(doc, chunk_index=0, text="历史材料")

    with QdrantKnowledgeRepository.local(
        path=path,
        embedding_provider=provider,
        collection_name="idempotence_test",
    ) as repo:
        repo.upsert_chunks([chunk])
        repo.upsert_chunks([chunk])

        assert repo.count() == 1
