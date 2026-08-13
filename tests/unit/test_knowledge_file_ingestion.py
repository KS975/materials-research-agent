from io import BytesIO
from pathlib import Path

import pytest
from docx import Document as DocxDocument

from file_processing import ChatFileParser
from knowledge.embeddings import HashEmbeddingProvider
from knowledge.file_ingestion import KnowledgeFileIngestionService
from knowledge.repository import QdrantKnowledgeRepository
from schemas.user_context import UserContext


def _docx_bytes(*paragraphs: str) -> bytes:
    doc = DocxDocument()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _ctx(project_ids=(115,)) -> UserContext:
    return UserContext(
        user_id="local-test",
        company_id="company-a",
        project_ids=tuple(project_ids),
        permission_source="test",
    )


def test_real_docx_parser_can_feed_long_term_qdrant_index(tmp_path: Path):
    content = _docx_bytes(
        "历史实验报告",
        "样品出现冲击强度下降，需要核查配方、工艺和测试条件。",
        "当前证据不足以确定单一因果原因。",
    )
    service = KnowledgeFileIngestionService(ChatFileParser())
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="t05_ingestion",
    ) as repo:
        result = service.index_bytes(
            filename="历史冲击强度报告.docx",
            content=content,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            project_id=115,
            ctx=_ctx(),
            repository=repo,
        )

        assert result.chunks_indexed >= 1
        assert result.project_id == 115
        assert result.company_id == "company-a"
        assert result.source_id.startswith("manual-upload:")
        assert result.parser == "python-docx"

        hits = repo.search(
            query="冲击强度下降",
            company_id="company-a",
            project_ids=[115],
            limit=5,
        )
        assert hits
        assert hits[0].chunk.filename == "历史冲击强度报告.docx"
        assert hits[0].chunk.source_id == result.source_id
        assert hits[0].chunk.project_id == 115
        assert "冲击强度" in hits[0].chunk.text


def test_same_file_reindex_is_idempotent(tmp_path: Path):
    content = _docx_bytes("样品冲击强度下降问题分析。")
    service = KnowledgeFileIngestionService(ChatFileParser())
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="t05_idempotence",
    ) as repo:
        first = service.index_bytes(
            filename="same.docx",
            content=content,
            media_type="",
            project_id=115,
            ctx=_ctx(),
            repository=repo,
        )
        count_after_first = repo.count()

        second = service.index_bytes(
            filename="same.docx",
            content=content,
            media_type="",
            project_id=115,
            ctx=_ctx(),
            repository=repo,
        )

        assert first.document_id == second.document_id
        assert first.source_id == second.source_id
        assert repo.count() == count_after_first


def test_long_term_index_permission_is_fail_closed(tmp_path: Path):
    content = _docx_bytes("项目 115 的历史报告")
    service = KnowledgeFileIngestionService(ChatFileParser())
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="t05_permission",
    ) as repo:
        with pytest.raises(PermissionError):
            service.index_bytes(
                filename="forbidden.docx",
                content=content,
                media_type="",
                project_id=120,
                ctx=_ctx(project_ids=(115,)),
                repository=repo,
            )

        assert repo.count() == 0



def test_secrets_are_redacted_before_qdrant_payload(tmp_path: Path):
    content = _docx_bytes(
        "系统配置",
        "BUSINESS_DB_USER=root",
        "BUSINESS_DB_PASSWORD=super-secret-password",
        "EMBEDDING_API_KEY=sk-test-should-not-be-stored",
        "PERMISSION_MODE=development_header",
    )
    service = KnowledgeFileIngestionService(ChatFileParser())
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="t05_redaction",
    ) as repo:
        result = service.index_bytes(
            filename="config.docx",
            content=content,
            media_type="",
            project_id=115,
            ctx=_ctx(),
            repository=repo,
        )

        assert result.secrets_redacted >= 2

        hits = repo.search(
            query="数据库密码 API Key 配置",
            company_id="company-a",
            project_ids=[115],
            limit=10,
        )
        assert hits

        joined = "\n".join(hit.chunk.text for hit in hits)
        assert "super-secret-password" not in joined
        assert "sk-test-should-not-be-stored" not in joined
        assert "BUSINESS_DB_PASSWORD=[REDACTED]" in joined
        assert "EMBEDDING_API_KEY=[REDACTED]" in joined


def test_delete_document_removes_only_target_scope(tmp_path: Path):
    content = _docx_bytes("历史报告")
    service = KnowledgeFileIngestionService(ChatFileParser())
    provider = HashEmbeddingProvider(dimension=64)

    with QdrantKnowledgeRepository.local(
        path=tmp_path / "qdrant",
        embedding_provider=provider,
        collection_name="t05_delete",
    ) as repo:
        first = service.index_bytes(
            filename="a.docx",
            content=content,
            media_type="",
            project_id=115,
            ctx=_ctx(),
            repository=repo,
            source_id="manual:a",
        )
        second = service.index_bytes(
            filename="b.docx",
            content=content,
            media_type="",
            project_id=115,
            ctx=_ctx(),
            repository=repo,
            source_id="manual:b",
        )
        assert repo.count() == 2

        repo.delete_document(
            company_id="company-a",
            project_id=115,
            document_id=first.document_id,
        )

        assert repo.count() == 1
        hits = repo.search(
            query="历史报告",
            company_id="company-a",
            project_ids=[115],
            limit=10,
        )
        assert hits
        assert all(hit.chunk.document_id == second.document_id for hit in hits)
