from knowledge.models import KnowledgeChunk, KnowledgeDocument, sha256_text


def _document() -> KnowledgeDocument:
    return KnowledgeDocument.from_text(
        document_id="doc-001",
        company_id="company-a",
        project_id=115,
        filename="历史冲击强度报告.docx",
        source_type="manual_index",
        text="第一段。\n样品冲击强度下降问题分析。",
        parser="python-docx",
        uploaded_by_user_id="local-test",
    )


def test_document_hash_and_permission_payload():
    doc = _document()

    assert doc.content_hash == sha256_text("第一段。\n样品冲击强度下降问题分析。")

    payload = doc.qdrant_payload_base()
    assert payload["company_id"] == "company-a"
    assert payload["project_id"] == 115
    assert payload["document_id"] == "doc-001"
    assert payload["filename"] == "历史冲击强度报告.docx"


def test_chunk_payload_preserves_chinese_and_scope():
    doc = _document()
    text = "样品冲击强度下降问题分析"

    chunk = KnowledgeChunk.from_document(
        doc,
        chunk_index=0,
        text=text,
        locator_type="paragraph",
        paragraph_start=3,
        paragraph_end=5,
    )

    payload = chunk.qdrant_payload()

    assert payload["text"] == text
    assert payload["company_id"] == "company-a"
    assert payload["project_id"] == 115
    assert payload["document_id"] == "doc-001"
    assert payload["paragraph_start"] == 3
    assert payload["paragraph_end"] == 5
    assert chunk.content_hash == sha256_text(text)
    assert chunk.point_id


def test_point_id_is_deterministic_and_content_sensitive():
    doc = _document()

    a = KnowledgeChunk.from_document(doc, chunk_index=0, text="相同内容")
    b = KnowledgeChunk.from_document(doc, chunk_index=0, text="相同内容")
    c = KnowledgeChunk.from_document(doc, chunk_index=0, text="不同内容")

    assert a.point_id == b.point_id
    assert a.point_id != c.point_id


def test_same_document_chunk_index_in_other_project_has_different_point_id():
    doc_a = _document()
    doc_b = KnowledgeDocument.from_text(
        document_id="doc-001",
        company_id="company-a",
        project_id=120,
        filename="历史冲击强度报告.docx",
        source_type="manual_index",
        text="第一段。\n样品冲击强度下降问题分析。",
    )

    a = KnowledgeChunk.from_document(doc_a, chunk_index=0, text="相同内容")
    b = KnowledgeChunk.from_document(doc_b, chunk_index=0, text="相同内容")

    assert a.point_id != b.point_id
