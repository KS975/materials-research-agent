from dataclasses import replace

import pytest

from file_processing.models import ParsedChunk, ParsedDocument
from runtime.chat_attachments import ChatAttachmentStore
from schemas.user_context import UserContext


def ctx(projects=(115,)):
    return UserContext(
        user_id="u1",
        company_id="c1",
        project_ids=projects,
        permission_source="test",
    )


def test_attachment_owner_and_project_scope(tmp_path):
    store = ChatAttachmentStore(str(tmp_path), ttl_minutes=30)
    parsed = ParsedDocument(
        filename="a.pdf",
        media_type="application/pdf",
        parser="test",
        text="hello",
        chunks=(ParsedChunk(index=0, text="hello", page=1),),
        page_count=1,
    )
    item = store.save(
        ctx=ctx(),
        filename="a.pdf",
        media_type="application/pdf",
        original_bytes=b"abc",
        parsed=parsed,
    )
    assert store.get(item.attachment_id, ctx()).filename == "a.pdf"

    wrong_company = UserContext("u1", "c2", (115,), "test")
    with pytest.raises(PermissionError):
        store.get(item.attachment_id, wrong_company)

    with pytest.raises(PermissionError):
        store.get(item.attachment_id, ctx((120,)))


def test_attachment_all_projects_scope_is_company_bounded(tmp_path):
    store = ChatAttachmentStore(str(tmp_path), ttl_minutes=30)
    parsed = ParsedDocument(
        filename="all.pdf",
        media_type="application/pdf",
        parser="test",
        text="hello",
        chunks=(ParsedChunk(index=0, text="hello", page=1),),
        page_count=1,
    )
    all_ctx = UserContext(
        user_id="u1",
        company_id="c1",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )
    item = store.save(
        ctx=all_ctx,
        filename="all.pdf",
        media_type="application/pdf",
        original_bytes=b"abc",
        parsed=parsed,
    )

    assert store.get(item.attachment_id, all_ctx).filename == "all.pdf"

    with pytest.raises(PermissionError):
        store.get(item.attachment_id, ctx((115,)))

    other_company_all = UserContext(
        user_id="u1",
        company_id="c2",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )
    with pytest.raises(PermissionError):
        store.get(item.attachment_id, other_company_all)
