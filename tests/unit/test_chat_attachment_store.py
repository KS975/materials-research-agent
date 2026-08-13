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
