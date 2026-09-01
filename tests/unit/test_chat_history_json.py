from __future__ import annotations

import json

import pytest

from runtime.chat_history import (
    ChatHistoryNotFoundError,
    ChatHistoryStore,
    ChatHistoryValidationError,
)
from schemas.user_context import UserContext


def _ctx(user="user-a", company="company-a"):
    return UserContext(
        user_id=user,
        company_id=company,
        project_ids=(),
        permission_source="test",
        all_projects=True,
        organization_id="org-a",
        organization_level="1",
    )


def _append(store, ctx, conversation_id, message_id="message-0001"):
    return store.append_exchange(
        ctx=ctx,
        conversation_id=conversation_id,
        client_message_id=message_id,
        user_content="所有样品冲击强度的平均值是多少？",
        assistant_content="有效记录的平均值为 32.5 kJ/m²。",
        assistant_meta={"intent": "performance_statistics"},
        evidence=[{"source": "business_mysql"}],
    )


def test_json_history_round_trip_and_safe_owner_scope(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    conversation_id = store.new_conversation_id()
    summary = _append(store, _ctx(), conversation_id)

    assert summary["message_count"] == 2
    restored = store.get_conversation(_ctx(), conversation_id)
    assert restored["owner"]["user_id"] == "user-a"
    assert restored["owner"]["company_id"] == "company-a"
    assert restored["owner"]["organization_id"] == "org-a"
    assert [item["role"] for item in restored["messages"]] == ["user", "assistant"]
    assert "authorization" not in json.dumps(restored).lower()


def test_other_user_or_company_cannot_see_conversation(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    conversation_id = store.new_conversation_id()
    _append(store, _ctx(), conversation_id)

    assert store.list_conversations(_ctx(user="other"))["total"] == 0
    assert store.list_conversations(_ctx(company="other"))["total"] == 0
    with pytest.raises(ChatHistoryNotFoundError):
        store.get_conversation(_ctx(user="other"), conversation_id)


def test_duplicate_client_message_id_is_idempotent(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    conversation_id = store.new_conversation_id()
    _append(store, _ctx(), conversation_id)
    summary = _append(store, _ctx(), conversation_id)
    assert summary["message_count"] == 2


def test_rename_delete_and_invalid_identifier(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    conversation_id = store.new_conversation_id()
    _append(store, _ctx(), conversation_id)

    assert store.rename(_ctx(), conversation_id, "冲击强度统计")["title"] == "冲击强度统计"
    store.delete(_ctx(), conversation_id)
    with pytest.raises(ChatHistoryNotFoundError):
        store.get_conversation(_ctx(), conversation_id)
    with pytest.raises(ChatHistoryValidationError):
        store.get_conversation(_ctx(), "../../etc/passwd")


def test_message_limit_is_bounded(tmp_path):
    store = ChatHistoryStore(tmp_path / "history", max_messages=20)
    conversation_id = store.new_conversation_id()
    for index in range(15):
        _append(store, _ctx(), conversation_id, f"message-{index:04d}")
    restored = store.get_conversation(_ctx(), conversation_id)
    assert len(restored["messages"]) == 20
    assert restored["messages_trimmed"] is True
