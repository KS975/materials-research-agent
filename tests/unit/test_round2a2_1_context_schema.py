from __future__ import annotations

from agent.conversation_context import (
    CONVERSATION_CONTEXT_SCHEMA_VERSION,
    build_conversation_hints,
)
from agent.deepseek_intent_router import INTENT_ROUTER_CONTEXT_SCHEMA_VERSION


def test_router_and_context_schema_versions_match():
    assert CONVERSATION_CONTEXT_SCHEMA_VERSION == INTENT_ROUTER_CONTEXT_SCHEMA_VERSION


def test_task_refinement_intent_is_present_and_deterministic():
    hints = build_conversation_hints(
        "那我只想看看性能",
        [{"role": "user", "content": "查看样品3811具体信息"}],
    )
    assert hints.task_refinement_intent == "get_performance"
    assert hints.active_sample_identifier == "3811"

