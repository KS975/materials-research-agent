from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system: str, user: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _history_before_history_question():
    return [
        {"role": "user", "content": "帮我看看3073"},
        {"role": "assistant", "content": "已查询3073"},
        {"role": "user", "content": "不是3073，是3081"},
        {"role": "assistant", "content": "已查询3081"},
        {"role": "user", "content": "不是3081，是3811，我弄错了"},
        {"role": "assistant", "content": "已查询3811"},
        {"role": "user", "content": "我可以只查看性能么"},
        {"role": "assistant", "content": "冲击强度24"},
    ]


def test_referential_history_question_becomes_sample_history_analysis():
    router = DeepSeekIntentRouter(FakeLLM({
        # Even if the LLM returns generic RAG, backend V2.1 upgrades it because
        # “这个” has one deterministic active sample in user history.
        "primary_intent": "search_historical_knowledge",
        "tool_name": None,
        "tool_args": {},
    }))

    decision = router.route(
        "历史上有没有和这个类似的冲击强度异常？",
        history=_history_before_history_question(),
    )

    assert decision.intent == "sample_historical_similarity"
    assert decision.tool_args["identifier"] == "3811"
    assert decision.tool_args["target_metric"] == "冲击强度"
    assert "样品3811" in decision.tool_args["history_query"]
    assert [step.name for step in decision.tool_plan] == [
        "get_sample_context",
        "historical_knowledge",
        "sample_historical_similarity",
    ]


def test_project_only_followup_keeps_previous_history_task_and_query():
    history = _history_before_history_question() + [
        {"role": "user", "content": "历史上有没有和3811类似的冲击强度异常？"},
        {"role": "assistant", "content": "检索到一条类似历史记录"},
    ]
    router = DeepSeekIntentRouter(FakeLLM({
        # This intentionally simulates the bad V2 behavior: the model tries to
        # treat “Project 115呢？” as a new generic history request.
        "primary_intent": "search_historical_knowledge",
        "tool_name": None,
        "tool_args": {},
    }))

    decision = router.route("Project 115呢？", history=history)

    assert decision.intent == "sample_historical_similarity"
    assert decision.tool_args["identifier"] == "3811"
    assert decision.tool_args["target_metric"] == "冲击强度"
    assert decision.tool_args["project_id"] == 115
    assert decision.tool_args["history_query"] == "历史上有没有和3811类似的冲击强度异常？"
    assert decision.context_reference["use_previous_task"] is True
    assert decision.scope["projects"] == [115]


def test_all_projects_followup_resets_scope_but_keeps_history_task():
    history = _history_before_history_question() + [
        {"role": "user", "content": "历史上有没有和3811类似的冲击强度异常？"},
        {"role": "assistant", "content": "检索到历史记录"},
        {"role": "user", "content": "Project 115呢？"},
        {"role": "assistant", "content": "Project115范围结果"},
    ]
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "search_historical_knowledge",
        "tool_name": None,
        "tool_args": {"project_id": 115},
    }))

    decision = router.route("那全部项目呢？", history=history)

    assert decision.intent == "sample_historical_similarity"
    assert decision.tool_args["identifier"] == "3811"
    assert "project_id" not in decision.tool_args
    assert decision.scope["projects"] == "all_authorized"
    assert decision.tool_args["history_query"] == "历史上有没有和3811类似的冲击强度异常？"


def test_active_sample_uses_latest_explicit_correction_not_all_old_ids():
    router = DeepSeekIntentRouter(FakeLLM({
        "primary_intent": "get_formula",
        "tool_name": "get_formula",
        "tool_args": {},
        "context_reference": {"action": "refine_previous", "use_previous_sample": True},
    }))

    decision = router.route("只看配方", history=_history_before_history_question())
    assert decision.tool_args["identifier"] == "3811"
