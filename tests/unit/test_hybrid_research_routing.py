from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter


class FakeLLM:
    def complete(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "domain": "knowledge",
                "primary_intent": "hybrid_research_qa",
                "secondary_intents": [],
                "entities": {},
                "scope": {"company": "current", "projects": [115]},
                "constraints": {"cross_source": True},
                "context_reference": {"action": "new_request"},
                "tool_name": None,
                "tool_args": {"project_id": 115},
                "tool_plan": [],
                "needs_clarification": False,
                "clarification_question": "",
                "reasoning_summary": "需要综合数据库与历史资料",
            },
            ensure_ascii=False,
        )


def test_hybrid_research_intent_is_no_tool_and_fixed_workflow():
    decision = DeepSeekIntentRouter(FakeLLM()).route(
        "查 Project 115 中 EXP-128 的研发上下文，并结合历史资料综合判断。"
    )

    assert decision.intent == "hybrid_research_qa"
    assert decision.tool_name is None
    assert decision.tool_args["project_id"] == 115
    assert decision.tool_args["identifier"] == "EXP-128"
    routing = decision.to_routing_meta()
    assert routing["tool_plan"][0]["kind"] == "workflow"
    assert routing["tool_plan"][0]["name"] == "hybrid_research_qa"
