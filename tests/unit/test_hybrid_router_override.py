from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter


class ClarifyingLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "domain": "conversation",
                "primary_intent": "clarification_required",
                "tool_name": None,
                "tool_args": {},
                "needs_clarification": True,
                "clarification_question": "请说明竞品。",
                "reasoning_summary": "模型请求澄清",
            },
            ensure_ascii=False,
        )


def test_deterministic_hybrid_research_overrides_model_clarification():
    decision = DeepSeekIntentRouter(ClarifyingLLM()).route(
        "与竞品 ABS 的性能差距是多少？结合历史数据判断。"
    )
    assert decision.intent == "hybrid_research_qa"
    assert decision.needs_clarification is False
