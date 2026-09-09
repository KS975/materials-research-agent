from __future__ import annotations

import json

from agent.deepseek_intent_router import DeepSeekIntentRouter
from api.chat_ui import _looks_like_hybrid_research


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


def test_stage4_analysis_phrases_use_hybrid_research_router():
    for message in (
        "哪些变量影响冲击强度？结合历史资料分析。",
        "找冲击强度大于35、MFR小于18的稳定工艺窗口，并结合资料。",
        "冲击强度和MFR为什么冲突？结合历史资料。",
        "分析正常批次和异常批次差异，并结合资料。",
        "生成当前项目阶段总结报告，并结合历史资料。",
    ):
        assert _looks_like_hybrid_research(message)

    decision = DeepSeekIntentRouter(ClarifyingLLM()).route(
        "哪些变量影响冲击强度？结合历史资料分析。"
    )
    assert decision.intent == "hybrid_research_qa"
