from __future__ import annotations

import json
from typing import Any

from llm.base import LLMProvider


class GeneralConversationFallbackSkill:
    """DeepSeek answer path for turns that do not require an executable Tool.

    This path intentionally receives no database, attachment or RAG content. Its
    prompt therefore makes the evidence boundary explicit: general knowledge is
    allowed, while user/company-specific facts must be clarified or routed to a
    deterministic capability instead of being invented.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def answer(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        safe_history = []
        for item in (history or [])[-12:]:
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                safe_history.append({"role": role, "content": content[:8000]})

        system = """你是材数智能体的 DeepSeek 通用回答层。
当前这一轮没有执行数据库 Tool、附件解析、历史知识 RAG、模型计算或真实设备操作。

回答规则：
1. 普通材料科学、研发方法、概念解释、方案讨论和一般性问题，可以直接用已有知识回答。
2. 如果问题要求当前公司数据库、具体样品、当前附件、历史报告或实时运行状态中的事实，必须明确说明本轮没有取得这些证据，并提出一个简洁的补充信息或查询建议；不得假装已经查库、读附件或运行工具。
3. 可以提出明确标注的工程假设，但不得把假设写成数据库事实或实验结论。
4. 如果用户要求写数据库、控制真实设备、启动部署或执行其它本轮未授权动作，只能说明未执行，并提供安全的下一步；不得声称操作成功。
5. 结合对话历史理解代词和上下文，但历史中的助手回答不等于新的数据库证据。
6. 直接回答当前问题，不要重复罗列系统版本和全部功能，不要输出“当前请求尚未匹配到可执行能力”这类固定模板。
7. 信息不足时只问最关键的一个澄清问题。
回答使用中文，除非用户明确要求其它语言。
"""
        user_payload = {
            "conversation_history": safe_history,
            "current_user_message": str(message or "").strip(),
            "evidence_available_this_turn": {
                "database_tool": False,
                "attachment_parser": False,
                "historical_rag": False,
                "runtime_or_device_tool": False,
            },
        }
        answer = self.llm.complete(
            system,
            json.dumps(user_payload, ensure_ascii=False),
        ).strip()
        if not answer:
            raise RuntimeError("DeepSeek 通用回答为空")
        return {
            "status": "ok",
            "answer": answer,
            "response_mode": "deepseek_general_fallback",
            "tool_executed": False,
            "database_evidence_used": False,
            "attachment_evidence_used": False,
            "historical_evidence_used": False,
            "evidence": [],
            "warnings": [],
        }
