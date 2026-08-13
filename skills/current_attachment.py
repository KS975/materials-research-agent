from __future__ import annotations

import json
import re
from typing import Any

from llm.base import LLMProvider
from runtime.chat_attachments import ChatAttachmentStore
from schemas.user_context import UserContext


class CurrentAttachmentSkill:
    """Analyze current-chat attachments without creating a long-term index."""

    def __init__(self, store: ChatAttachmentStore, llm: LLMProvider):
        self.store = store
        self.llm = llm

    def answer(
        self,
        *,
        message: str,
        attachment_ids: list[str],
        ctx: UserContext,
    ) -> dict[str, Any]:
        if not attachment_ids:
            raise ValueError("当前问题没有携带 Chat 附件")

        attachments = [self.store.get(item, ctx) for item in attachment_ids]
        selected = self._select_chunks(message, attachments, limit=12)
        if not selected:
            return {
                "status": "no_text",
                "answer": "当前附件没有可用于回答的文本内容。",
                "evidence": [],
                "warnings": ["附件解析结果为空"],
            }

        evidence = []
        source_blocks = []
        for rank, item in enumerate(selected, start=1):
            chunk = item["chunk"]
            attachment = item["attachment"]
            location = self._location(chunk)
            source_blocks.append(
                f"[SOURCE {rank}] 文件={attachment.filename}；{location}\n{chunk.get('text', '')}"
            )
            evidence.append(
                {
                    "source": "chat_attachment",
                    "attachment_id": attachment.attachment_id,
                    "filename": attachment.filename,
                    "page": chunk.get("page"),
                    "paragraph_start": chunk.get("paragraph_start"),
                    "paragraph_end": chunk.get("paragraph_end"),
                    "chunk_index": chunk.get("index"),
                }
            )

        system = """你是材数智能体 V0.1.2 当前附件分析器。
你只能依据提供的 CURRENT CHAT ATTACHMENT SOURCES 回答，不得使用数据库中未提供的信息，也不得补充外部材料知识作为文件事实。
规则：
1. 先回答用户问题，不要复述系统规则。
2. 文件没有写到的内容明确说“附件中未找到”。
3. 不得把猜测写成文件事实。
4. 如果用户要求“分析/总结这份报告”，按报告真实内容提炼：目的/对象、方法或工艺、测试或结果、结论/问题；某项没有就省略或说明未找到。
5. 数值、单位、样品名、条件必须保持来源原文语义。
6. 末尾给出【附件依据】，使用 SOURCE 编号说明依据来自哪些文件/页或段落。
回答中文。
"""
        user = (
            f"用户问题：{message}\n\n"
            "CURRENT CHAT ATTACHMENT SOURCES:\n"
            + "\n\n".join(source_blocks)
        )
        answer = self.llm.complete(system, user)
        return {
            "status": "ok",
            "answer": answer,
            "attachments": [
                {
                    "attachment_id": x.attachment_id,
                    "filename": x.filename,
                    "parser": x.parser,
                    "page_count": x.page_count,
                    "char_count": x.char_count,
                    "chunk_count": x.chunk_count,
                }
                for x in attachments
            ],
            "evidence": evidence,
            "warnings": [],
        }

    def _select_chunks(self, message: str, attachments: list, limit: int) -> list[dict[str, Any]]:
        query_terms = self._terms(message)
        generic_analysis = any(
            word in message
            for word in ("分析", "总结", "概括", "这份报告", "这个文件", "附件", "文档")
        )
        candidates = []
        for attachment in attachments:
            for chunk in attachment.chunks:
                text = str(chunk.get("text") or "")
                terms = self._terms(text)
                score = len(query_terms & terms)
                if generic_analysis:
                    # Preserve document order for broad report analysis.
                    score += max(0, 6 - int(chunk.get("index") or 0)) * 0.01
                candidates.append(
                    {
                        "score": score,
                        "attachment": attachment,
                        "chunk": chunk,
                    }
                )
        candidates.sort(
            key=lambda x: (
                -x["score"],
                x["attachment"].filename,
                int(x["chunk"].get("index") or 0),
            )
        )
        if generic_analysis:
            # Broad summary should cover the beginning and additional ranked chunks.
            chosen = candidates[:limit]
        else:
            positive = [x for x in candidates if x["score"] > 0]
            chosen = (positive or candidates)[:limit]
        return chosen

    @staticmethod
    def _terms(text: str) -> set[str]:
        latin = {x.lower() for x in re.findall(r"[A-Za-z0-9_.%℃²/-]{2,}", text)}
        chinese = re.sub(r"[^\u4e00-\u9fff]", "", text)
        grams = {chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))}
        return latin | grams

    @staticmethod
    def _location(chunk: dict) -> str:
        if chunk.get("page"):
            return f"第 {chunk['page']} 页 / chunk {chunk.get('index')}"
        start = chunk.get("paragraph_start")
        end = chunk.get("paragraph_end")
        if start:
            return f"段落 {start}-{end or start} / chunk {chunk.get('index')}"
        return f"chunk {chunk.get('index')}"
