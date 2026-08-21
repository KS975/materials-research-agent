from __future__ import annotations

import json
import re
from typing import Any

from llm.base import LLMProvider
from runtime.chat_attachments import ChatAttachmentStore
from schemas.user_context import UserContext


class CurrentAttachmentSkill:
    """Analyze current-chat attachments without creating a long-term index."""

    # Broad XLSX summaries need full-sheet coverage. The previous fixed 12-chunk
    # limit could silently omit the tail of a wide spreadsheet (for example,
    # rows 81-95 in a 13-chunk workbook). Keep an explicit context budget
    # instead of a hard first-N cutoff.
    XLSX_GENERIC_MAX_CHUNKS = 40
    XLSX_GENERIC_MAX_CHARS = 60000

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
                    "sheet_name": chunk.get("sheet_name"),
                    "row_start": chunk.get("row_start"),
                    "row_end": chunk.get("row_end"),
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
6. 末尾给出【附件依据】，使用 SOURCE 编号说明依据来自哪些文件/页、段落或 Excel 行范围。
7. 对 XLSX 的“分析/总结/概括”类问题，必须覆盖所有已提供 SOURCE，不得只总结前半部分；优先依据“工作表元数据”报告准确的最大行、最大列和解析覆盖范围。若第 1 行明显是表头，可把数据行数表述为“总行数减表头行”，但不要凭空估算。
8. 对 Excel 中百分比配方、重复材料名或疑似异常字段，只能基于原始单元格做数据质量提示；不得自行改正。百分比直接相加不等于 100% 时，应提示“可能存在不同计量基准或原始记录需核实”。
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
            "warnings": self._coverage_warnings(selected, attachments, message),
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
            # Spreadsheet summaries are different from retrieval questions: a
            # fixed first-N cut can hide late experiment rows. For XLSX, keep
            # document order and include the whole workbook while it fits a
            # bounded context budget. Other document types keep the historical
            # retrieval limit.
            xlsx_candidates = [
                item for item in candidates if item["attachment"].parser == "openpyxl"
            ]
            if xlsx_candidates:
                xlsx_candidates.sort(
                    key=lambda x: (
                        x["attachment"].filename,
                        int(x["chunk"].get("index") or 0),
                    )
                )
                chosen: list[dict[str, Any]] = []
                char_total = 0
                for item in xlsx_candidates:
                    chunk_chars = len(str(item["chunk"].get("text") or ""))
                    if chosen and (
                        len(chosen) >= self.XLSX_GENERIC_MAX_CHUNKS
                        or char_total + chunk_chars > self.XLSX_GENERIC_MAX_CHARS
                    ):
                        break
                    chosen.append(item)
                    char_total += chunk_chars
                return chosen

            # Broad PDF/DOCX summary keeps the existing bounded behavior.
            return candidates[:limit]

        positive = [x for x in candidates if x["score"] > 0]
        return (positive or candidates)[:limit]

    def _coverage_warnings(
        self,
        selected: list[dict[str, Any]],
        attachments: list,
        message: str,
    ) -> list[str]:
        generic_analysis = any(
            word in message
            for word in ("分析", "总结", "概括", "这份报告", "这个文件", "附件", "文档")
        )
        if not generic_analysis:
            return []

        selected_by_attachment: dict[str, set[int]] = {}
        for item in selected:
            attachment = item["attachment"]
            selected_by_attachment.setdefault(attachment.attachment_id, set()).add(
                int(item["chunk"].get("index") or 0)
            )

        warnings: list[str] = []
        for attachment in attachments:
            if attachment.parser != "openpyxl":
                continue
            selected_count = len(selected_by_attachment.get(attachment.attachment_id, set()))
            if selected_count < attachment.chunk_count:
                warnings.append(
                    f"XLSX 宽表摘要受上下文预算限制：{attachment.filename} "
                    f"已覆盖 {selected_count}/{attachment.chunk_count} 个数据块；"
                    "建议针对未覆盖行范围继续追问。"
                )
        return warnings

    @staticmethod
    def _terms(text: str) -> set[str]:
        latin = {x.lower() for x in re.findall(r"[A-Za-z0-9_.%℃²/-]{2,}", text)}
        chinese = re.sub(r"[^\u4e00-\u9fff]", "", text)
        grams = {chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))}
        return latin | grams

    @staticmethod
    def _location(chunk: dict) -> str:
        if chunk.get("sheet_name"):
            start = chunk.get("row_start")
            end = chunk.get("row_end")
            if start:
                return (
                    f"工作表 {chunk['sheet_name']} / 行 {start}-{end or start} "
                    f"/ chunk {chunk.get('index')}"
                )
            return f"工作表 {chunk['sheet_name']} / chunk {chunk.get('index')}"
        if chunk.get("page"):
            return f"第 {chunk['page']} 页 / chunk {chunk.get('index')}"
        start = chunk.get("paragraph_start")
        end = chunk.get("paragraph_end")
        if start:
            return f"段落 {start}-{end or start} / chunk {chunk.get('index')}"
        return f"chunk {chunk.get('index')}"
