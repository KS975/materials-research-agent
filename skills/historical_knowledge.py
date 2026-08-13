from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable

from llm.base import LLMProvider
from schemas.user_context import UserContext


class HistoricalKnowledgeRAGSkill:
    """Answer from long-term project knowledge stored in Qdrant.

    Boundary:
    - this skill does NOT query MySQL;
    - this skill does NOT read current Chat temporary attachments;
    - every retrieval is restricted by company_id + one authorized project_id;
    - low-scoring retrievals are filtered before they are sent to the LLM.
    """

    def __init__(
        self,
        repository_opener: Callable[[], AbstractContextManager],
        llm: LLMProvider,
        *,
        score_threshold: float = 0.42,
        max_hits: int = 5,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if max_hits <= 0:
            raise ValueError("max_hits must be > 0")

        self.repository_opener = repository_opener
        self.llm = llm
        self.score_threshold = float(score_threshold)
        self.max_hits = int(max_hits)

    def answer(
        self,
        *,
        message: str,
        project_id: int,
        ctx: UserContext,
    ) -> dict[str, Any]:
        if not ctx.can_access_project(project_id):
            raise PermissionError("当前用户无权检索该项目历史知识")

        with self.repository_opener() as repo:
            hits = repo.search(
                query=message,
                company_id=ctx.company_id,
                project_ids=[project_id],
                limit=self.max_hits,
                score_threshold=self.score_threshold,
            )

        if not hits:
            return {
                "status": "no_relevant_history",
                "answer": (
                    "当前项目的历史知识库中未检索到足够相关的资料。"
                    "这不代表历史上一定没有类似情况，只能说明当前已索引资料中没有找到可靠匹配。"
                ),
                "project_id": project_id,
                "evidence": [],
                "warnings": [
                    f"未找到相似度达到 {self.score_threshold:.2f} 的历史知识片段"
                ],
            }

        evidence: list[dict[str, Any]] = []
        source_blocks: list[str] = []

        for rank, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            location = self._location(chunk)
            source_blocks.append(
                f"[SOURCE {rank}] "
                f"score={hit.score:.6f}；"
                f"文件={chunk.filename}；"
                f"project_id={chunk.project_id}；"
                f"source_id={chunk.source_id or '-'}；"
                f"{location}\n"
                f"{chunk.text}"
            )
            evidence.append(
                {
                    "source": "knowledge_index",
                    "score": hit.score,
                    "document_id": chunk.document_id,
                    "source_id": chunk.source_id,
                    "filename": chunk.filename,
                    "project_id": chunk.project_id,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page_number,
                    "paragraph_start": chunk.paragraph_start,
                    "paragraph_end": chunk.paragraph_end,
                    "locator_type": chunk.locator_type,
                }
            )

        system = """你是“材数智能体”V0.1.2 的历史知识 RAG 分析器。
你只能依据提供的 HISTORICAL KNOWLEDGE SOURCES 回答当前问题。

规则：
1. 这些来源来自当前用户有权限访问的项目历史 Knowledge Index。
2. 不得把数据库中未提供的数据、当前 Chat 临时附件、外部知识写成历史资料事实。
3. 找到相似记录时，要说明“有哪些相似点”，但不得因为相似就断言因果关系相同。
4. 对“历史有没有类似问题”这类问题：
   - 先给结论：检索到 / 未检索到；
   - 再列出最相关历史资料和相似点；
   - 明确证据边界或缺口。
5. 数值、单位、样品名、工艺、测试条件必须保持来源原意。
6. 不要复述密码、API Key、Token 等敏感凭证；来源若有 [REDACTED]，保持 [REDACTED]。
7. 末尾给出【历史资料依据】，按 SOURCE 编号说明文件和页/段落/chunk。
8. “没有检索到”只能表述为当前已索引资料中没有足够相关证据，不能证明现实中绝对不存在。
回答中文。
"""
        user = (
            f"用户问题：{message}\n"
            f"当前检索项目：{project_id}\n\n"
            "HISTORICAL KNOWLEDGE SOURCES:\n"
            + "\n\n".join(source_blocks)
        )

        answer = self.llm.complete(system, user)
        return {
            "status": "ok",
            "answer": answer,
            "project_id": project_id,
            "hit_count": len(hits),
            "score_threshold": self.score_threshold,
            "evidence": evidence,
            "warnings": [],
        }

    @staticmethod
    def _location(chunk) -> str:
        if chunk.page_number is not None:
            return f"第 {chunk.page_number} 页 / chunk {chunk.chunk_index}"
        if chunk.paragraph_start is not None:
            end = (
                chunk.paragraph_end
                if chunk.paragraph_end is not None
                else chunk.paragraph_start
            )
            return (
                f"段落 {chunk.paragraph_start}-{end} "
                f"/ chunk {chunk.chunk_index}"
            )
        return f"chunk {chunk.chunk_index}"
