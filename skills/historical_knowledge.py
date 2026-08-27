from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable

from llm.base import LLMProvider
from runtime.progress import emit_progress
from schemas.user_context import UserContext


class HistoricalKnowledgeRAGSkill:
    """Answer from long-term knowledge stored in Qdrant.

    Boundary:
    - this skill does NOT query MySQL;
    - this skill does NOT read current Chat temporary attachments;
    - every retrieval is always restricted by company_id;
    - project scope is either one explicit authorized project, the caller's
      authorized project list, or (only when UserContext.all_projects=True)
      every project inside the current company;
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
        project_id: int | None,
        ctx: UserContext,
    ) -> dict[str, Any]:
        project_ids, all_projects, scope = self._resolve_retrieval_scope(
            ctx=ctx,
            project_id=project_id,
        )

        emit_progress(
            "knowledge_search",
            "running",
            "检索历史知识库",
            f"正在用“{message}”检索 {scope['display_name']}。",
            query_preview=message,
            detail_items=[
                {"label": "检索范围", "value": scope["display_name"]},
                {"label": "最多返回", "value": f"{self.max_hits} 个知识片段"},
                {"label": "相似度门槛", "value": f"{self.score_threshold:.2f}"},
            ],
        )

        with self.repository_opener() as repo:
            hits = repo.search(
                query=message,
                company_id=ctx.company_id,
                project_ids=project_ids,
                all_projects=all_projects,
                limit=self.max_hits,
                score_threshold=self.score_threshold,
            )

        hit_previews = [
            {
                "rank": rank,
                "filename": hit.chunk.filename,
                "project_id": hit.chunk.project_id,
                "score": round(float(hit.score), 4),
                "location": self._location(hit.chunk),
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        emit_progress(
            "knowledge_search",
            "completed",
            "历史资料检索完成",
            (
                f"命中 {len(hits)} 个达到门槛的历史片段。"
                if hits
                else "当前已索引历史资料中没有命中达到相似度门槛的片段。"
            ),
            query_preview=message,
            evidence_preview=hit_previews,
            detail_items=[
                {"label": "可靠命中", "value": f"{len(hits)} 个"},
                {"label": "检索范围", "value": scope["display_name"]},
            ],
        )

        if not hits:
            return {
                "status": "no_relevant_history",
                "answer": (
                    f"{scope['display_name']}中未检索到足够相关的历史资料。"
                    "这不代表历史上一定没有类似情况，只能说明当前已索引资料中没有找到可靠匹配。"
                ),
                "project_id": project_id,
                "retrieval_scope": scope,
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
1. 这些来源只来自当前用户有权限访问的当前公司历史 Knowledge Index；如果检索范围跨多个项目，必须保留并展示每条来源自己的 project_id。
2. 不得把数据库中未提供的数据、当前 Chat 临时附件、外部知识写成历史资料事实。
3. 找到相似记录时，要说明“有哪些相似点”，但不得因为相似就断言因果关系相同。
4. 对“历史有没有类似问题”这类问题：
   - 先给结论：检索到 / 未检索到；
   - 再列出最相关历史资料和相似点；
   - 明确证据边界或缺口。
5. 数值、单位、样品名、工艺、测试条件必须保持来源原意。
6. 不要复述密码、API Key、Token 等敏感凭证；来源若有 [REDACTED]，保持 [REDACTED]。
7. 末尾给出【历史资料依据】，按 SOURCE 编号说明文件、project_id 和页/段落/chunk。
8. “没有检索到”只能表述为当前已索引资料中没有足够相关证据，不能证明现实中绝对不存在。
9. 如果不同 SOURCE 来自不同项目，不得把它们合并成同一个实验或同一个项目事实。
回答中文。
"""
        user = (
            f"用户问题：{message}\n"
            f"历史检索范围：{scope['display_name']}\n"
            f"company_id：{ctx.company_id}\n\n"
            "HISTORICAL KNOWLEDGE SOURCES:\n"
            + "\n\n".join(source_blocks)
        )

        emit_progress(
            "knowledge_answer",
            "running",
            "阅读命中片段并组织回答",
            "正在比较历史资料中的相似点、差异点和证据边界。",
            evidence_preview=hit_previews,
            plan_summary="历史相似性只作为类比证据，不直接推断当前问题具有相同原因。",
        )
        answer = self.llm.complete(system, user)
        emit_progress(
            "knowledge_answer",
            "completed",
            "历史知识回答已生成",
            f"回答引用了 {len(hits)} 个授权历史片段，并保留每条来源的位置与项目范围。",
            detail_items=[{"label": "引用片段", "value": f"{len(hits)} 个"}],
        )
        return {
            "status": "ok",
            "answer": answer,
            "project_id": project_id,
            "retrieval_scope": scope,
            "hit_count": len(hits),
            "score_threshold": self.score_threshold,
            "evidence": evidence,
            "warnings": [],
        }

    @staticmethod
    def _resolve_retrieval_scope(
        *,
        ctx: UserContext,
        project_id: int | None,
    ) -> tuple[list[int], bool, dict[str, Any]]:
        if project_id is not None:
            project_id = int(project_id)
            if not ctx.can_access_project(project_id):
                raise PermissionError("当前用户无权检索该项目历史知识")
            return (
                [project_id],
                False,
                {
                    "mode": "explicit_project",
                    "company_id": ctx.company_id,
                    "project_ids": [project_id],
                    "display_name": f"当前公司 Project {project_id} 的历史知识库",
                },
            )

        if ctx.all_projects:
            return (
                [],
                True,
                {
                    "mode": "company_all_projects",
                    "company_id": ctx.company_id,
                    "project_ids": "*",
                    "display_name": "当前公司全部项目的历史知识库",
                },
            )

        if ctx.project_ids:
            project_ids = sorted({int(item) for item in ctx.project_ids})
            return (
                project_ids,
                False,
                {
                    "mode": "authorized_projects",
                    "company_id": ctx.company_id,
                    "project_ids": project_ids,
                    "display_name": "当前用户已授权项目范围的历史知识库",
                },
            )

        raise PermissionError("当前用户没有可用于历史知识检索的项目权限")

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
