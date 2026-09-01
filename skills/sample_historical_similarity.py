from __future__ import annotations

import json
from contextlib import AbstractContextManager
from typing import Any, Callable

from agent.tool_registry import ToolRegistry
from llm.base import LLMProvider
from runtime.progress import emit_progress
from schemas.user_context import UserContext
from skills.historical_knowledge import HistoricalKnowledgeRAGSkill


class SampleHistoricalSimilaritySkill:
    """Compare one current MySQL sample with historical RAG evidence.

    This is intentionally different from the two-sample joint analysis:
    - one sample is fetched from the read-only business database;
    - historical retrieval may span every authorized project in the current
      company, or one explicit project selected by the user;
    - historical similarity is evidence for analogy only, never proof of the
      same root cause.
    """

    def __init__(
        self,
        registry: ToolRegistry,
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
        self.registry = registry
        self.repository_opener = repository_opener
        self.llm = llm
        self.score_threshold = float(score_threshold)
        self.max_hits = int(max_hits)

    def answer(
        self,
        *,
        message: str,
        history_query: str,
        identifier: str | int,
        target_metric: str = "",
        project_id: int | None,
        ctx: UserContext,
    ) -> dict[str, Any]:
        emit_progress(
            "sample_context_query",
            "running",
            f"读取当前样品 {identifier}",
            "正在从业务 MySQL 读取样品身份、配方、工艺、性能和测试条件。",
            detail_items=[
                {"label": "只读工具", "value": "get_sample_context"},
                {"label": "样品标识", "value": str(identifier)},
                {"label": "关注指标", "value": target_metric or "未限定"},
            ],
        )
        database_result = self.registry.execute(
            "get_sample_context",
            identifier=identifier,
            ctx=ctx,
        )
        if not isinstance(database_result, dict) or database_result.get("status") != "ok":
            emit_progress(
                "sample_context_query",
                "failed",
                "当前样品读取失败",
                "没有在当前授权数据库范围内确定该样品。",
            )
            return {
                "status": "database_error",
                "answer": "当前样品无法在授权数据库范围内确定，因此没有继续做历史相似案例分析。",
                "identifier": identifier,
                "project_id": project_id,
                "database_result": database_result,
                "knowledge_hits": [],
                "evidence": self._mysql_evidence(database_result),
                "warnings": [],
            }

        project_ids, all_projects, scope = HistoricalKnowledgeRAGSkill._resolve_retrieval_scope(
            ctx=ctx,
            project_id=project_id,
        )
        effective_query = self._knowledge_query(
            message=message,
            history_query=history_query,
            identifier=identifier,
            database_result=database_result,
            target_metric=target_metric,
        )

        sample = database_result.get("sample") or {}
        emit_progress(
            "sample_context_query",
            "completed",
            "当前样品事实已读取",
            (
                f"已读取 {sample.get('id', identifier)}"
                + (f"（{sample.get('name')}）" if sample.get("name") else "")
                + f"，Project {sample.get('project_id', '-')}。"
            ),
            detail_items=[
                {"label": "样品", "value": str(sample.get("id", identifier))},
                {"label": "名称", "value": str(sample.get("name") or "未记录")},
                {"label": "Project", "value": str(sample.get("project_id", "-"))},
                {"label": "关注指标", "value": target_metric or "未限定"},
            ],
        )

        emit_progress(
            "sample_history_search",
            "running",
            "检索相似历史案例",
            f"正在用当前样品事实构造的检索词搜索 {scope['display_name']}。",
            query_preview=effective_query,
            detail_items=[
                {"label": "检索范围", "value": scope["display_name"]},
                {"label": "最多返回", "value": f"{self.max_hits} 个知识片段"},
                {"label": "相似度门槛", "value": f"{self.score_threshold:.2f}"},
            ],
        )

        with self.repository_opener() as repo:
            hits = repo.search(
                query=effective_query,
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
                "location": HistoricalKnowledgeRAGSkill._location(hit.chunk),
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        emit_progress(
            "sample_history_search",
            "completed",
            "相似历史案例检索完成",
            (
                f"命中 {len(hits)} 个达到门槛的历史片段。"
                if hits
                else "当前已索引历史资料中没有命中达到相似度门槛的片段。"
            ),
            query_preview=effective_query,
            evidence_preview=hit_previews,
            detail_items=[{"label": "可靠命中", "value": f"{len(hits)} 个"}],
        )

        sample_label = f"{sample.get('id', identifier)}"
        if sample.get("name"):
            sample_label += f"（{sample['name']}）"

        mysql_evidence = self._mysql_evidence(database_result)
        if not hits:
            return {
                "status": "no_relevant_history",
                "answer": (
                    f"已读取当前样品 {sample_label} 的数据库事实，但在"
                    f"{scope['display_name']}中未检索到足够相关的历史案例。"
                    "这只能说明当前已索引资料中没有可靠匹配，不能证明历史上绝对不存在类似情况。"
                ),
                "identifier": identifier,
                "target_metric": target_metric,
                "project_id": project_id,
                "history_query": effective_query,
                "analysis_scope": scope,
                "database_result": database_result,
                "knowledge_hits": [],
                "evidence": mysql_evidence,
                "warnings": [
                    f"未找到相似度达到 {self.score_threshold:.2f} 的历史知识片段"
                ],
            }

        source_blocks: list[str] = []
        knowledge_evidence: list[dict[str, Any]] = []
        compact_hits: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            location = HistoricalKnowledgeRAGSkill._location(chunk)
            source_blocks.append(
                f"[SOURCE {rank}] score={hit.score:.6f}；"
                f"文件={chunk.filename}；project_id={chunk.project_id}；"
                f"source_id={chunk.source_id or '-'}；{location}\n{chunk.text}"
            )
            item = {
                "source": "knowledge_index",
                "evidence_type": "historical_knowledge",
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
            knowledge_evidence.append(item)
            compact_hits.append({
                "score": hit.score,
                "filename": chunk.filename,
                "project_id": chunk.project_id,
                "source_id": chunk.source_id,
                "chunk_index": chunk.chunk_index,
            })

        system = """你是“材数智能体”的单样品 + 历史知识联合分析器。
你必须同时使用 DATABASE FACTS 与 HISTORICAL KNOWLEDGE SOURCES，但严格区分两种证据。

规则：
1. DATABASE FACTS 是当前样品的只读数据库事实，不得修改或补造。
2. HISTORICAL KNOWLEDGE SOURCES 是当前公司授权范围内的历史资料；跨项目命中时必须保留每条来源自己的 project_id。
3. 用户问“这个样品历史上有没有类似问题/异常/案例”时：先明确是否检索到相似历史证据，再说明当前样品与历史案例的相似点和不同点。
4. 相似不等于相同原因。不得把历史案例中的原因直接断言为当前样品的原因。
5. 如果当前样品缺测试条件、关键工艺、配方计量基准等，要明确列为证据缺口。
6. 数值、单位、样品名必须保持 DATABASE FACTS / SOURCE 原意。
7. 结尾分成【数据库依据】和【历史资料依据】；历史资料按 SOURCE 编号注明文件、project_id 和位置。
8. 不得把当前样品所属 project 与历史来源 project 混为一谈；用户限定的 project_id 只是历史检索范围。
9. 不得输出或推断任何密码、Token、API Key。
回答中文。
"""

        user = (
            f"用户当前表达：{message}\n"
            f"延续后的历史检索主题：{history_query or message}\n"
            f"实际历史检索 Query：{effective_query}\n"
            f"关注指标：{target_metric or '未限定'}\n"
            f"历史检索范围：{scope['display_name']}\n\n"
            "DATABASE FACTS:\n"
            + json.dumps(database_result, ensure_ascii=False, default=str)
            + "\n\nHISTORICAL KNOWLEDGE SOURCES:\n"
            + "\n\n".join(source_blocks)
        )
        emit_progress(
            "sample_history_synthesis",
            "running",
            "比较当前样品与历史案例",
            "正在区分数据库事实、历史相似点、差异点和需验证的假设。",
            evidence_preview=hit_previews,
            plan_summary="历史案例只用于类比，不会把历史原因直接套用到当前样品。",
        )
        answer = self.llm.complete(system, user)
        emit_progress(
            "sample_history_synthesis",
            "completed",
            "样品与历史案例分析完成",
            f"回答已联合当前样品事实和 {len(hits)} 个历史片段。",
            detail_items=[
                {"label": "当前样品", "value": sample_label},
                {"label": "历史命中", "value": f"{len(hits)} 个"},
            ],
        )

        return {
            "status": "ok",
            "analysis_type": "sample_historical_similarity",
            "answer": answer,
            "identifier": identifier,
            "target_metric": target_metric,
            "project_id": project_id,
            "history_query": effective_query,
            "analysis_scope": scope,
            "database_result": database_result,
            "knowledge_hits": compact_hits,
            "hit_count": len(hits),
            "score_threshold": self.score_threshold,
            "evidence": mysql_evidence + knowledge_evidence,
            "warnings": list(database_result.get("warnings", []) or []),
        }

    @staticmethod
    def _knowledge_query(
        *,
        message: str,
        history_query: str,
        identifier: str | int,
        database_result: dict[str, Any],
        target_metric: str,
    ) -> str:
        base = str(history_query or message or "").strip()
        sample = database_result.get("sample") or {}
        sample_name = str(sample.get("name") or "").strip()
        metric = str(target_metric or "").strip()

        descriptors = [f"当前样品 {identifier}"]
        if sample_name:
            descriptors.append(sample_name)
        if metric:
            descriptors.append(f"关注性能 {metric}")
            matched = [
                item for item in database_result.get("performance", []) or []
                if metric in str(item.get("name") or item.get("raw_key") or "")
                or str(item.get("name") or item.get("raw_key") or "") in metric
            ]
            if len(matched) == 1:
                descriptors.append(
                    f"数据库记录值 {matched[0].get('value')} {matched[0].get('unit') or ''}".strip()
                )
        return base + "\n" + "；".join(descriptors)

    @staticmethod
    def _mysql_evidence(database_result: Any) -> list[dict[str, Any]]:
        if not isinstance(database_result, dict):
            return []
        output: list[dict[str, Any]] = []
        for item in database_result.get("evidence", []) or []:
            if isinstance(item, dict):
                output.append({**item, "evidence_type": "mysql"})
        return output
