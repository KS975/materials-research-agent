from __future__ import annotations

import json
from contextlib import AbstractContextManager
from typing import Any, Callable

from agent.tool_registry import ToolRegistry
from llm.base import LLMProvider
from schemas.user_context import UserContext
from skills.analysis import AnalysisSkill


class JointMySQLKnowledgeAnalysisSkill:
    """Combine read-only MySQL facts with historical Qdrant evidence.

    Safety boundaries:
    - MySQL access still goes through the existing Tool -> Repository path.
    - company_id is always enforced by the business repositories and Qdrant.
    - If the user explicitly specifies a project, both MySQL and Qdrant are
      narrowed to that project.
    - If no project is specified, both sources use the caller's authorized
      scope; all company projects are only allowed when ctx.all_projects=True.
    - Historical similarity is evidence, not proof of identical causality.
    - This skill does not read current Chat temporary attachments.
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
        self.analysis_skill = AnalysisSkill(registry)

    def answer(
        self,
        *,
        message: str,
        project_id: int | None,
        left_identifier: str | int,
        right_identifier: str | int,
        target_metric: str,
        direction_claim: str,
        ctx: UserContext,
    ) -> dict[str, Any]:
        db_ctx, knowledge_project_ids, knowledge_all_projects, scope = (
            self._resolve_analysis_scope(ctx=ctx, project_id=project_id)
        )

        database_result = self.analysis_skill.execute(
            "compare_samples",
            {
                "left_identifier": left_identifier,
                "right_identifier": right_identifier,
                "target_metric": target_metric,
                "direction": direction_claim,
            },
            db_ctx,
        )

        if not isinstance(database_result, dict) or database_result.get("status") != "ok":
            return {
                "status": "database_error",
                "answer": self._database_error_answer(database_result),
                "project_id": project_id,
                "analysis_scope": scope,
                "database_result": database_result,
                "knowledge_hits": [],
                "evidence": self._mysql_evidence(database_result),
                "warnings": list(
                    database_result.get("warnings", [])
                    if isinstance(database_result, dict)
                    else []
                ),
            }

        facts = database_result.get("facts") or {}
        left_sample = facts.get("left_sample") or {}
        right_sample = facts.get("right_sample") or {}

        # Defense in depth. Repositories already enforce company + project
        # permissions, but joint analysis verifies returned project membership
        # again before combining it with historical evidence.
        for sample in (left_sample, right_sample):
            raw_project = sample.get("project_id")
            if raw_project is None:
                raise PermissionError(
                    "联合分析返回的样品缺少 project_id，已拒绝继续历史资料检索"
                )
            sample_project = int(raw_project)
            if not ctx.can_access_project(sample_project):
                raise PermissionError(
                    "联合分析发现样品超出当前项目权限范围，已拒绝继续分析"
                )
            if project_id is not None and sample_project != int(project_id):
                raise PermissionError(
                    "联合分析发现样品项目与指定历史知识检索项目不一致，已拒绝继续分析"
                )

        knowledge_query = self._knowledge_query(
            target_metric=target_metric,
            direction_claim=direction_claim,
            message=message,
        )

        with self.repository_opener() as repo:
            hits = repo.search(
                query=knowledge_query,
                company_id=ctx.company_id,
                project_ids=knowledge_project_ids,
                all_projects=knowledge_all_projects,
                limit=self.max_hits,
                score_threshold=self.score_threshold,
            )

        knowledge_sources: list[str] = []
        knowledge_hit_summaries: list[dict[str, Any]] = []
        knowledge_evidence: list[dict[str, Any]] = []

        for rank, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            location = self._location(chunk)
            knowledge_sources.append(
                f"[HISTORY {rank}] "
                f"score={hit.score:.6f}；"
                f"文件={chunk.filename}；"
                f"project_id={chunk.project_id}；"
                f"source_id={chunk.source_id or '-'}；"
                f"{location}\n"
                f"{chunk.text}"
            )
            knowledge_hit_summaries.append(
                {
                    "score": hit.score,
                    "document_id": chunk.document_id,
                    "source_id": chunk.source_id,
                    "filename": chunk.filename,
                    "project_id": chunk.project_id,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page_number,
                    "paragraph_start": chunk.paragraph_start,
                    "paragraph_end": chunk.paragraph_end,
                    "text": chunk.text,
                }
            )
            knowledge_evidence.append(
                {
                    "evidence_type": "knowledge_index",
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

        mysql_evidence = self._mysql_evidence(database_result)
        combined_evidence = [*mysql_evidence, *knowledge_evidence]

        system = """你是“材数智能体”V0.1.2 联合分析器。
你将同时收到：
A. MYSQL FACTS：通过既有只读 Tool/Repository 从业务 MySQL 获取的结构化事实；
B. HISTORICAL KNOWLEDGE：从当前公司、当前用户授权范围内的 Qdrant 历史资料中检索出的证据。

必须遵守：
1. 先写【数据库事实】，只陈述 MYSQL FACTS 中明确存在的信息。
2. 再写【历史资料】，只陈述 HISTORICAL KNOWLEDGE 中明确存在的信息。
   历史资料可能来自当前公司的不同项目，必须保留每条来源自己的 project_id，不能把不同项目合并成同一次实验。
3. 再写【联合判断】。可以指出当前数据库事实与历史记录的相似点/差异点，
   但“历史相似”绝不等于“原因相同”，不得把相关性写成因果。
4. 再写【假设】。假设只能标注为需要验证的工程假设，不得伪装成数据库事实。
5. 再写【证据缺口】。必须保留数据库分析中的测试条件、控制实验、
   重复实验/统计稳定性等缺口；历史资料自身缺失的信息也要指出。
6. 最后写【结论边界】。明确当前证据能说明什么、不能说明什么。
7. 如果没有检索到足够相关历史资料，要明确写：
   “当前已索引历史资料中没有找到足够相关证据”，
   不能写成“历史上不存在”。
8. 不得补全来源中不存在的数值、单位、测试条件、原料批次或实验结论。
9. 不得复述密码、API Key、Token；[REDACTED] 必须保持为 [REDACTED]。
10. 最后给出【证据来源】，分别标识 MySQL 记录来源与历史文件来源；历史文件来源要写 project_id。
回答中文，结构清楚、保守、可审计。
"""

        mysql_payload = json.dumps(
            database_result,
            ensure_ascii=False,
            default=str,
        )
        history_payload = (
            "\n\n".join(knowledge_sources)
            if knowledge_sources
            else "NO_RELEVANT_HISTORY_HITS"
        )

        user = (
            f"用户问题：{message}\n"
            f"联合分析范围：{scope['display_name']}\n"
            f"历史检索查询：{knowledge_query}\n\n"
            f"MYSQL FACTS:\n{mysql_payload}\n\n"
            f"HISTORICAL KNOWLEDGE:\n{history_payload}"
        )

        answer = self.llm.complete(system, user)

        warnings = list(database_result.get("warnings", []))
        if not hits:
            warnings.append(
                f"当前已索引历史资料中没有相似度达到 "
                f"{self.score_threshold:.2f} 的可靠匹配"
            )

        return {
            "status": "ok",
            "analysis_type": "joint_mysql_historical_knowledge",
            "project_id": project_id,
            "analysis_scope": scope,
            "database_result": database_result,
            "knowledge_query": knowledge_query,
            "knowledge_hit_count": len(hits),
            "knowledge_hits": knowledge_hit_summaries,
            "score_threshold": self.score_threshold,
            "answer": answer,
            "evidence": combined_evidence,
            "warnings": warnings,
        }

    @staticmethod
    def _resolve_analysis_scope(
        *,
        ctx: UserContext,
        project_id: int | None,
    ) -> tuple[UserContext, list[int], bool, dict[str, Any]]:
        if project_id is not None:
            project_id = int(project_id)
            if not ctx.can_access_project(project_id):
                raise PermissionError("当前用户无权进行该项目的联合分析")

            scoped_ctx = UserContext(
                user_id=ctx.user_id,
                company_id=ctx.company_id,
                project_ids=(project_id,),
                permission_source=ctx.permission_source,
                all_projects=False,
            )
            return (
                scoped_ctx,
                [project_id],
                False,
                {
                    "mode": "explicit_project",
                    "company_id": ctx.company_id,
                    "project_ids": [project_id],
                    "display_name": f"当前公司 Project {project_id}",
                },
            )

        if ctx.all_projects:
            return (
                ctx,
                [],
                True,
                {
                    "mode": "company_all_projects",
                    "company_id": ctx.company_id,
                    "project_ids": "*",
                    "display_name": "当前公司全部项目",
                },
            )

        if ctx.project_ids:
            project_ids = sorted({int(item) for item in ctx.project_ids})
            return (
                ctx,
                project_ids,
                False,
                {
                    "mode": "authorized_projects",
                    "company_id": ctx.company_id,
                    "project_ids": project_ids,
                    "display_name": "当前用户已授权项目范围",
                },
            )

        raise PermissionError("当前用户没有可用于联合分析的项目权限")

    @staticmethod
    def _knowledge_query(
        *,
        target_metric: str,
        direction_claim: str,
        message: str,
    ) -> str:
        metric = str(target_metric or "").strip()
        direction = str(direction_claim or "").strip()
        if metric:
            return (
                f"历史上是否出现过与“{metric}{direction or '异常'}”类似的问题、"
                f"异常现象、排查记录或实验报告？"
            )
        return message

    @staticmethod
    def _mysql_evidence(database_result: Any) -> list[dict[str, Any]]:
        if not isinstance(database_result, dict):
            return []
        output: list[dict[str, Any]] = []
        for item in database_result.get("evidence", []) or []:
            if not isinstance(item, dict):
                continue
            output.append(
                {
                    **item,
                    "evidence_type": "mysql",
                }
            )
        return output

    @staticmethod
    def _database_error_answer(database_result: Any) -> str:
        if not isinstance(database_result, dict):
            return "数据库证据获取失败，因此未继续进行历史资料联合分析。"

        status = database_result.get("status")
        if status == "target_metric_not_found":
            available = (
                "、".join(database_result.get("available_changed_performance", []))
                or "无"
            )
            return (
                f"数据库对比中没有找到目标指标"
                f"“{database_result.get('target_metric')}”，"
                f"当前可用的变化性能指标：{available}。"
                "由于数据库事实不完整，本次未继续进行联合归因分析。"
            )

        if status == "left_error":
            return "左侧样品无法在当前授权范围内确定，未继续联合分析。"
        if status == "right_error":
            return "右侧样品无法在当前授权范围内确定，未继续联合分析。"

        return f"数据库证据获取失败（status={status}），未继续联合分析。"

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
            return f"段落 {chunk.paragraph_start}-{end} / chunk {chunk.chunk_index}"
        return f"chunk {chunk.chunk_index}"
