from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.evidence_frame import EvidenceFrameBuilder
from llm.base import LLMProvider
from runtime.chat_attachments import ChatAttachmentStore
from runtime.progress import emit_progress
from schemas.user_context import UserContext
from skills.material_intelligence import MaterialIntelligenceSkill


_EXPLICIT_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|\d{2,})"
    r"(?![A-Za-z0-9_.-])"
)


class HybridResearchQASkill:
    """Run one fixed cross-source research workflow.

    MySQL, vector, upload, and dialog inputs are normalized into EvidenceFrame
    before synthesis. The LLM never chooses SQL, never sees credentials, and
    cannot turn one source into evidence from another source.
    """

    def __init__(
        self,
        *,
        registry,
        llm: LLMProvider,
        material_intelligence: MaterialIntelligenceSkill,
        attachment_store: ChatAttachmentStore,
        vector_tool_name: str = "search_vector_knowledge",
        max_vector_hits: int = 5,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.material_intelligence = material_intelligence
        self.attachment_store = attachment_store
        self.vector_tool_name = vector_tool_name
        self.max_vector_hits = max(1, min(int(max_vector_hits), 50))

    def answer(
        self,
        *,
        message: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
        attachment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        args = dict(tool_args or {})
        scoped_ctx, project_ids, all_projects, scope = self._resolve_scope(
            args, ctx
        )
        query = str(args.get("history_query") or args.get("query") or message).strip()

        emit_progress(
            "hybrid_research_plan",
            "completed",
            "混合研究问答计划已确认",
            "将并行获取结构化事实、向量证据和当前输入，统一对齐后综合回答。",
            detail_items=[
                {"label": "权限范围", "value": scope["display_name"]},
                {"label": "结构化来源", "value": "外部 MySQL 只读 Tool"},
                {"label": "非结构化来源", "value": "向量检索 Tool"},
                {"label": "当前输入", "value": f"{len(attachment_ids or [])} 个附件"},
            ],
            plan_summary=(
                "所有来源先转换为 EvidenceFrame；执行过程只展示状态，最终报告统一生成。"
            ),
        )

        strategy = self._structured_strategy(args, message)
        emit_progress(
            "mysql_evidence_recall",
            "running",
            "读取结构化证据",
            f"正在通过 {strategy['tool_name']} 获取授权范围内的确定性事实。",
            detail_items=[
                {"label": "策略", "value": strategy["strategy"]},
                {"label": "只读工具", "value": strategy["tool_name"]},
            ],
        )
        emit_progress(
            "vector_evidence_recall",
            "running",
            "检索向量证据",
            f"正在并行检索 {scope['display_name']} 的知识片段。",
            query_preview=query,
        )
        vector_result: dict[str, Any]
        vector_warnings: list[str] = []
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-qa") as pool:
            mysql_future = pool.submit(strategy["executor"], scoped_ctx)
            vector_future = pool.submit(
                self.registry.execute,
                self.vector_tool_name,
                query=query,
                project_ids=project_ids,
                all_projects=all_projects,
                limit=self.max_vector_hits,
                ctx=scoped_ctx,
            )
            try:
                vector_result = vector_future.result()
            except Exception as exc:
                vector_result = {
                    "status": "vector_unavailable",
                    "hit_count": 0,
                    "hits": [],
                    "warnings": [f"向量检索不可用：{type(exc).__name__}"],
                }
                vector_warnings.append(
                    "向量证据源不可用，本轮仅能基于结构化事实和当前输入回答；不会用推测填补资料缺口。"
                )
                emit_progress(
                    "vector_evidence_recall",
                    "failed",
                    "向量证据不可用",
                    f"向量检索失败：{type(exc).__name__}。已保留证据缺口。",
                    error_type=type(exc).__name__,
                )
            else:
                emit_progress(
                    "vector_evidence_recall",
                    "completed",
                    "向量证据检索完成",
                    f"命中 {vector_result.get('hit_count', 0)} 个授权知识片段。",
                    hit_count=vector_result.get("hit_count", 0),
                )
            mysql_result = mysql_future.result()

        emit_progress(
            "mysql_evidence_recall",
            "completed",
            "结构化证据读取完成",
            f"结构化查询状态：{mysql_result.get('status', '-') if isinstance(mysql_result, dict) else '-'}。",
            result_status=str(mysql_result.get("status")) if isinstance(mysql_result, dict) else None,
        )

        attachments = []
        for attachment_id in attachment_ids or []:
            attachments.append(
                self.attachment_store.get(str(attachment_id), scoped_ctx)
            )

        builder = EvidenceFrameBuilder(ctx=scoped_ctx)
        builder.add_dialog_record(message)
        builder.add_mysql_result(mysql_result if isinstance(mysql_result, dict) else {})
        builder.add_vector_result(vector_result)
        builder.add_upload_records(attachments)
        frame = builder.build(warnings=vector_warnings)

        emit_progress(
            "evidence_alignment",
            "completed",
            "证据统一与对齐完成",
            (
                f"已合并 {len(frame.records)} 条证据，发现 "
                f"{len(frame.conflicts)} 组冲突。"
            ),
            evidence_count=len(frame.records),
            conflict_count=len(frame.conflicts),
            source_summary=frame.source_summary,
        )

        answer = self._synthesize(message=message, frame=frame)
        emit_progress(
            "hybrid_final_report",
            "completed",
            "混合研究报告已生成",
            "报告已按统一证据链完成综合分析。",
            evidence_count=len(frame.records),
        )

        mysql_warnings = (
            list(mysql_result.get("warnings") or [])
            if isinstance(mysql_result, dict)
            else []
        )
        warnings = [
            *vector_warnings,
            *list(vector_result.get("warnings") or []),
            *mysql_warnings,
        ]
        source_types = set(frame.source_summary)
        status = "ok" if frame.records else "no_evidence"
        if source_types and source_types <= {"dialog"}:
            status = "no_evidence"

        return {
            "status": status,
            "answer": answer,
            "analysis_type": "hybrid_research_qa",
            "query": query,
            "analysis_scope": scope,
            "structured_strategy": {
                "strategy": strategy["strategy"],
                "tool_name": strategy["tool_name"],
            },
            "mysql_result": mysql_result,
            "vector_result": vector_result,
            "evidence_frame": frame.model_dump(mode="json"),
            "vector_fact_extraction": {
                "status": "NOT_SUPPORTED",
                "reason": "本期不从向量文本抽取结构化建模事实；向量证据仅作为回答证据。",
            },
            "evidence_alignment": {
                "entity_resolver": "L1 normalized subject id",
                "unit_policy": "仅合并等价单位写法；不猜测物理换算",
                "conflict_resolver": "PRESERVE_ALL",
                "entity_link_count": len(frame.entity_links),
                "conflict_count": len(frame.conflicts),
            },
            "evidence": self._public_evidence(frame),
            "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        }

    def _structured_strategy(
        self,
        args: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        identifier = str(args.get("identifier") or "").strip()
        left = str(args.get("left_identifier") or "").strip()
        right = str(args.get("right_identifier") or "").strip()
        filters = args.get("filters")
        similarity_scope = str(args.get("similarity_scope") or "").strip()
        wants_similarity = (
            bool(similarity_scope)
            or any(
                marker in message
                for marker in ("相似配方", "类似配方", "相似样品", "类似样品", "相似实验", "类似实验")
            )
        )

        if not identifier:
            found = _EXPLICIT_IDENTIFIER.findall(message)
            identifier = str(found[-1]) if len(found) == 1 else ""
        if identifier and wants_similarity:
            similarity_args = {
                "identifier": identifier,
                "similarity_scope": similarity_scope or (
                    "formula"
                    if any(marker in message for marker in ("配方", "原料", "组分"))
                    else "process"
                    if any(marker in message for marker in ("工艺", "流程", "加工"))
                    else "combined"
                ),
                "top_n": int(args.get("top_n") or 5),
                "keyword": str(args.get("keyword") or ""),
            }
            return {
                "strategy": "structured_similarity",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self.material_intelligence.execute_intent(
                    "similar_samples",
                    "list_samples_for_analysis",
                    similarity_args,
                    ctx,
                ),
            }
        if identifier:
            return {
                "strategy": "explicit_sample_profile",
                "tool_name": "get_sample_context",
                "executor": lambda ctx: self.registry.execute(
                    "get_sample_context",
                    identifier=identifier,
                    ctx=ctx,
                ),
            }
        if left and right:
            return {
                "strategy": "explicit_pair_comparison",
                "tool_name": "compare_samples",
                "executor": lambda ctx: self.registry.execute(
                    "compare_samples",
                    left_identifier=left,
                    right_identifier=right,
                    ctx=ctx,
                ),
            }
        if isinstance(filters, list) and filters:
            return {
                "strategy": "structured_multi_condition_filter",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self.material_intelligence.execute_intent(
                    "find_samples_multi_condition",
                    "list_samples_for_analysis",
                    dict(args),
                    ctx,
                ),
            }
        keyword = str(args.get("keyword") or "").strip()
        if keyword:
            return {
                "strategy": "sample_keyword_recall",
                "tool_name": "find_samples",
                "executor": lambda ctx: self.registry.execute(
                    "find_samples",
                    keyword=keyword,
                    limit=20,
                    ctx=ctx,
                ),
            }
        return {
            "strategy": "bounded_authorized_sample_scan",
            "tool_name": "list_samples_for_analysis",
            "executor": lambda ctx: self.registry.execute(
                "list_samples_for_analysis",
                keyword="",
                limit=100,
                ctx=ctx,
            ),
        }

    def _synthesize(self, *, message: str, frame) -> str:
        payload = frame.model_dump(mode="json")
        system = """你是材数智能体的混合研究问答器。
输入的 EVIDENCE FRAME 已由后端完成来源归一、权限过滤、基础实体对齐和冲突标记。

要求：
1. 不要先分别写“数据库答案”和“RAG答案”，必须围绕用户问题给综合结论。
2. 每个事实性结论后用 record_id 标注证据，例如 [evf-xxx/record-id]。
3. MySQL、vector_api、upload、dialog 的证据分型不得混淆；对话输入只能作为本轮约束，不能当作实验事实。
4. 冲突必须保留差异和来源，不得静默选择一方。
5. 证据不足时明确写缺口；不得编造实验、性能、原料或历史结论。
6. 相关性不得写成因果。
7. 最终回答包含：结论、证据依据、风险/缺口。不要暴露内部表名、SQL、凭证或向量服务地址。
回答中文，结构简洁。
"""
        user = f"用户问题：{message}\n\nEVIDENCE FRAME:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
        return self.llm.complete(system, user)

    def _resolve_scope(
        self,
        args: dict[str, Any],
        ctx: UserContext,
    ) -> tuple[UserContext, list[int], bool, dict[str, Any]]:
        raw_project_id = args.get("project_id")
        if raw_project_id is None or str(raw_project_id).strip() == "":
            project_ids = (
                []
                if ctx.all_projects
                else sorted({int(item) for item in ctx.project_ids})
            )
            all_projects = bool(ctx.all_projects)
            display = (
                "当前公司全部项目"
                if all_projects
                else "当前用户已授权项目：" + (",".join(map(str, project_ids)) or "-")
            )
            return (
                ctx,
                project_ids,
                all_projects,
                {
                    "mode": "company_all_projects" if all_projects else "authorized_projects",
                    "company_id": ctx.company_id,
                    "project_ids": "*" if all_projects else project_ids,
                    "display_name": display,
                },
            )

        try:
            project_id = int(raw_project_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("混合研究问答的 project_id 必须是整数") from exc
        if not ctx.can_access_project(project_id):
            raise PermissionError("当前用户无权进行该项目的混合研究问答")
        scoped_ctx = UserContext(
            user_id=ctx.user_id,
            company_id=ctx.company_id,
            project_ids=(project_id,),
            permission_source=ctx.permission_source,
            all_projects=False,
            organization_id=ctx.organization_id,
            organization_level=ctx.organization_level,
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

    @staticmethod
    def _public_evidence(frame) -> list[dict[str, Any]]:
        output = []
        for record in frame.records:
            output.append(
                {
                    "record_id": record.record_id,
                    "source_type": record.source_type.value,
                    "subject_id": record.subject_id,
                    "entity_type": record.entity_type,
                    "attribute": record.attribute,
                    "source_uri": record.source_uri,
                    "confidence": record.confidence,
                    "authority_level": record.authority_level.value,
                    "conflict_group": record.conflict_group,
                    "alignment_level": record.alignment_level,
                }
            )
        return output
