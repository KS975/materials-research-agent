from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from agent.evidence_frame import EvidenceFrameBuilder
from agent.evidence_context import EvidenceContextCompressor
from agent.evidence_relation import build_source_relation
from agent.research_scenarios import resolve_research_scenario
from agent.research_slots import normalize_research_slots, parse_target_filters
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
        context_compressor: EvidenceContextCompressor | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.material_intelligence = material_intelligence
        self.attachment_store = attachment_store
        self.vector_tool_name = vector_tool_name
        self.max_vector_hits = max(1, min(int(max_vector_hits), 50))
        self.context_compressor = context_compressor or EvidenceContextCompressor()

    def answer(
        self,
        *,
        message: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
        attachment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        args = normalize_research_slots(
            message=message,
            args=dict(tool_args or {}),
        )
        scoped_ctx, project_ids, all_projects, scope = self._resolve_scope(
            args, ctx
        )
        query = str(args.get("history_query") or args.get("query") or message).strip()
        research_workflow = resolve_research_scenario(message, args)
        if research_workflow["execution_status"] != "SUPPORTED":
            raise ValueError(
                f"研究场景“{research_workflow['scenario_name']}”属于"
                f"{research_workflow['execution_status']}，当前切片未开放执行"
            )
        if research_workflow["scenario_id"] in {3, 14}:
            if not isinstance(args.get("filters"), list) or not args["filters"]:
                parsed_filters = parse_target_filters(message)
                if parsed_filters:
                    args["filters"] = parsed_filters
            elif args["filters"]:
                args["filters"] = [
                    {
                        **item,
                        "section": str(item.get("section") or "performance"),
                    }
                    for item in args["filters"]
                    if isinstance(item, dict)
                ]

        emit_progress(
            "hybrid_research_plan",
            "completed",
            "混合研究问答计划已确认",
            "将并行获取结构化事实、向量证据和当前输入，统一对齐后综合回答。",
            detail_items=[
                {"label": "权限范围", "value": scope["display_name"]},
                {"label": "研究 Workflow", "value": research_workflow["workflow_name"]},
                {"label": "场景", "value": research_workflow["scenario_name"]},
                {"label": "结构化来源", "value": "外部 MySQL 只读 Tool"},
                {"label": "非结构化来源", "value": "向量检索 Tool"},
                {"label": "当前输入", "value": f"{len(attachment_ids or [])} 个附件"},
            ],
            plan_summary=(
                "所有来源先转换为 EvidenceFrame；执行过程只展示状态，最终报告统一生成。"
            ),
        )

        strategy = self._structured_strategy(
            args, message, research_workflow["scenario_id"]
        )
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
            vector_context = copy_context()
            vector_future = pool.submit(
                vector_context.run,
                self.registry.execute,
                self.vector_tool_name,
                query=self._vector_query(query, args, research_workflow["scenario_id"]),
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

        scenario_warnings = self._scenario_boundary_warnings(
            research_workflow["scenario_id"],
            mysql_result if isinstance(mysql_result, dict) else {},
        )
        builder = EvidenceFrameBuilder(ctx=scoped_ctx)
        builder.add_dialog_record(message)
        builder.add_mysql_result(mysql_result if isinstance(mysql_result, dict) else {})
        builder.add_vector_result(vector_result)
        builder.add_upload_records(attachments)
        frame = builder.build(warnings=[*vector_warnings, *scenario_warnings])
        source_relation = build_source_relation(frame, vector_result)

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

        answer, synthesis = self._synthesize(
            message=message,
            frame=frame,
            source_relation=source_relation,
        )
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
        if synthesis["status"] == "degraded":
            warnings.append(
                "LLM 综合失败，已降级为后端确定性证据摘要；跨源综合结论需重新生成。"
            )
        if source_relation["level"] != "ENTITY_LINKED":
            warnings.append(source_relation["warning"])
        source_types = set(frame.source_summary)
        status = "ok" if frame.records else "no_evidence"
        if source_types and source_types <= {"dialog"}:
            status = "no_evidence"

        return {
            "status": status,
            "answer": answer,
            "synthesis": synthesis,
            "analysis_type": "hybrid_research_qa",
            "query": query,
            "resolved_tool_args": args,
            "research_workflow": research_workflow,
            "analysis_scope": scope,
            "structured_strategy": {
                "strategy": strategy["strategy"],
                "tool_name": strategy["tool_name"],
            },
            "mysql_result": mysql_result,
            "vector_result": vector_result,
            "evidence_frame": frame.model_dump(mode="json"),
            "source_relation": source_relation,
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
            "warnings": list(
                dict.fromkeys(
                    str(item)
                    for item in [*warnings, research_workflow["boundary"]]
                )
            ),
        }

    def _structured_strategy(
        self,
        args: dict[str, Any],
        message: str,
        scenario_id: int,
    ) -> dict[str, Any]:
        identifier = str(args.get("identifier") or "").strip()
        left = str(args.get("left_identifier") or "").strip()
        right = str(args.get("right_identifier") or "").strip()
        filters = args.get("filters")
        similarity_scope = str(args.get("similarity_scope") or "").strip()
        wants_similarity = (
            bool(similarity_scope)
            or (
                any(
                    marker in message
                    for marker in ("相似", "类似", "相近", "最像", "最接近", "接近")
                )
                and any(
                    marker in message
                    for marker in (
                        "配方", "组分", "原料", "样品", "实验", "工艺", "条件", "性能"
                    )
                )
            )
        )

        if not identifier:
            found = _EXPLICIT_IDENTIFIER.findall(message)
            identifier = str(found[-1]) if len(found) == 1 else ""
        if scenario_id == 5:
            return {
                "strategy": "material_usage_effect_scan",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self._material_usage_scan(
                    ctx, args, substitution=False
                ),
            }
        if scenario_id == 6:
            return {
                "strategy": "material_substitution_history_scan",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self._material_usage_scan(
                    ctx, args, substitution=True
                ),
            }
        target_filters = list(args.get("filters") or [])
        if target_filters:
            target_filters = [
                {
                    **item,
                    "section": str(item.get("section") or "performance"),
                }
                for item in target_filters
                if isinstance(item, dict)
            ]
        if not target_filters and scenario_id in {3, 14}:
            target_filters = parse_target_filters(message)
        if scenario_id in {3, 14} and target_filters:
            filter_args = {
                **dict(args),
                "filters": target_filters,
            }
            return {
                "strategy": "structured_multi_condition_filter",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self.material_intelligence.execute_intent(
                    "find_samples_multi_condition",
                    "list_samples_for_analysis",
                    filter_args,
                    ctx,
                ),
            }
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

    def _material_usage_scan(
        self,
        ctx: UserContext,
        args: dict[str, Any],
        *,
        substitution: bool,
    ) -> dict[str, Any]:
        if substitution:
            original = str(args.get("original_material") or "").strip()
            replacement = str(args.get("replacement_material") or "").strip()
            if not original or not replacement:
                raise ValueError(
                    "原料替代历史检索需要 original_material 和 replacement_material"
                )
        else:
            material_name = str(
                args.get("material_name") or args.get("keyword") or ""
            ).strip()
            if not material_name:
                raise ValueError("原料使用效果查询缺少 material_name")
        source = self.registry.execute(
            "list_samples_for_analysis",
            keyword="",
            limit=100,
            ctx=ctx,
        )
        if not substitution:
            matches = self._samples_using_material(source, material_name)
            matched_count = len(matches)
            try:
                result_limit = max(1, min(int(args.get("result_limit") or 50), 100))
            except (TypeError, ValueError):
                result_limit = 50
            truncated = matched_count > result_limit
            matches = matches[:result_limit]
            return {
                "status": "ok",
                "analysis_type": "material_usage_effect",
                "material_name": material_name,
                "count": len(matches),
                "matched_count": matched_count,
                "result_limit": result_limit,
                "truncated": truncated,
                "matched_samples": matches,
                "scan_scope": {
                    "sample_count": source.get("count"),
                    "total_matches": source.get("total_matches"),
                    "scan_complete": source.get("scan_complete"),
                },
                "evidence": [
                    {"source": "eln_sample", "record_id": item["sample"]["id"]}
                    for item in matches
                ],
                "warnings": [
                    *list(source.get("warnings") or []),
                    "原料命中基于授权样品的配方字段名称或原始键，不代表供应商或牌号完全相同。",
                    *(
                        [
                            f"原料使用效果命中 {matched_count} 条，超过返回上限 {result_limit}，已截断展示。"
                        ]
                        if truncated
                        else []
                    ),
                ],
            }

        original_rows = {
            str(item["sample"]["id"]): item
            for item in self._samples_using_material(source, original)
        }
        replacement_rows = {
            str(item["sample"]["id"]): item
            for item in self._samples_using_material(source, replacement)
        }
        both_ids = sorted(set(original_rows) & set(replacement_rows), key=int)
        records = []
        for sample_id in both_ids:
            records.append({
                "sample": original_rows[sample_id]["sample"],
                "original": original_rows[sample_id]["matched_fields"],
                "replacement": replacement_rows[sample_id]["matched_fields"],
                "performance": original_rows[sample_id].get("performance"),
                "interpretation": "CO_OCCURRENCE_ONLY",
            })
        return {
            "status": "ok",
            "analysis_type": "material_substitution_history",
            "original_material": original,
            "replacement_material": replacement,
            "count": len(records),
            "matched_samples": records,
            "scan_scope": {
                "sample_count": source.get("count"),
                "total_matches": source.get("total_matches"),
                "scan_complete": source.get("scan_complete"),
            },
            "evidence": [
                {"source": "eln_sample", "record_id": int(sample_id)}
                for sample_id in both_ids
            ],
            "warnings": [
                *list(source.get("warnings") or []),
                "同一样品同时含两种原料只能说明共存，不能自动证明发生过替代；替代结论需结合时间、项目记录和文档证据。",
            ],
        }

    @staticmethod
    def _samples_using_material(
        source: dict[str, Any], material_name: str
    ) -> list[dict[str, Any]]:
        needle = str(material_name or "").strip().casefold()
        if not needle:
            return []
        matches = []
        for row in source.get("samples") or []:
            fields = [
                item
                for item in row.get("formula") or []
                if needle in str(item.get("name") or "").casefold()
                or needle in str(item.get("raw_key") or "").casefold()
            ]
            if not fields:
                continue
            performance = [
                item
                for item in row.get("performance") or []
                if isinstance(item.get("value"), (int, float))
            ][:10]
            matches.append({
                "sample": row.get("sample") or {},
                "matched_fields": fields,
                "performance": performance,
            })
        return matches

    @staticmethod
    def _vector_query(
        default_query: str,
        args: dict[str, Any],
        scenario_id: int,
    ) -> str:
        candidates_by_scenario = {
            5: ("material_name", "keyword"),
            6: ("original_material", "replacement_material"),
            12: ("phenomenon", "keyword"),
            13: ("competitor_name", "keyword"),
        }
        parts: list[str] = []
        for key in candidates_by_scenario.get(scenario_id, ()):
            value = str(args.get(key) or "").strip()
            if value:
                parts.append(value)
        if parts:
            suffix = {5: "使用效果", 6: "替代历史", 12: "异常失效", 13: "竞品对标"}
            return " ".join([*parts, suffix.get(scenario_id, "")]).strip()
        return default_query

    @staticmethod
    def _scenario_boundary_warnings(
        scenario_id: int,
        mysql_result: dict[str, Any],
    ) -> list[str]:
        if scenario_id == 7:
            return [
                "当前结构化库没有通用“失败”标签；MySQL 仅提供授权样品上下文，"
                "失败结论必须由显式字段或文档证据支撑。"
            ]
        if scenario_id in {12, 13}:
            return [
                "现象/竞品与结构化样品之间没有专用实体映射；向量证据只作主题证据，"
                "不得直接视为某条 MySQL 记录。"
            ]
        return list(mysql_result.get("warnings") or [])

    def _synthesize(
        self,
        *,
        message: str,
        frame,
        source_relation: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        emit_progress(
            "llm_synthesis",
            "running",
            "生成混合研究综合报告",
            "正在基于压缩证据视图生成最终报告。",
        )
        system = """你是材数智能体的混合研究问答器。
输入的 COMPACT EVIDENCE CONTEXT 已由后端完成来源归一、权限过滤、基础实体对齐、冲突标记和有界压缩。

要求：
1. 不要先分别写“数据库答案”和“RAG答案”，必须围绕用户问题给综合结论。
2. 每个事实性结论后用 record_id 标注证据，例如 [evf-xxx/record-id]。
3. MySQL、vector_api、upload、dialog 的证据分型不得混淆；对话输入只能作为本轮约束，不能当作实验事实。
4. 冲突必须保留差异和来源，不得静默选择一方。
5. 证据不足时明确写缺口；不得编造实验、性能、原料或历史结论。
6. 相关性不得写成因果。
7. 最终回答包含：结论、证据依据、风险/缺口。不要暴露内部表名、SQL、凭证或向量服务地址。
8. selection.omitted_records 大于 0 时，不得推断被省略证据的内容。
9. source_relation.level 不是 ENTITY_LINKED 时，不得把向量文档写成同一样品、实验或配方的结构化事实。
回答中文，结构简洁。
"""
        attempts: list[dict[str, Any]] = []
        last_error_type = ""
        for aggressive in (False, True):
            context, serialized = self.context_compressor.build(
                frame=frame,
                query=message,
                aggressive=aggressive,
            )
            context["source_relation"] = source_relation
            serialized = json.dumps(
                context,
                ensure_ascii=False,
                default=str,
            )
            user = (
                f"用户问题：{message}\n\n"
                f"COMPACT EVIDENCE CONTEXT:\n{serialized}"
            )
            try:
                answer = self.llm.complete(system, user)
            except Exception as exc:
                last_error_type = type(exc).__name__
                attempts.append(
                    {
                        "status": "failed",
                        "aggressive_retry": aggressive,
                        "error_type": last_error_type,
                        "context_chars": len(user),
                        "selected_records": context["selection"]["selected_records"],
                    }
                )
                continue
            attempts.append(
                {
                    "status": "ok",
                    "aggressive_retry": aggressive,
                    "context_chars": len(user),
                    "selected_records": context["selection"]["selected_records"],
                }
            )
            emit_progress(
                "llm_synthesis",
                "completed",
                "LLM 综合报告已生成",
                "报告已基于有界证据上下文生成。",
                context_chars=len(user),
            )
            return answer, {
                "status": "ok",
                "mode": "llm_compact_context",
                "attempts": attempts,
                "context_selection": context["selection"],
            }

        answer = self._fallback_report(message=message, frame=frame)
        emit_progress(
            "llm_synthesis",
            "degraded",
            "LLM 综合未完成",
            f"模型综合两次未完成（{last_error_type}），已返回后端确定性证据摘要。",
            error_type=last_error_type,
        )
        return answer, {
            "status": "degraded",
            "mode": "deterministic_evidence_summary",
            "attempts": attempts,
            "error_type": last_error_type,
        }

    def _fallback_report(self, *, message: str, frame) -> str:
        lines = [
            "结论：LLM 综合未完成，本轮仅能给出后端保留下来的确定性证据摘要；不进行跨源推断。",
            f"研究问题：{message}",
            f"证据来源：{self._source_summary_text(frame)}",
        ]
        selected = self._fallback_records(frame)
        if selected:
            lines.append("代表性证据：")
            lines.extend(
                f"- [{record.record_id}] {record.subject_id} / "
                f"{record.attribute} = {record.value}"
                f"{f' {record.unit}' if record.unit else ''} "
                f"({record.source_type.value})"
                for record in selected
            )
        else:
            lines.append("代表性证据：无。")
        if frame.conflicts:
            lines.append(
                f"冲突证据：{len(frame.conflicts)} 组，"
                "已全部保留在 EvidenceFrame 中，未静默取舍。"
            )
        lines.append("缺口：LLM 综合失败，跨源结论需要重新生成；各来源证据可继续追溯。")
        return "\n".join(lines)

    @staticmethod
    def _fallback_records(frame):
        mysql = [
            record for record in frame.records if record.source_type.value == "mysql"
        ][:12]
        vectors = [
            record
            for record in frame.records
            if record.source_type.value == "vector_api"
        ][:5]
        uploads = [
            record for record in frame.records if record.source_type.value == "upload"
        ][:3]
        return [*mysql, *vectors, *uploads]

    @staticmethod
    def _source_summary_text(frame) -> str:
        if not frame.source_summary:
            return "无"
        return "，".join(
            f"{source}={count}" for source, count in frame.source_summary.items()
        )

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
