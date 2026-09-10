from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from agent.evidence_frame import EvidenceFrameBuilder
from agent.evidence_context import EvidenceContextCompressor
from agent.evidence_relation import build_source_relation
from agent.research_analysis import (
    research_analysis_chart_data,
    run_research_analysis,
)
from agent.scenario_aggregator import (
    aggregate_scenario_result,
    build_data_cards,
    build_dimension_status,
    serialize_aggregated_summary,
)
from agent.research_scenarios import resolve_research_scenario
from agent.research_slots import normalize_research_slots, parse_target_filters
from llm.base import LLMProvider
from runtime.chat_attachments import ChatAttachmentStore
from runtime.progress import emit_progress
from runtime.research_execution_audit import record_research_execution
from schemas.user_context import UserContext
from skills.material_intelligence import MaterialIntelligenceSkill


_EXPLICIT_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|\d{2,})"
    r"(?![A-Za-z0-9_.-])"
)


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            float(text)
            return True
        except (TypeError, ValueError):
            return False
    return False
_STAGE4_SCENARIOS = {8, 9, 10, 11, 18}
_DISPLAY_SCENARIOS = {15, 16, 20}
_INTERFACE_RESERVED_SCENARIOS = {19}
_EVIDENCE_RECORD_ID = re.compile(
    r"\b(?:mysql|vector|upload|dialog|derived)-[a-f0-9]{20}\b"
)
_INTERNAL_RECORD_REF = re.compile(
    r"\[\s*(?:mysql|vector|upload|dialog|derived|evf)-[a-f0-9]{8,}[^\]]*\]"
)
_INTERNAL_TABLE_NAMES = (
    "eln_sample",
    "sample_materials",
    "data_column",
    "mat_project",
    "eln_synthesis_exp",
    "eln_verify_item",
    "eln_verify_exp",
    "archive_data",
)
_SCENARIO_VECTOR_KEYWORDS: dict[int, tuple[str, ...]] = {
    7: ("失败", "配方", "实验"),
    8: ("关键变量", "影响因素"),
    9: ("工艺窗口", "稳定窗口", "稳定区间"),
    10: ("性能冲突", "冲突分析", "权衡"),
    11: ("批次差异",),
    15: ("检测结果", "关联"),
    16: ("图谱", "曲线", "DSC", "TGA", "粒径"),
    18: ("阶段总结", "阶段报告"),
    20: ("跨项目", "复用"),
}
logger = logging.getLogger(__name__)



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
        if research_workflow["scenario_id"] in _INTERFACE_RESERVED_SCENARIOS:
            return self._interface_reserved_result(
                message=message,
                args=args,
                research_workflow=research_workflow,
                scope=scope,
            )
        if research_workflow["execution_status"] not in {"SUPPORTED", "SUPPORTED_DISPLAY"}:
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
                    "知识资料源暂不可用，本轮仅能基于结构化事实和当前输入回答；"
                    "不会用推测填补资料缺口。"
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
        analysis_result: dict[str, Any] | None = None
        analysis_record_id: str | None = None
        aggregated_summary: dict[str, Any] | None = None
        aggregated_record_id: str | None = None
        if research_workflow["scenario_id"] in _STAGE4_SCENARIOS:
            emit_progress(
                "evidence_dataset_build",
                "completed",
                "Evidence Dataset 已构建",
                "已将授权样品的配方、工艺和性能字段转换为只读分析数据集。",
                sample_count=len(
                    (mysql_result.get("samples") or [])
                    if isinstance(mysql_result, dict)
                    else []
                ),
            )
            emit_progress(
                "deterministic_analysis",
                "running",
                "执行确定性分析",
                "正在计算变量、冲突、批次、窗口或阶段汇总。",
            )
            analysis_result = run_research_analysis(
                scenario_id=research_workflow["scenario_id"],
                source=mysql_result if isinstance(mysql_result, dict) else {},
                args=args,
                message=message,
            )
            analysis_record_id = builder.add_derived_result(analysis_result)
            emit_progress(
                "deterministic_analysis",
                "completed",
                "确定性分析完成",
                f"分析状态：{analysis_result.get('status', '-')}；结果已写入 derived 证据。",
                analysis_status=str(analysis_result.get("status")),
                chart_data=research_analysis_chart_data(analysis_result),
            )

        if research_workflow["scenario_id"] not in _INTERFACE_RESERVED_SCENARIOS:
            aggregated_summary = aggregate_scenario_result(
                scenario_id=research_workflow["scenario_id"],
                mysql_result=mysql_result if isinstance(mysql_result, dict) else {},
                vector_result=vector_result,
                analysis_result=analysis_result,
            )
            aggregated_summary.update(
                build_dimension_status(
                    aggregated_summary,
                    research_workflow.get("needed_dimensions"),
                    vector_result=vector_result,
                    upload_count=len(attachments),
                )
            )
            aggregated_record_id = builder.add_derived_result({
                "analysis_type": "scenario_aggregated_summary",
                "status": "ok",
                "scenario_id": research_workflow["scenario_id"],
                "summary": aggregated_summary,
                "warnings": [],
            })
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
            aggregated_summary=aggregated_summary,
            analysis_record_id=analysis_record_id,
            answer_format=str(research_workflow.get("answer_format") or "query"),
        )
        if research_workflow["scenario_id"] in _STAGE4_SCENARIOS:
            citation_validation = self._validate_citations(answer, frame)
            if not citation_validation["valid"]:
                answer = self._analysis_fallback_report(
                    result=analysis_result or {},
                    record_id=analysis_record_id or "",
                )
                citation_validation = self._validate_citations(answer, frame)
                synthesis = {
                    **synthesis,
                    "status": "deterministic",
                    "mode": "deterministic_analysis_report",
                    "citation_validation": citation_validation,
                }
            else:
                synthesis["citation_validation"] = citation_validation
        answer = self._sanitize_answer(
            answer,
            research_workflow["scenario_id"],
            missing_dimensions=(
                aggregated_summary.get("missing_dimension_labels")
                if isinstance(aggregated_summary, dict)
                else []
            ),
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
        if synthesis.get("mode") == "deterministic_analysis_report":
            warnings.append("LLM 报告缺少有效证据引用，已改用确定性分析报告。")
        if source_relation["level"] != "ENTITY_LINKED":
            warnings.append(source_relation["warning"])
        source_types = set(frame.source_summary)
        status = "ok" if frame.records else "no_evidence"
        if source_types and source_types <= {"dialog"}:
            status = "no_evidence"
        if analysis_result is not None:
            status = str(analysis_result.get("status") or status)

        missing_dimensions = (
            list(aggregated_summary.get("missing_dimensions") or [])
            if isinstance(aggregated_summary, dict)
            else []
        )
        record_research_execution(
            user_id=ctx.user_id,
            company_id=ctx.company_id,
            scenario_id=research_workflow["scenario_id"],
            workflow_id=research_workflow["workflow_id"],
            strategy=strategy["strategy"],
            status=status,
            evidence_count=len(frame.records),
            vector_hit_count=int(vector_result.get("hit_count") or 0),
            synthesis_status=str(synthesis.get("status") or ""),
            missing_dimensions=missing_dimensions,
        )
        logger.info(
            "hybrid_research_qa scenario=%s workflow=%s strategy=%s status=%s "
            "evidence=%s vector_hits=%s synthesis=%s missing_dimensions=%s",
            research_workflow["scenario_id"],
            research_workflow["workflow_id"],
            strategy["strategy"],
            status,
            len(frame.records),
            vector_result.get("hit_count", 0),
            synthesis.get("status"),
            ",".join(str(item) for item in missing_dimensions),
        )

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
            "analysis_result": analysis_result,
            "chart_data": research_analysis_chart_data(analysis_result or {}),
            "data_cards": build_data_cards(
                aggregated_summary,
                research_workflow["scenario_id"],
            ),
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
                    self._business_warning(item)
                    for item in [*warnings, research_workflow["boundary"]]
                    if self._business_warning(item)
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
        if scenario_id in _STAGE4_SCENARIOS:
            return {
                "strategy": "authorized_evidence_dataset_scan",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self.registry.execute(
                    "list_samples_for_analysis",
                    keyword=str(args.get("keyword") or ""),
                    limit=500,
                    ctx=ctx,
                ),
            }
        if scenario_id == 15:
            return {
                "strategy": "result_identifier_match",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self._result_association_scan(
                    ctx, args, message
                ),
            }
        if scenario_id == 16:
            return {
                "strategy": "structured_spectrum_feature_query",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self._spectrum_feature_scan(
                    ctx, args, message
                ),
            }
        if scenario_id == 20:
            return {
                "strategy": "cross_project_read_only_asset_query",
                "tool_name": "list_samples_for_analysis",
                "executor": lambda ctx: self._cross_project_asset_scan(
                    ctx, args, message
                ),
            }
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
                    else "performance"
                    if any(marker in message for marker in ("性能", "指标", "物性"))
                    else "combined"
                ),
                "top_n": int(args.get("top_n") or 10),
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
            material_name = self._clean_material_name(material_name)
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
    def _clean_material_name(raw: str) -> str:
        """Strip common query verbs from a material name extracted from a question."""
        name = str(raw or "").strip()
        for prefix in ("查", "查询", "查找", "看", "查看", "了解一下"):
            if name.startswith(prefix):
                candidate = name[len(prefix):].strip()
                if candidate:
                    return candidate
                break
        return name

    def _interface_reserved_result(
        self,
        *,
        message: str,
        args: dict[str, Any],
        research_workflow: dict[str, Any],
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "interface_reserved",
            "answer": (
                f"场景“{research_workflow['scenario_name']}”的正式闭环功能暂未开放，"
                "当前系统仅预留了接口位置。实验回流、Dataset 版本、Challenger 模型评估"
                "和模型晋级将在试点或交付阶段实现。当前操作不会修改任何数据或模型。"
            ),
            "synthesis": {
                "status": "not_implemented",
                "mode": "interface_reserved",
            },
            "analysis_type": "hybrid_research_qa",
            "query": message,
            "resolved_tool_args": args,
            "research_workflow": research_workflow,
            "analysis_scope": scope,
            "mysql_result": None,
            "vector_result": None,
            "analysis_result": None,
            "chart_data": None,
            "evidence_frame": None,
            "source_relation": None,
            "evidence": [],
            "warnings": [research_workflow["boundary"]],
        }

    def _result_association_scan(
        self,
        ctx: UserContext,
        args: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        identifiers = _EXPLICIT_IDENTIFIER.findall(message) or [
            str(args.get("identifier") or "")
        ]
        identifiers = [item for item in identifiers if item]
        if not identifiers:
            return {
                "status": "missing_identifier",
                "analysis_type": "result_association",
                "matches": [],
                "warnings": [
                    "未在当前输入中识别到样品、实验或配方编号；请提供编号以生成关联建议。"
                ],
            }
        matches = []
        ambiguity_warnings = []
        for identifier in identifiers[:3]:
            result = self.registry.execute(
                "list_samples_for_analysis",
                keyword=identifier,
                limit=10,
                ctx=ctx,
            )
            samples = list(result.get("samples") or result.get("results") or [])
            # Normalize nested list_samples_for_analysis rows to flat sample info.
            normalized = []
            for item in samples:
                sample = item.get("sample") if isinstance(item, dict) else None
                if isinstance(sample, dict):
                    normalized.append(sample)
                elif isinstance(item, dict):
                    normalized.append(item)
            samples = normalized
            if len(samples) == 1:
                sample = samples[0]
                matches.append({
                    "identifier": identifier,
                    "match_status": "UNIQUE_MATCH",
                    "matched_sample": {
                        "sample_id": sample.get("sample_id")
                        or sample.get("id")
                        or identifier,
                        "project_id": sample.get("project_id"),
                        "name": sample.get("name") or sample.get("sample_name"),
                    },
                    "confirmation_required": True,
                })
            elif len(samples) > 1:
                ambiguity_warnings.append(
                    f"编号 {identifier} 匹配到 {len(samples)} 个候选，需要用户确认。"
                )
                matches.append({
                    "identifier": identifier,
                    "match_status": "AMBIGUOUS",
                    "candidate_count": len(samples),
                    "candidates": [
                        {
                            "sample_id": item.get("sample_id") or item.get("id"),
                            "project_id": item.get("project_id"),
                            "name": item.get("name") or item.get("sample_name"),
                        }
                        for item in samples[:5]
                    ],
                    "confirmation_required": True,
                })
            else:
                ambiguity_warnings.append(
                    f"编号 {identifier} 在当前授权范围内未找到匹配样品。"
                )
                matches.append({
                    "identifier": identifier,
                    "match_status": "NO_MATCH",
                    "confirmation_required": False,
                })
        return {
            "status": "ok" if matches else "no_match",
            "analysis_type": "result_association",
            "matches": matches,
            "warnings": ambiguity_warnings,
            "write_policy": "NO_WRITE",
        }

    def _spectrum_feature_scan(
        self,
        ctx: UserContext,
        args: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        keyword = str(args.get("keyword") or "").strip()
        if not keyword:
            identifiers = _EXPLICIT_IDENTIFIER.findall(message)
            keyword = identifiers[-1] if identifiers else ""
        result = self.registry.execute(
            "list_samples_for_analysis",
            keyword=keyword,
            limit=50,
            ctx=ctx,
        )
        samples = list(result.get("samples") or [])
        spectrum_keywords = (
            "DSC", "TGA", "粒径", "谱图", "显微", "热重", "差示扫描",
            "熔点", "分解温度", "玻璃化转变",
        )
        matched_fields = []
        for sample in samples:
            for key, value in sample.items():
                if not isinstance(value, (int, float, str)):
                    continue
                if any(kw.lower() in str(key).lower() for kw in spectrum_keywords):
                    matched_fields.append({
                        "sample_id": sample.get("sample_id") or sample.get("id"),
                        "field": key,
                        "value": value,
                    })
        if matched_fields:
            return {
                "status": "ok",
                "analysis_type": "spectrum_feature_query",
                "feature_count": len(matched_fields),
                "features": matched_fields[:50],
                "warnings": [],
            }
        return {
            "status": "no_structured_features",
            "analysis_type": "spectrum_feature_query",
            "feature_count": 0,
            "features": [],
            "warnings": [
                "当前授权数据中未找到图谱/曲线的结构化特征值；"
                "展示版不支持图像识别，无法从图片文件自动提取特征。"
            ],
        }

    def _cross_project_asset_scan(
        self,
        ctx: UserContext,
        args: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        result = self.registry.execute(
            "list_samples_for_analysis",
            keyword="",
            limit=100,
            ctx=ctx,
        )
        samples = list(result.get("samples") or [])
        projects: dict[str, dict[str, Any]] = {}
        for sample in samples:
            project_id = str(
                sample.get("project_id") or sample.get("project") or "unknown"
            )
            if project_id not in projects:
                projects[project_id] = {
                    "project_id": project_id,
                    "sample_count": 0,
                    "assets": [],
                }
            projects[project_id]["sample_count"] += 1
        for project in projects.values():
            if project["sample_count"] > 0:
                project["assets"].append({
                    "asset_type": "historical_samples",
                    "count": project["sample_count"],
                    "access": "READ_ONLY",
                    "source": "external_mysql",
                })
        return {
            "status": "ok",
            "analysis_type": "cross_project_asset_query",
            "project_count": len(projects),
            "projects": list(projects.values()),
            "write_policy": "NO_WRITE",
            "warnings": [],
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
                if _is_numeric_value(item.get("value"))
            ][:10]
            matches.append({
                "sample": row.get("sample") or {},
                "formula": row.get("formula") or [],
                "matched_fields": fields,
                "performance": performance,
            })
        return matches

    @staticmethod
    def _validate_citations(answer: str, frame) -> dict[str, Any]:
        valid_ids = {record.record_id for record in frame.records}
        found = list(dict.fromkeys(_EVIDENCE_RECORD_ID.findall(str(answer or ""))))
        valid_found = [record_id for record_id in found if record_id in valid_ids]
        invalid_found = [record_id for record_id in found if record_id not in valid_ids]
        return {
            "valid": bool(valid_found) and not invalid_found,
            "valid_count": len(valid_found),
            "invalid_count": len(invalid_found),
            "invalid_ids": invalid_found,
            "policy": "阶段四结论必须至少引用一个真实 EvidenceFrame record_id。",
        }

    @staticmethod
    def _sanitize_answer(
        answer: str,
        scenario_id: int,
        missing_dimensions: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        """Remove internal process details from user-visible answers.

        Internal record ids, table names, execution-plan prefixes and SQL
        fragments never belong in the answer body. Scenario guardrails are
        appended when the model omitted them.
        """
        text = str(answer or "")
        text = _INTERNAL_RECORD_REF.sub("（见证据依据）", text)
        text = _EVIDENCE_RECORD_ID.sub("相关证据", text)
        for table in _INTERNAL_TABLE_NAMES:
            text = re.sub(rf"\b{re.escape(table)}\b", "业务数据表", text)
        text = re.sub(
            r"^(?:我需要|我将|我先|我打算|接下来我)[^。\n]*[。\n]+",
            "",
            text,
        )
        text = re.sub(
            r"SELECT\s+[^\n;]*?\s+FROM\s+[^\n;]+;?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = re.sub(
            r"(?ms)^\s*#{1,6}\s*证据依据\s*$.*?(?=^\s*#{1,6}\s+|\Z)",
            "",
            text,
        )
        text = re.sub(
            r"(?ms)^\s*\*\*证据依据\*\*\s*$.*?(?=\n\s*\n|\Z)",
            "",
            text,
        )
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        missing_labels = [
            str(item).strip()
            for item in (missing_dimensions or [])
            if str(item).strip()
        ]
        if missing_labels and not any(label in text for label in missing_labels):
            text = (
                f"数据提示：当前缺少{'、'.join(missing_labels)}，"
                "以下结论仅基于已有证据。\n\n"
                f"{text}"
            ).strip()
        if scenario_id in {8, 9, 10, 11} and "因果" not in text:
            text += (
                "\n\n> 注：以上为基于历史数据的相关性分析，"
                "不能据此直接判定因果关系。"
            )
        if scenario_id in {1, 2, 3, 4, 5, 6} and "单位" not in text:
            text += "\n\n> 注：配方和工艺数值为原始字段单位，未做物理换算。"
        return text

    @staticmethod
    def _analysis_fallback_report(
        *,
        result: dict[str, Any],
        record_id: str,
    ) -> str:
        analysis_type = str(result.get("analysis_type") or "research_analysis")
        dataset = result.get("dataset") or {}
        lines = [
            (
                f"结论：{analysis_type} 状态为 {result.get('status', '-')}；"
                f"分析范围包含 {dataset.get('sample_count', 0)} 条授权样品。[{record_id}]"
            ),
            (
                f"样本覆盖：读取 {dataset.get('sample_count', 0)} / "
                f"{dataset.get('total_matching_sample_count', dataset.get('sample_count', 0))} 条；"
                f"扫描完整={bool(dataset.get('scan_complete', True))}。"
            ),
        ]
        collection_keys = {
            "key_variable_analysis": "variables",
            "performance_conflict_analysis": "conflicts",
            "batch_difference_analysis": "differences",
            "process_window_discovery": "windows",
        }
        collection_key = collection_keys.get(analysis_type)
        rows = list(result.get(collection_key) or []) if collection_key else []
        if rows:
            lines.append("确定性结果（前 5 项）：")
            for row in rows[:5]:
                label = str(row.get("field_label") or row.get("field") or "-")
                if analysis_type == "key_variable_analysis":
                    detail = (
                        f"correlation={row.get('correlation')}, "
                        f"n={row.get('paired_sample_count')}"
                    )
                elif analysis_type == "performance_conflict_analysis":
                    detail = (
                        f"{row.get('left_target')} r={row.get('left_correlation')}; "
                        f"{row.get('right_target')} r={row.get('right_correlation')}"
                    )
                elif analysis_type == "batch_difference_analysis":
                    detail = (
                        f"{row.get('left_group')} mean={row.get('left_mean')}; "
                        f"{row.get('right_group')} mean={row.get('right_mean')}"
                    )
                else:
                    detail = (
                        f"recommended={row.get('recommended_min')}~"
                        f"{row.get('recommended_max')} {row.get('unit') or ''}, "
                        f"n={row.get('sample_count')}"
                    ).strip()
                lines.append(f"- {label}: {detail}")
        if analysis_type == "stage_report_analysis":
            lines.append("性能概览：")
            for row in (result.get("target_statistics") or [])[:5]:
                lines.append(
                    f"- {row.get('field_label')}: n={row.get('sample_count')}, "
                    f"mean={row.get('mean')} {row.get('unit') or ''}".rstrip()
                )
        if result.get("warnings"):
            lines.append(
                "限制：" + "；".join(str(item) for item in result["warnings"][:5])
            )
        if result.get("conclusion_limit"):
            lines.append(f"结论边界：{result['conclusion_limit']}")
        return "\n".join(lines)

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
        if scenario_id in {1, 2, 3, 4, 14}:
            identifier = str(args.get("identifier") or "")
            if not identifier:
                found = _EXPLICIT_IDENTIFIER.findall(default_query)
                identifier = found[-1] if len(found) == 1 else ""
            keyword = str(args.get("keyword") or "")
            filters = args.get("filters") or []
            filter_terms = [
                str(item.get("field") or item.get("section") or "")
                for item in filters
                if isinstance(item, dict)
            ]
            parts = [identifier, keyword, *filter_terms]
            return " ".join(p for p in parts if p).strip() or default_query
        if scenario_id == 17:
            keyword = str(args.get("keyword") or args.get("project_name") or "")
            return keyword if keyword else default_query
        keywords = _SCENARIO_VECTOR_KEYWORDS.get(scenario_id)
        if keywords is not None:
            identifier = str(args.get("identifier") or "")
            if not identifier:
                found = _EXPLICIT_IDENTIFIER.findall(default_query)
                identifier = found[-1] if len(found) == 1 else ""
            keyword = str(args.get("keyword") or "")
            matched = [item for item in keywords if item in default_query]
            filter_terms = [
                str(item.get("field") or item.get("section") or "")
                for item in (args.get("filters") or [])
                if isinstance(item, dict)
            ]
            parts = [identifier, keyword, *matched, *filter_terms]
            query = " ".join(part for part in parts if part).strip()
            # Fall back to the canonical scenario keywords instead of the raw
            # question so long sentences do not dilute vector similarity.
            return query or " ".join(keywords)
        return default_query

    @staticmethod
    def _business_warning(value: Any) -> str:
        text = str(value or "").strip()
        replacements = (
            ("未解析动态字段", "部分字段未完全登记，可能存在数据遗漏"),
            ("动态字段", "扩展字段"),
            ("向量库", "知识库"),
            ("向量证据", "知识片段"),
            ("向量片段", "知识片段"),
            ("companyId", "授权范围"),
            ("API", "服务"),
            ("workflow", "处理流程"),
            ("EvidenceFrame", "证据链"),
        )
        for source, target in replacements:
            text = text.replace(source, target)
        return text

    @staticmethod
    def _scenario_boundary_warnings(
        scenario_id: int,
        mysql_result: dict[str, Any],
    ) -> list[str]:
        if scenario_id == 7:
            return [
                "当前数据没有统一的“失败”标签，系统仅依据明确失败状态或历史资料"
                "判断，不能把普通历史实验直接视为失败。"
            ]
        if scenario_id in {12, 13}:
            return [
                "现象或竞品信息目前只能做主题匹配，不能直接认定对应某个样品或实验。"
            ]
        if scenario_id == 15:
            return [
                "展示版仅生成检测结果关联建议，不写入业务数据库；"
                "歧义或多候选时必须由用户确认后才能形成正式关联。",
                *list(mysql_result.get("warnings") or []),
            ]
        if scenario_id == 16:
            return [
                "展示版仅查询已有结构化图谱/曲线特征值，不做图像识别或 OCR；"
                "无结构化特征时明确降级，不从图片推测数值。",
                *list(mysql_result.get("warnings") or []),
            ]
        if scenario_id == 20:
            return [
                "跨项目资产仅做当前权限范围内只读查询，不做资产迁移、授权变更或写入。",
                *list(mysql_result.get("warnings") or []),
            ]
        return list(mysql_result.get("warnings") or [])

    def _synthesize(
        self,
        *,
        message: str,
        frame,
        source_relation: dict[str, Any],
        aggregated_summary: dict[str, Any] | None = None,
        analysis_record_id: str | None = None,
        answer_format: str = "query",
    ) -> tuple[str, dict[str, Any]]:
        emit_progress(
            "llm_synthesis",
            "running",
            "生成混合研究综合报告",
            "正在基于场景化结构化摘要生成最终报告。",
        )
        format_instruction = {
            "query": (
                "查询类回答：先给1-2句直接结论和可执行推荐；不要重复罗列卡片中的"
                "全部数据；最后给出必要的风险或提示。"
            ),
            "analysis": (
                "分析类回答：先给明确结论，再引用支撑结论的关键数据、计算范围"
                "和证据引用；明确区分相关性与因果关系；最后给出风险或缺口。"
            ),
            "plan": (
                "方案类回答：先给目标判断，再给出按优先级排列的行动方案、所需"
                "数据或验证步骤；不得把候选方案写成已验证事实；最后给出限制条件。"
            ),
        }.get(answer_format, "回答应简洁、证据可追溯，并明确说明数据缺口。")
        system = f"""你是材数智能体的混合研究问答器。
输入的 STRUCTURED EVIDENCE SUMMARY 已由后端完成来源归一、权限过滤、基础实体对齐和场景化聚合。

要求：
1. 不要先分别写“数据库答案”和“RAG答案”，必须围绕用户问题给综合结论。
2. {format_instruction}
3. 仅当 answer_format=analysis 时，结论必须引用 citation_anchors 中的 record_id。
4. 结构化摘要与向量证据分型不得混淆；对话输入只能作为本轮约束，不能当作实验事实。
5. 冲突必须保留差异和来源，不得静默选择一方。
6. 证据不足时明确写缺口；不得编造实验、性能、原料或历史结论。
7. 相关性不得写成因果。
8. 不要单独写“证据依据”段，也不要重复罗列系统卡片已展示的数据；正文统一为“结论/方案 + 风险或缺口”。
9. source_relation.level 不是 ENTITY_LINKED 时，不得把向量文档写成同一样品、实验或配方的结构化事实。
10. structured_summary 中的数值已按业务精度归一化，必须原样引用，不得再次修改、舍入或估算。
11. structured_summary 中的数值是原始字段单位，未做物理换算。
12. 如果 structured_summary 含 missing_dimension_labels，结论开头必须明确说明缺了什么业务维度，并把结论限定在已有证据范围内。
回答中文，结构简洁。
"""
        summary = aggregated_summary if isinstance(aggregated_summary, dict) else {}
        degrade_level = summary.get("degrade_level")
        missing_fields = summary.get("missing_fields") or []
        missing_dimension_labels = summary.get("missing_dimension_labels") or []
        context = {
            "schema_version": 2,
            "answer_format": answer_format,
            "structured_summary": summary,
            "degrade_level": degrade_level,
            "missing_fields": missing_fields,
            "missing_dimension_labels": missing_dimension_labels,
            "source_relation": source_relation,
            "source_summary": frame.source_summary,
            "conflict_count": len(frame.conflicts),
            "warnings": [str(item) for item in frame.warnings[:8]],
            "citation_anchors": (
                [{"record_id": analysis_record_id, "kind": "derived_analysis"}]
                if analysis_record_id
                else []
            ),
            "evidence_record_count": len(frame.records),
        }
        attempts: list[dict[str, Any]] = []
        last_error_type = ""
        for aggressive in (False, True):
            serialized = self.context_compressor.bound_context(
                context,
                aggressive=aggressive,
            )
            user = (
                f"用户问题：{message}\n\n"
                f"STRUCTURED EVIDENCE SUMMARY:\n{serialized}"
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
                    }
                )
                continue
            attempts.append(
                {
                    "status": "ok",
                    "aggressive_retry": aggressive,
                    "context_chars": len(user),
                }
            )
            emit_progress(
                "llm_synthesis",
                "completed",
                "LLM 综合报告已生成",
                "报告已基于场景化结构化摘要生成。",
                context_chars=len(user),
            )
            return answer, {
                "status": "ok",
                "mode": "llm_structured_summary",
                "attempts": attempts,
                "context_chars": len(user),
                "summary_truncated": serialized.endswith("...[truncated]"),
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
