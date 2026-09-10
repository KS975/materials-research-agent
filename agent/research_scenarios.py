from __future__ import annotations

import re
from typing import Any, Mapping


RESEARCH_WORKFLOW_VERSION = "research-workflow-v1"

RESEARCH_WORKFLOWS: dict[str, dict[str, Any]] = {
    "hybrid_search_rank": {
        "display_name": "混合检索与排序",
        "scenario_ids": [1, 2, 3, 5, 6, 7, 12, 13],
        "execution_status": "SUPPORTED",
        "steps": [
            "resolve_permission_scope",
            "parallel_mysql_vector_recall",
            "deterministic_filter_or_similarity",
            "build_evidence_frame",
            "rank_candidates",
            "final_report",
        ],
    },
    "evidence_profile": {
        "display_name": "证据联查与画像",
        "scenario_ids": [4, 17],
        "execution_status": "SUPPORTED",
        "steps": [
            "resolve_permission_scope",
            "parallel_mysql_vector_recall",
            "entity_alignment",
            "build_evidence_frame",
            "profile_or_project_answer",
        ],
    },
    "research_cold_start": {
        "display_name": "新项目冷启动",
        "scenario_ids": [14],
        "execution_status": "SUPPORTED",
        "steps": [
            "parse_target_and_constraints",
            "parallel_mysql_vector_recall",
            "retrieve_similar_history_and_failures",
            "build_evidence_frame",
            "compose_first_round_options",
        ],
    },
    "result_feature_ingestion": {
        "display_name": "结果自动关联与特征接入",
        "scenario_ids": [15, 16],
        "execution_status": "SUPPORTED_DISPLAY",
        "steps": [
            "parse_result_or_spectrum_request",
            "match_sample_experiment_formula",
            "query_existing_structured_features",
            "generate_pending_confirmation_or_degradation",
        ],
    },
    "historical_analysis": {
        "display_name": "历史数据分析",
        "scenario_ids": [8, 10, 11],
        "execution_status": "SUPPORTED",
        "steps": [
            "build_evidence_dataset",
            "key_variable_or_conflict_analysis",
            "batch_difference_analysis",
            "trace_conclusion_to_samples",
        ],
    },
    "process_window": {
        "display_name": "配方/工艺窗口发现",
        "scenario_ids": [9],
        "execution_status": "SUPPORTED",
        "steps": [
            "build_evidence_dataset",
            "filter_feasible_region",
            "robustness_analysis",
            "output_stable_window",
        ],
    },
    "stage_report": {
        "display_name": "阶段总结生成",
        "scenario_ids": [18],
        "execution_status": "SUPPORTED",
        "steps": [
            "define_project_time_scope",
            "aggregate_experiments_and_results",
            "generate_outline",
            "validate_citations",
            "final_report",
        ],
    },
    "experiment_loopback": {
        "display_name": "实验回流接口预留",
        "scenario_ids": [19],
        "execution_status": "INTERFACE_RESERVED",
        "steps": [
            "route_to_reserved_interface",
            "return_not_yet_available",
        ],
    },
    "cross_project_query": {
        "display_name": "跨项目资产只读查询",
        "scenario_ids": [20],
        "execution_status": "SUPPORTED_DISPLAY",
        "steps": [
            "resolve_permission_scope",
            "list_authorized_projects_and_assets",
            "filter_by_target_project",
            "return_read_only_summary",
        ],
    },
}

_SCENARIO_NAMES_BY_ID = {
    1: "相似配方检索",
    2: "相似样品/相似实验检索",
    3: "按目标性能反查历史方案",
    4: "配方-工艺-性能联查",
    5: "原料使用效果查询",
    6: "原料替代历史检索",
    7: "失败配方/失败实验检索",
    8: "关键变量识别",
    9: "配方/工艺窗口发现",
    10: "多性能冲突分析",
    11: "批次差异分析",
    12: "异常与失效案例检索",
    13: "竞品对标",
    14: "新项目冷启动",
    15: "测试结果自动关联样品",
    16: "图谱/曲线结果复用",
    17: "项目知识快速问答",
    18: "自动生成阶段总结",
    19: "模型版本与实验回流",
    20: "跨项目复用",
}

SCENARIO_NAMES: dict[int, str] = {
    scenario_id: _SCENARIO_NAMES_BY_ID[scenario_id]
    for workflow in RESEARCH_WORKFLOWS.values()
    for scenario_id in workflow["scenario_ids"]
}

_REQUIRED_DIMENSIONS_BY_SCENARIO: dict[int, tuple[str, ...]] = {
    1: ("formula", "documents"),
    2: ("formula", "process", "performance", "documents"),
    3: ("performance", "formula"),
    4: ("formula", "process", "performance", "documents"),
    5: ("formula", "performance", "documents"),
    6: ("formula", "performance", "documents"),
    7: ("formula", "performance", "status", "documents"),
    8: ("dataset", "formula", "process", "performance"),
    9: ("dataset", "formula", "process", "performance"),
    10: ("dataset", "formula", "process", "performance"),
    11: ("dataset", "formula", "process", "performance"),
    12: ("sample", "documents"),
    13: ("sample", "performance", "documents"),
    14: ("formula", "process", "performance", "documents"),
    15: ("result_association", "sample"),
    16: ("spectrum", "sample"),
    17: ("project", "documents"),
    18: ("project", "dataset", "performance", "documents"),
    19: ("dataset", "model", "performance"),
    20: ("project", "assets"),
}

_EXPLICIT_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|\d{2,})"
    r"(?![A-Za-z0-9_.-])"
)


def resolve_research_scenario(
    message: str,
    args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a research request into the fixed eight-workflow taxonomy.

    The classifier is deterministic backend code. The LLM may extract slots,
    but it cannot invent a workflow id or bypass a stage gate.
    """
    payload = dict(args or {})
    text = str(message or "").strip()
    scenario_id = _requested_scenario_id(payload)
    if scenario_id is None:
        scenario_id = _classify_message(text, payload)

    workflow_id = _workflow_id_for_scenario(scenario_id)
    workflow = RESEARCH_WORKFLOWS[workflow_id]
    needed_dimensions = _needed_dimensions(scenario_id, text)
    return {
        "schema_version": RESEARCH_WORKFLOW_VERSION,
        "workflow_id": workflow_id,
        "workflow_name": workflow["display_name"],
        "scenario_id": scenario_id,
        "scenario_name": SCENARIO_NAMES[scenario_id],
        "execution_status": workflow["execution_status"],
        "covered_scenario_ids": list(workflow["scenario_ids"]),
        "steps": list(workflow["steps"]),
        "needed_dimensions": needed_dimensions,
        "answer_format": _answer_format(scenario_id, text),
        "read_only": workflow_id in {
            "hybrid_search_rank",
            "evidence_profile",
            "research_cold_start",
            "historical_analysis",
            "process_window",
            "stage_report",
            "result_feature_ingestion",
            "experiment_loopback",
            "cross_project_query",
        },
        "write_policy": "NO_WRITE" if workflow_id in {
            "hybrid_search_rank",
            "evidence_profile",
            "research_cold_start",
            "historical_analysis",
            "process_window",
            "stage_report",
            "result_feature_ingestion",
            "experiment_loopback",
            "cross_project_query",
        } else "HUMAN_APPROVAL_REQUIRED",
        "boundary": _boundary_for(workflow_id),
    }


def _needed_dimensions(scenario_id: int, text: str) -> list[str]:
    if scenario_id == 2:
        dimensions = []
        if "配方" in text or "原料" in text or "组分" in text:
            dimensions.append("formula")
        if "工艺" in text or "流程" in text or "条件" in text:
            dimensions.append("process")
        if "性能" in text or "指标" in text or "物性" in text:
            dimensions.append("performance")
        if not dimensions:
            dimensions.extend(("formula", "process", "performance"))
        dimensions.append("documents")
        return _dedupe(dimensions)
    dimensions = list(_REQUIRED_DIMENSIONS_BY_SCENARIO.get(scenario_id, ()))
    if scenario_id in {1, 2, 4, 5, 6, 7, 12, 13, 14}:
        if "工艺" in text or "流程" in text or "条件" in text:
            dimensions = _append_dimension(dimensions, "process")
        if "性能" in text or "指标" in text or "物性" in text:
            dimensions = _append_dimension(dimensions, "performance")
        if "配方" in text or "原料" in text or "组分" in text:
            dimensions = _append_dimension(dimensions, "formula")
    if scenario_id in {8, 10}:
        for metric in _extract_metric_terms(text):
            if any(
                marker in metric
                for marker in ("温度", "时间", "压力", "转速", "速度", "工序", "工艺")
            ):
                dimensions = _append_dimension(dimensions, "process")
            elif any(
                marker in metric
                for marker in ("含量", "用量", "添加", "配比", "组分", "原料")
            ):
                dimensions = _append_dimension(dimensions, "formula")
            else:
                dimensions = _append_dimension(dimensions, "performance")
    return _dedupe(dimensions)


def _answer_format(scenario_id: int, text: str) -> str:
    if scenario_id in {8, 9, 10, 11, 17}:
        return "analysis"
    if scenario_id in {14, 18, 19}:
        return "plan"
    if any(
        marker in text
        for marker in ("为什么", "原因", "冲突", "权衡", "关键变量", "差异分析", "综合判断")
    ):
        return "analysis"
    return "query"


def _extract_metric_terms(text: str) -> list[str]:
    return [
        item
        for item in re.split(r"[，。；;、\s]+", text)
        if item
    ]


def _append_dimension(values: list[str], dimension: str) -> list[str]:
    if dimension not in values:
        values.append(dimension)
    return values


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def looks_like_hybrid_research_request(message: str) -> bool:
    """Recognize stage-3 questions that require joined source reasoning."""
    text = str(message or "").strip()
    if _looks_like_similarity_phrase(text):
        return True
    if _looks_like_failure_request(text):
        return True
    if _looks_like_material_usage_request(text):
        return True
    if _looks_like_material_substitution_request(text):
        return True
    if _looks_like_target_filter_request(text):
        return True
    if _looks_like_stage4_analysis_request(text):
        return True
    markers = (
        "历史资料", "历史案例", "知识库", "向量库", "结合历史", "综合历史",
        "相似配方", "相似的配方", "类似配方", "类似的配方", "相近配方", "相近的配方",
        "相似样品", "相似的样品", "类似样品", "类似的样品", "相近样品", "相近的样品",
        "相似实验", "相似的实验", "类似实验", "类似的实验", "相近实验", "相近的实验",
        "原料使用效果", "原料替代", "替代历史", "失败配方", "失败实验",
        "异常案例", "失效案例", "竞品对标", "新项目冷启动", "项目知识问答",
        "去年做过", "为什么停用",
        "实验回流", "模型版本", "模型更新", "Challenger",
        "检测结果", "测试结果", "LIMS",
        "图谱", "曲线", "DSC", "TGA", "粒径", "谱图", "显微图",
        "跨项目复用", "跨项目", "其他项目复用",
    )
    return any(marker in text for marker in markers)


def _requested_scenario_id(args: Mapping[str, Any]) -> int | None:
    raw = args.get("scenario_id")
    try:
        scenario_id = int(raw)
    except (TypeError, ValueError):
        return None
    return scenario_id if scenario_id in SCENARIO_NAMES else None


def _classify_message(text: str, args: Mapping[str, Any]) -> int:
    if _looks_like_stage4_analysis_request(text):
        return _stage4_scenario(text)
    if _looks_like_experiment_loopback_request(text):
        return 19
    if _looks_like_result_association_request(text):
        return 15
    if _looks_like_spectrum_curve_request(text):
        return 16
    if _looks_like_cross_project_request(text):
        return 20
    if any(marker in text for marker in ("新项目冷启动", "新项目立项", "首轮方案", "从零开始")):
        return 14
    if "为什么" in text and "停用" in text:
        return 17
    if _looks_like_material_substitution_request(text) or (
        args.get("original_material")
        and args.get("replacement_material")
    ):
        return 6
    if args.get("material_name") or _looks_like_material_usage_request(text):
        return 5
    if _looks_like_failure_request(text):
        return 7
    if any(
        marker in text
        for marker in ("异常案例", "失效案例", "开裂", "析出", "变色", "老化", "粘接失效")
    ):
        return 12
    if any(marker in text for marker in ("竞品", "对标")):
        return 13
    if _looks_like_target_filter_request(text):
        return 3
    if isinstance(args.get("filters"), list) and args["filters"]:
        return 3
    if _looks_like_similarity_phrase(text):
        return _similarity_target(text)
    if _EXPLICIT_IDENTIFIER.search(text) or args.get("identifier"):
        return 4
    if any(marker in text for marker in ("项目知识", "项目问答", "为什么停用", "去年")):
        return 17
    return 1


def _looks_like_experiment_loopback_request(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "实验回流", "结果回流", "数据回流", "模型版本", "模型更新",
            "重新训练", "重训模型", "Challenger", "挑战者模型", "模型晋级",
        )
    )


def _looks_like_result_association_request(text: str) -> bool:
    if any(
        marker in text
        for marker in (
            "检测结果自动关联", "测试结果自动关联", "LIMS结果关联",
            "LIMS结果回样品", "检测结果错配", "结果自动关联",
        )
    ):
        return True
    return ("检测结果" in text or "测试结果" in text) and any(
        marker in text for marker in ("关联", "匹配", "对应样品", "对应实验")
    )


def _looks_like_spectrum_curve_request(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "图谱", "曲线", "DSC", "TGA", "粒径分布", "谱图",
            "显微图", "差示扫描", "热重分析",
        )
    )


def _looks_like_cross_project_request(text: str) -> bool:
    return any(
        marker in text
        for marker in ("跨项目复用", "跨项目", "其他项目复用", "复用资产", "资产复用")
    )


def _looks_like_similarity_phrase(text: str) -> bool:
    similarity_markers = ("相似", "类似", "相近", "最像", "最接近", "接近")
    domain_markers = (
        "配方", "组分", "组成", "原料", "样品", "样本", "实验",
        "工艺", "流程", "加工", "条件", "性能",
    )
    return any(marker in text for marker in similarity_markers) and any(
        marker in text for marker in domain_markers
    )


def _looks_like_failure_request(text: str) -> bool:
    return "失败" in text and any(
        marker in text for marker in ("配方", "实验", "路线", "案例", "调整")
    )


def _looks_like_material_usage_request(text: str) -> bool:
    if any(
        marker in text
        for marker in ("使用效果", "用在哪些", "用过哪些", "用量多少")
    ):
        return True
    return "使用" in text and "样品" in text and any(
        marker in text for marker in ("性能", "怎么样", "效果")
    )


def _looks_like_material_substitution_request(text: str) -> bool:
    if "替代" not in text and "替换" not in text:
        return False
    return any(
        marker in text for marker in ("原料", "材料", "牌号", "供应商")
    ) or bool(re.search(r"[A-Za-z0-9][^，。？?]{0,30}(?:替代|替换)", text))


def _looks_like_target_filter_request(text: str) -> bool:
    operators = (
        "大于等于", "小于等于", "大于", "小于", "高于", "低于",
        "不低于", "不高于", "至少", "至多", "介于", ">=", "<=", ">", "<",
    )
    collections = ("样品", "样本", "实验", "方案", "配方", "工艺")
    return any(operator in text for operator in operators) and any(
        marker in text for marker in collections
    )


def _looks_like_stage4_analysis_request(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "关键变量", "影响因素", "影响最大", "多性能冲突", "性能冲突",
            "冲突分析", "为什么冲突", "怎么冲突", "权衡", "哪些变量", "什么变量", "变量影响",
            "批次差异", "正常批次", "异常批次",
            "工艺窗口", "配方窗口", "稳定窗口", "稳定区间", "阶段总结",
            "阶段报告", "研发阶段报告", "生成报告",
        )
    )


def _stage4_scenario(text: str) -> int:
    if any(marker in text for marker in ("阶段总结", "阶段报告", "研发阶段报告", "生成报告")):
        return 18
    if any(marker in text for marker in ("工艺窗口", "配方窗口", "稳定窗口", "稳定区间")):
        return 9
    if any(marker in text for marker in ("批次差异", "正常批次", "异常批次")):
        return 11
    if any(
        marker in text
        for marker in (
            "多性能冲突", "性能冲突", "冲突分析", "为什么冲突", "怎么冲突",
            "权衡", "为什么下降", "此消彼长",
        )
    ):
        return 10
    return 8


def _similarity_target(text: str) -> int:
    # A formula search is scenario 1 even when the returned rows are samples.
    # Process, experiment, and combined condition searches are scenario 2.
    if "性能" in text:
        return 2
    if any(marker in text for marker in ("配方", "组分", "组成", "原料")):
        return 1
    return 2


def _workflow_id_for_scenario(scenario_id: int) -> str:
    for workflow_id, workflow in RESEARCH_WORKFLOWS.items():
        if scenario_id in workflow["scenario_ids"]:
            return workflow_id
    return "hybrid_search_rank"


def _boundary_for(workflow_id: str) -> str:
    return {
        "hybrid_search_rank": (
            "候选只表示历史证据匹配，不是实验验证结论；结构化筛选与向量相似度必须分别溯源。"
        ),
        "evidence_profile": "画像结论仅覆盖当前授权项目和返回证据，不推断未记录实验。",
        "research_cold_start": "首轮方案仅用于研发起点，正式实验前必须经过可行性与安全审查。",
        "result_feature_ingestion": "展示版仅生成关联建议或查询已有结构化特征，不写入业务数据库；歧义时需用户确认。",
        "historical_analysis": "统计结论必须绑定样本范围、字段和单位；相关性不得写成因果。",
        "process_window": "窗口结论必须同时给出可行区间、支撑样本和稳健性，不外推历史可行域。",
        "stage_report": "报告必须绑定授权范围和证据记录；不得把推测写成已验证结论。",
        "experiment_loopback": "实验回流正式闭环暂未开放，当前仅预留接口，不会修改数据或模型。",
        "cross_project_query": "跨项目资产仅做权限范围内只读查询，不做资产迁移或写入。",
    }[workflow_id]
