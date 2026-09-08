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
        "execution_status": "PLANNED_STAGE_5",
        "steps": [
            "read_external_or_uploaded_result",
            "match_sample_experiment_formula",
            "extract_features",
            "human_review",
            "register_pending_record",
        ],
    },
    "historical_analysis": {
        "display_name": "历史数据分析",
        "scenario_ids": [8, 10, 11],
        "execution_status": "PLANNED_STAGE_4",
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
        "execution_status": "PLANNED_STAGE_4",
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
        "execution_status": "PLANNED_STAGE_4",
        "steps": [
            "define_project_time_scope",
            "aggregate_experiments_and_results",
            "generate_outline",
            "validate_citations",
            "final_report",
        ],
    },
    "closed_loop_asset": {
        "display_name": "模型实验闭环与跨项目复用",
        "scenario_ids": [19, 20],
        "execution_status": "PLANNED_STAGE_5",
        "steps": [
            "identify_returned_experiment",
            "create_dataset_version",
            "train_challenger",
            "human_approval",
            "publish_reusable_asset",
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
    return {
        "schema_version": RESEARCH_WORKFLOW_VERSION,
        "workflow_id": workflow_id,
        "workflow_name": workflow["display_name"],
        "scenario_id": scenario_id,
        "scenario_name": SCENARIO_NAMES[scenario_id],
        "execution_status": workflow["execution_status"],
        "covered_scenario_ids": list(workflow["scenario_ids"]),
        "steps": list(workflow["steps"]),
        "read_only": workflow_id in {
            "hybrid_search_rank",
            "evidence_profile",
            "research_cold_start",
        },
        "write_policy": "NO_WRITE" if workflow_id in {
            "hybrid_search_rank",
            "evidence_profile",
            "research_cold_start",
        } else "HUMAN_APPROVAL_REQUIRED",
        "boundary": _boundary_for(workflow_id),
    }


def looks_like_hybrid_research_request(message: str) -> bool:
    """Recognize stage-3 questions that require joined source reasoning."""
    text = str(message or "").strip()
    markers = (
        "历史资料", "历史案例", "知识库", "向量库", "结合历史", "综合历史",
        "相似配方", "类似配方", "相似样品", "类似样品", "相似实验", "类似实验",
        "原料使用效果", "原料替代", "替代历史", "失败配方", "失败实验",
        "异常案例", "失效案例", "竞品对标", "新项目冷启动", "项目知识问答",
        "去年做过", "为什么停用",
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
    if any(marker in text for marker in ("新项目冷启动", "新项目立项", "首轮方案", "从零开始")):
        return 14
    if (
        any(marker in text for marker in ("替代", "替换"))
        and any(marker in text for marker in ("原料", "材料", "牌号", "供应商"))
    ) or (
        args.get("original_material")
        and args.get("replacement_material")
    ):
        return 6
    if args.get("material_name") or any(
        marker in text
        for marker in ("使用效果", "用在哪些", "用过哪些", "用量多少")
    ):
        return 5
    if any(marker in text for marker in ("失败配方", "失败实验", "失败案例", "失败路线")):
        return 7
    if any(
        marker in text
        for marker in ("异常案例", "失效案例", "开裂", "析出", "变色", "老化", "粘接失效")
    ):
        return 12
    if any(marker in text for marker in ("竞品", "对标")):
        return 13
    if isinstance(args.get("filters"), list) and args["filters"]:
        return 3
    if any(marker in text for marker in ("相似配方", "类似配方")):
        return 1
    if any(marker in text for marker in ("相似样品", "类似样品", "相似实验", "类似实验")):
        return 2
    if _EXPLICIT_IDENTIFIER.search(text) or args.get("identifier"):
        return 4
    if any(marker in text for marker in ("项目知识", "项目问答", "为什么停用", "去年")):
        return 17
    return 1


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
        "result_feature_ingestion": "结果与特征登记需要人工审核，本阶段仅保留工作流位置。",
        "historical_analysis": "统计结论必须绑定样本范围、字段和单位；本阶段仅保留工作流位置。",
        "process_window": "窗口结论必须同时给出可行区间和稳健性；本阶段仅保留工作流位置。",
        "stage_report": "报告引用必须通过证据校验；本阶段仅保留工作流位置。",
        "closed_loop_asset": "数据版本、模型晋级和跨项目复用必须人工审批；本阶段仅保留工作流位置。",
    }[workflow_id]
