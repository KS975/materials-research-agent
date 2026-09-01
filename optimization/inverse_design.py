from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from .applicability import ApplicabilityDomainCalibrator
from .candidate_generation import CandidateGenerator
from .pareto import (
    ObjectiveSpec,
    diverse_select,
    non_dominated_sort,
    normalized_utilities,
)
from .search_space import SearchSpace
from runtime.progress import emit_progress


REQUEST_STAGE = "V0.1.4-T17_inverse_design_request"
VALID_THRESHOLD_OPERATORS = {">=", ">", "<=", "<"}


class InverseDesignError(RuntimeError):
    """Raised when an inverse-design request cannot be executed safely."""


@dataclass(frozen=True)
class ParsedInverseDesignRequest:
    project_id: int
    request_name: str
    objectives: tuple[ObjectiveSpec, ...]
    recommendation_count: int
    candidate_count: int
    max_attempts: int
    random_state: int
    soft_penalty_weight: float
    diversity_weight: float
    source: str
    raw_request_text: str | None = None


def _finite_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise InverseDesignError(f"{path} 必须是有限数值")
    return float(value)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InverseDesignError(f"{path} 必须是正整数")
    return int(value)


def parse_inverse_design_request(
    data: dict[str, Any],
    *,
    source: str = "json",
    raw_request_text: str | None = None,
) -> ParsedInverseDesignRequest:
    if not isinstance(data, dict):
        raise InverseDesignError("inverse design request 必须是 JSON object")

    if data.get("stage") != REQUEST_STAGE:
        raise InverseDesignError(
            f"stage 必须是 {REQUEST_STAGE}"
        )

    project_id = data.get("project_id")
    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise InverseDesignError("project_id 必须是整数")

    request_name = str(
        data.get("request_name") or "inverse_design"
    ).strip()
    if not request_name:
        raise InverseDesignError("request_name 不能为空")

    raw_objectives = data.get("objectives")
    if not isinstance(raw_objectives, list) or not raw_objectives:
        raise InverseDesignError("objectives 至少需要 1 个目标")

    objectives: list[ObjectiveSpec] = []
    seen_metrics: set[str] = set()

    for index, item in enumerate(raw_objectives):
        if not isinstance(item, dict):
            raise InverseDesignError(
                f"objectives[{index}] 必须是 JSON object"
            )

        metric = str(item.get("metric") or "").strip()
        if not metric:
            raise InverseDesignError(
                f"objectives[{index}].metric 不能为空"
            )
        if metric in seen_metrics:
            raise InverseDesignError(f"目标重复: {metric}")
        seen_metrics.add(metric)

        direction = str(
            item.get("direction") or ""
        ).strip().lower()
        if direction not in {"maximize", "minimize"}:
            raise InverseDesignError(
                f"{metric}.direction 必须是 maximize 或 minimize"
            )

        threshold = item.get("threshold")
        if not isinstance(threshold, dict):
            raise InverseDesignError(
                f"{metric}.threshold 是 T17 必填项"
            )

        operator = str(
            threshold.get("operator") or ""
        ).strip()
        if operator not in VALID_THRESHOLD_OPERATORS:
            raise InverseDesignError(
                f"{metric}.threshold.operator 非法"
            )

        threshold_value = _finite_number(
            threshold.get("value"),
            f"{metric}.threshold.value",
        )

        # For inverse design, keep direction consistent with the user's
        # threshold semantics. This prevents a request like "at least 40"
        # from accidentally being optimized downward.
        if operator in {">=", ">"} and direction != "maximize":
            raise InverseDesignError(
                f"{metric}: {operator} 阈值应使用 maximize"
            )
        if operator in {"<=", "<"} and direction != "minimize":
            raise InverseDesignError(
                f"{metric}: {operator} 阈值应使用 minimize"
            )

        weight = _finite_number(
            item.get("weight", 1.0),
            f"{metric}.weight",
        )
        if weight < 0:
            raise InverseDesignError(
                f"{metric}.weight 不能小于 0"
            )

        objectives.append(
            ObjectiveSpec(
                metric=metric,
                direction=direction,
                threshold_operator=operator,
                threshold_value=threshold_value,
                weight=weight,
            )
        )

    recommendation_count = _positive_int(
        data.get("recommendation_count", 5),
        "recommendation_count",
    )
    candidate_count = _positive_int(
        data.get("candidate_count", 600),
        "candidate_count",
    )
    max_attempts = _positive_int(
        data.get("max_attempts", max(10000, candidate_count * 100)),
        "max_attempts",
    )
    random_state = data.get("random_state", 42)
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise InverseDesignError("random_state 必须是整数")

    soft_penalty_weight = _finite_number(
        data.get("soft_penalty_weight", 0.20),
        "soft_penalty_weight",
    )
    if soft_penalty_weight < 0:
        raise InverseDesignError(
            "soft_penalty_weight 不能小于 0"
        )

    diversity_weight = _finite_number(
        data.get("diversity_weight", 0.35),
        "diversity_weight",
    )
    if not 0 <= diversity_weight <= 1:
        raise InverseDesignError(
            "diversity_weight 必须位于 [0, 1]"
        )

    return ParsedInverseDesignRequest(
        project_id=project_id,
        request_name=request_name,
        objectives=tuple(objectives),
        recommendation_count=recommendation_count,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        random_state=int(random_state),
        soft_penalty_weight=soft_penalty_weight,
        diversity_weight=diversity_weight,
        source=source,
        raw_request_text=raw_request_text,
    )


def _parse_small_positive_integer(value: str) -> int:
    token = str(value or "").strip()
    if token.isdigit():
        return int(token)

    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if token == "十":
        return 10
    if "十" in token:
        left, right = token.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if token in digits:
        return digits[token]
    raise InverseDesignError(f"无法识别推荐数量: {value}")


def _clean_inverse_design_metric_phrase(value: str) -> str:
    metric = str(value or "").strip()
    metric = re.sub(
        r"(?:Project|项目)\s*#?\s*\d+\s*"
        r"(?:号)?\s*(?:项目)?\s*(?:范围)?\s*"
        r"(?:中|里(?:面)?|内)?\s*(?:的)?\s*[：:，,]?\s*",
        "",
        metric,
        count=1,
        flags=re.IGNORECASE,
    )
    metric = re.sub(
        r"^(我要|我希望|希望|目标|要求|同时|并且|且|"
        r"请把|请将|请让|让|使)",
        "",
        metric,
    ).strip(" ：:")
    metric = re.sub(
        r"^(?:(?:请|麻烦)\s*)?"
        r"(?:(?:给|帮|替|为)\s*我\s*)?"
        r"(?:推荐|设计|生成|给出|提供)\s*"
        r"(?:(?:\d+|[一二两三四五六七八九十]+)\s*(?:组|个|套)\s*)?"
        r"(?:(?:在|从)\s*)?"
        r"(?:(?:该|这个)\s*)?"
        r"(?:项目\s*)?"
        r"(?:中|里|内|的)?\s*",
        "",
        metric,
    ).strip(" ：:")
    return metric


def _resolve_inverse_design_metric(
    metric_phrase: str,
    allowed_metrics: tuple[str, ...] | None,
) -> str:
    cleaned = _clean_inverse_design_metric_phrase(metric_phrase)
    if allowed_metrics is None:
        if not cleaned:
            raise InverseDesignError(
                f"无法识别目标名称: {metric_phrase}"
            )
        return cleaned

    phrase_folded = str(metric_phrase).strip().casefold()
    candidates: list[tuple[int, int, str]] = []
    allowed_suffixes = {
        "",
        "值",
        "数值",
        "指标",
        "目标",
        "要求",
        "含量",
    }
    for metric in allowed_metrics:
        metric_folded = metric.casefold()
        position = phrase_folded.rfind(metric_folded)
        if position < 0:
            continue
        suffix = str(metric_phrase)[position + len(metric):].strip(" ：:")
        if suffix not in allowed_suffixes:
            continue
        candidates.append((position + len(metric), len(metric), metric))

    if not candidates:
        available = "、".join(allowed_metrics)
        shown = cleaned or str(metric_phrase).strip()
        raise InverseDesignError(
            f"无法将“{shown}”绑定到当前项目的 Modeling Gate 指标；"
            f"可用指标：{available}"
        )

    # Prefer the metric ending nearest to the comparator, then the longest
    # name. This deterministically chooses “悬臂梁冲击强度” over its suffix
    # “冲击强度” when both are present in the project schema.
    candidates.sort(reverse=True)
    return candidates[0][2]


def parse_inverse_design_text(
    text: str,
    *,
    project_id: int,
    candidate_count: int = 600,
    random_state: int = 42,
    allowed_metrics: Iterable[str] | None = None,
) -> ParsedInverseDesignRequest:
    """
    Deterministic acceptance parser for simple Chinese inverse-design requests.

    Supported style:
      冲击强度 >= 43、MFR >= 8.5，推荐5组方案
      冲击强度 ≥ 43，成本 ≤ 20，给我3组

    When allowed_metrics is supplied, every parsed objective must bind to an
    existing project Modeling Gate metric before any runtime path is resolved.
    This is intentionally narrow. It is not presented as a general-purpose
    natural-language understanding layer.
    """
    raw = str(text or "").strip()
    if not raw:
        raise InverseDesignError("request text 不能为空")

    metrics = None
    if allowed_metrics is not None:
        metrics = tuple(
            dict.fromkeys(
                str(metric).strip()
                for metric in allowed_metrics
                if str(metric).strip()
            )
        )
        if not metrics:
            raise InverseDesignError(
                "当前项目没有可用的 Modeling Gate 指标，无法执行 T17 逆设计"
            )

    objectives = []

    for segment in re.split(r"[，,、；;\n]+", raw):
        segment = segment.strip()
        if not segment:
            continue

        match = re.search(
            r"(>=|<=|≥|≤|>|<)\s*([-+]?\d+(?:\.\d+)?)",
            segment,
        )
        if not match:
            continue

        metric_phrase = segment[:match.start()].strip()
        metric = _resolve_inverse_design_metric(metric_phrase, metrics)

        if not metric:
            raise InverseDesignError(
                f"无法识别目标名称: {segment}"
            )

        operator = match.group(1)
        operator = {
            "≥": ">=",
            "≤": "<=",
        }.get(operator, operator)

        threshold_value = float(match.group(2))
        direction = (
            "maximize"
            if operator in {">=", ">"}
            else "minimize"
        )

        objectives.append(
            {
                "metric": metric,
                "direction": direction,
                "threshold": {
                    "operator": operator,
                    "value": threshold_value,
                },
                "weight": 1.0,
            }
        )

    if not objectives:
        raise InverseDesignError(
            "没有从 request text 中识别到“指标 + 比较符 + 数值”"
        )

    count_match = re.search(
        r"(\d+|[一二两三四五六七八九十]+)\s*"
        r"(?:组|个|套)\s*(?:方案|设计|候选)?",
        raw,
    )
    recommendation_count = (
        _parse_small_positive_integer(count_match.group(1))
        if count_match
        else 5
    )

    request_doc = {
        "stage": REQUEST_STAGE,
        "project_id": project_id,
        "request_name": "inverse_design_text",
        "objectives": objectives,
        "recommendation_count": recommendation_count,
        "candidate_count": candidate_count,
        "random_state": random_state,
        "soft_penalty_weight": 0.20,
        "diversity_weight": 0.35,
    }

    return parse_inverse_design_request(
        request_doc,
        source="text",
        raw_request_text=raw,
    )


def _standard_gate_path(
    runtime_root: Path,
    project_id: int,
    metric: str,
) -> Path:
    return (
        runtime_root
        / "v013"
        / "gates"
        / f"project_{project_id}_{metric}_modeling_gate.json"
    )


def _standard_model_path(
    runtime_root: Path,
    project_id: int,
    metric: str,
) -> Path:
    return (
        runtime_root
        / "v013"
        / "model_comparison"
        / f"project_{project_id}_{metric}"
        / "best_model.joblib"
    )


def _load_json(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        raise InverseDesignError(f"缺少文件: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise InverseDesignError(
            f"JSON 顶层必须是 object: {path}"
        )

    return data


def _threshold_margin(
    prediction: float,
    objective: ObjectiveSpec,
) -> float:
    threshold = float(objective.threshold_value)
    operator = str(objective.threshold_operator)

    if operator in {">=", ">"}:
        return float(prediction) - threshold
    if operator in {"<=", "<"}:
        return threshold - float(prediction)

    raise InverseDesignError(
        f"不支持 threshold operator: {operator}"
    )


def _passes_margin(
    margin: float,
    objective: ObjectiveSpec,
) -> bool:
    if objective.threshold_operator in {">", "<"}:
        return margin > 0
    return margin >= 0


def _normalized_shortfall(
    margin: float,
    objective: ObjectiveSpec,
) -> float:
    if margin >= 0:
        return 0.0
    reference = max(abs(float(objective.threshold_value)), 1.0)
    return (-margin / reference) * max(objective.weight, 0.0)


def _build_design_card(
    row: dict[str, Any],
    objectives: tuple[ObjectiveSpec, ...],
    *,
    rank: int | None,
) -> dict[str, Any]:
    target_margins = {
        obj.metric: float(
            _threshold_margin(
                row["predictions"][obj.metric],
                obj,
            )
        )
        for obj in objectives
    }

    rationale_parts = [
        "全部目标门槛通过",
        "Applicability Domain=IN_DOMAIN / LOW",
    ]

    if row.get("pareto_rank") == 1:
        rationale_parts.append("位于 Pareto Front")

    if row.get("soft_penalty", 0.0) > 0:
        rationale_parts.append(
            f"存在 soft penalty={row['soft_penalty']:.6f}"
        )
    else:
        rationale_parts.append("无 soft penalty")

    return {
        "recommendation_rank": rank,
        "candidate_id": row["candidate_id"],
        "features": row["features"],
        "predictions": row["predictions"],
        "target_margins": target_margins,
        "all_targets_pass": True,
        "pareto_rank": row.get("pareto_rank"),
        "base_utility": row.get("base_utility"),
        "adjusted_utility": row.get("adjusted_utility"),
        "soft_penalty": row.get("soft_penalty", 0.0),
        "soft_violations": row.get("soft_violations", []),
        "applicability_domain": {
            "status": row["applicability_domain"]["status"],
            "risk": row["applicability_domain"]["risk"],
            "reasons": row["applicability_domain"]["reasons"],
        },
        "rationale": "；".join(rationale_parts),
    }


def run_inverse_design(
    *,
    request: ParsedInverseDesignRequest,
    search_space: SearchSpace,
    dataset_csv: str | Path,
    runtime_root: str | Path = ".runtime",
    candidate_count_override: int | None = None,
    random_state_override: int | None = None,
) -> dict[str, Any]:
    """
    Execute the T17 inverse-design acceptance pipeline.

    Safety invariants:
    1. Every objective requires a Modeling Gate with:
       training_allowed=true AND official_model_allowed=true.
    2. Every prediction comes from a persisted sklearn best-model bundle.
    3. Only IN_DOMAIN candidates can satisfy the formal inverse-design pool.
    4. No feasible candidate => NO_FEASIBLE_DESIGN; never fabricate results.
    """
    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        raise InverseDesignError(
            f"缺少 ML 依赖: {exc}"
        ) from exc

    runtime_root = Path(runtime_root)
    dataset_csv = Path(dataset_csv)

    if request.project_id != search_space.project_id:
        raise InverseDesignError(
            "request.project_id 与 search space project_id 不一致"
        )

    models: dict[str, Any] = {}
    model_names: dict[str, str] = {}
    gates: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, str] = {}
    gate_paths: dict[str, str] = {}
    feature_columns: list[str] | None = None

    emit_progress(
        "model_validation",
        "running",
        "校验 Modeling Gate 与模型",
        f"正在校验 {len(request.objectives)} 个目标对应的准入报告和模型产物。",
        objective_count=len(request.objectives),
    )

    for objective in request.objectives:
        gate_path = _standard_gate_path(
            runtime_root,
            request.project_id,
            objective.metric,
        )
        gate = _load_json(gate_path)

        if gate.get("stage") != "V0.1.3-B_modeling_gate":
            raise InverseDesignError(
                f"{objective.metric}: gate stage 非法"
            )
        if gate.get("project_id") != request.project_id:
            raise InverseDesignError(
                f"{objective.metric}: gate project_id 不一致"
            )
        if gate.get("target_metric") != objective.metric:
            raise InverseDesignError(
                f"{objective.metric}: gate target_metric 不一致"
            )
        if gate.get("training_allowed") is not True:
            raise InverseDesignError(
                f"{objective.metric}: Modeling Gate 禁止训练"
            )
        if gate.get("official_model_allowed") is not True:
            raise InverseDesignError(
                f"{objective.metric}: Modeling Gate 不允许正式模型进入 T17"
            )

        model_path = _standard_model_path(
            runtime_root,
            request.project_id,
            objective.metric,
        )
        if not model_path.exists():
            raise InverseDesignError(
                f"{objective.metric}: best model 不存在: {model_path}"
            )

        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict):
            raise InverseDesignError(
                f"{objective.metric}: best model bundle 非法"
            )
        if bundle.get("project_id") != request.project_id:
            raise InverseDesignError(
                f"{objective.metric}: model project_id 不一致"
            )
        if bundle.get("target_metric") != objective.metric:
            raise InverseDesignError(
                f"{objective.metric}: model target_metric 不一致"
            )

        cols = bundle.get("feature_columns")
        if not isinstance(cols, list) or not cols:
            raise InverseDesignError(
                f"{objective.metric}: model feature_columns 缺失"
            )

        if feature_columns is None:
            feature_columns = list(cols)
        elif list(cols) != feature_columns:
            raise InverseDesignError(
                "T17 当前要求所有目标模型使用相同且同序的 feature_columns"
            )

        model = bundle.get("model")
        if model is None:
            raise InverseDesignError(
                f"{objective.metric}: model object 缺失"
            )

        models[objective.metric] = model
        model_names[objective.metric] = str(
            bundle.get("model_name") or type(model).__name__
        )
        gates[objective.metric] = {
            "decision": gate.get("decision"),
            "training_allowed": gate.get("training_allowed"),
            "official_model_allowed": gate.get("official_model_allowed"),
        }
        model_paths[objective.metric] = str(model_path)
        gate_paths[objective.metric] = str(gate_path)

    emit_progress(
        "model_validation",
        "completed",
        "模型校验通过",
        "全部目标均通过 Modeling Gate，模型字段顺序一致且可用于本轮计算。",
        model_names=model_names,
        feature_count=len(feature_columns or []),
    )

    assert feature_columns is not None

    missing_model_features = [
        col for col in feature_columns
        if col not in search_space.variable_map
    ]
    if missing_model_features:
        raise InverseDesignError(
            f"搜索空间缺少模型特征: {missing_model_features}"
        )

    categorical_model_features = [
        col for col in feature_columns
        if search_space.variable_map[col].kind == "categorical"
    ]
    if categorical_model_features:
        raise InverseDesignError(
            "T17 当前模型只接受数值特征；"
            f"发现 categorical model features: {categorical_model_features}"
        )

    ad = ApplicabilityDomainCalibrator.from_csv(
        dataset_csv,
        feature_columns=feature_columns,
    )

    candidate_count = (
        int(candidate_count_override)
        if candidate_count_override is not None
        else request.candidate_count
    )
    random_state = (
        int(random_state_override)
        if random_state_override is not None
        else request.random_state
    )

    if candidate_count <= 0:
        raise InverseDesignError(
            "candidate_count override 必须 > 0"
        )

    emit_progress(
        "candidate_generation",
        "running",
        "生成候选设计",
        f"正在按 Search Space 与 HARD constraints 生成 {candidate_count} 个候选。",
        requested_count=candidate_count,
    )
    generation = CandidateGenerator(
        search_space,
        random_state=random_state,
        id_prefix="V014_T17",
    ).generate(
        candidate_count=candidate_count,
        max_attempts=max(request.max_attempts, candidate_count),
    )

    if not generation["generation_complete"]:
        raise InverseDesignError(
            "候选生成未完成；请增加 max_attempts 或放宽 HARD constraints"
        )

    candidates = generation["candidates"]
    emit_progress(
        "candidate_generation",
        "completed",
        "候选设计生成完成",
        f"已生成 {len(candidates)} 个满足硬约束的候选。",
        generated_count=len(candidates),
    )

    emit_progress(
        "prediction_and_domain",
        "running",
        "预测并检查适用域",
        "正在用持久化模型预测目标，并逐项执行 Applicability Domain 检查。",
    )
    X = np.asarray(
        [
            [
                float(candidate["features"][col])
                for col in feature_columns
            ]
            for candidate in candidates
        ],
        dtype=float,
    )

    predictions_by_metric = {
        objective.metric: models[objective.metric].predict(X)
        for objective in request.objectives
    }

    evaluated: list[dict[str, Any]] = []

    for row_index, candidate in enumerate(candidates):
        predictions = {
            objective.metric: float(
                predictions_by_metric[objective.metric][row_index]
            )
            for objective in request.objectives
        }

        ad_report = ad.evaluate(candidate["features"])

        target_margins = {
            objective.metric: _threshold_margin(
                predictions[objective.metric],
                objective,
            )
            for objective in request.objectives
        }

        threshold_results = {
            objective.metric: _passes_margin(
                target_margins[objective.metric],
                objective,
            )
            for objective in request.objectives
        }

        total_shortfall = sum(
            _normalized_shortfall(
                target_margins[objective.metric],
                objective,
            )
            for objective in request.objectives
        )

        evaluated.append(
            {
                **candidate,
                "predictions": predictions,
                "target_margins": target_margins,
                "threshold_results": threshold_results,
                "all_target_thresholds_pass": all(
                    threshold_results.values()
                ),
                "applicability_domain": ad_report,
                "trusted_for_formal_design": (
                    ad_report["status"] == "IN_DOMAIN"
                ),
                "total_normalized_threshold_shortfall": float(
                    total_shortfall
                ),
            }
        )

    trusted = [
        row for row in evaluated
        if row["trusted_for_formal_design"]
    ]

    qualified = [
        row for row in trusted
        if row["all_target_thresholds_pass"]
    ]
    emit_progress(
        "prediction_and_domain",
        "completed",
        "预测与适用域检查完成",
        (
            f"{len(evaluated)} 个候选完成预测；{len(trusted)} 个处于 IN_DOMAIN；"
            f"{len(qualified)} 个同时满足全部目标。"
        ),
        evaluated_count=len(evaluated),
        trusted_in_domain=len(trusted),
        qualified_count=len(qualified),
    )

    near_misses = sorted(
        [
            row for row in trusted
            if not row["all_target_thresholds_pass"]
        ],
        key=lambda row: (
            row["total_normalized_threshold_shortfall"],
            row["soft_penalty"],
            row["candidate_id"],
        ),
    )[:10]

    recommendations: list[dict[str, Any]] = []
    pareto_front: list[dict[str, Any]] = []

    if qualified:
        emit_progress(
            "pareto_selection",
            "running",
            "执行 Pareto 与多样性筛选",
            "正在对可信候选计算非支配排序、综合效用和设计多样性。",
            qualified_count=len(qualified),
        )
        prediction_rows = [
            row["predictions"]
            for row in qualified
        ]

        pareto_ranks = non_dominated_sort(
            prediction_rows,
            list(request.objectives),
        )
        utilities = normalized_utilities(
            prediction_rows,
            list(request.objectives),
        )

        for row, pareto_rank, utility in zip(
            qualified,
            pareto_ranks,
            utilities,
        ):
            row["pareto_rank"] = int(pareto_rank)
            row["base_utility"] = float(utility)
            row["adjusted_utility"] = float(
                utility
                - request.soft_penalty_weight
                * row["soft_penalty"]
            )

        pareto_front = [
            row for row in qualified
            if row["pareto_rank"] == 1
        ]

        source_pool = (
            pareto_front
            if len(pareto_front) >= request.recommendation_count
            else sorted(
                qualified,
                key=lambda row: (
                    row["pareto_rank"],
                    -row["adjusted_utility"],
                    row["candidate_id"],
                ),
            )
        )

        recommendations = diverse_select(
            source_pool,
            count=min(
                request.recommendation_count,
                len(source_pool),
            ),
            feature_columns=feature_columns,
            utility_key="adjusted_utility",
            diversity_weight=request.diversity_weight,
        )
        emit_progress(
            "pareto_selection",
            "completed",
            "推荐方案筛选完成",
            (
                f"Pareto Front 共 {len(pareto_front)} 个候选；"
                f"最终选择 {len(recommendations)} 组差异化设计。"
            ),
            pareto_front_count=len(pareto_front),
            recommendation_count=len(recommendations),
        )
    else:
        emit_progress(
            "pareto_selection",
            "completed",
            "没有可信可行候选",
            "没有候选同时通过目标门槛与 IN_DOMAIN 要求，因此不会补造推荐方案。",
            pareto_front_count=0,
            recommendation_count=0,
        )

    if not qualified:
        status = "NO_FEASIBLE_DESIGN"
    elif len(recommendations) < request.recommendation_count:
        status = "PARTIAL_FEASIBLE_DESIGN"
    else:
        status = "SUCCESS"

    design_cards = [
        _build_design_card(
            row,
            request.objectives,
            rank=index,
        )
        for index, row in enumerate(recommendations, start=1)
    ]

    near_miss_cards = []
    for row in near_misses:
        near_miss_cards.append(
            {
                "candidate_id": row["candidate_id"],
                "features": row["features"],
                "predictions": row["predictions"],
                "target_margins": row["target_margins"],
                "threshold_results": row["threshold_results"],
                "total_normalized_threshold_shortfall": (
                    row["total_normalized_threshold_shortfall"]
                ),
                "soft_penalty": row["soft_penalty"],
                "applicability_domain": {
                    "status": row["applicability_domain"]["status"],
                    "risk": row["applicability_domain"]["risk"],
                },
            }
        )

    if status == "SUCCESS":
        answer = (
            f"找到 {len(qualified)} 个同时满足全部目标且处于 "
            f"IN_DOMAIN 的可信候选；从 Pareto/多样性结果中给出 "
            f"{len(design_cards)} 组设计。"
        )
    elif status == "PARTIAL_FEASIBLE_DESIGN":
        answer = (
            f"仅找到 {len(qualified)} 个满足全部目标的可信候选，"
            f"不足以安全给出 {request.recommendation_count} 组不同设计；"
            f"实际返回 {len(design_cards)} 组。"
        )
    else:
        answer = (
            "没有找到同时满足全部目标且处于 IN_DOMAIN 的可信候选。"
            "系统未生成或补造虚构方案；请降低目标门槛、扩大经过验证的"
            "搜索域，或补充实验数据后重试。"
        )

    is_fixture = "fixtures" in dataset_csv.parts

    return {
        "stage": "V0.1.4-T17_inverse_design",
        "status": status,
        "answer": answer,
        "project_id": request.project_id,
        "request": {
            "source": request.source,
            "request_name": request.request_name,
            "raw_request_text": request.raw_request_text,
            "recommendation_count": request.recommendation_count,
            "candidate_count": candidate_count,
            "random_state": random_state,
            "objectives": [
                {
                    "metric": objective.metric,
                    "direction": objective.direction,
                    "threshold_operator": objective.threshold_operator,
                    "threshold_value": objective.threshold_value,
                    "weight": objective.weight,
                }
                for objective in request.objectives
            ],
        },
        "evidence": {
            "gates": gates,
            "gate_paths": gate_paths,
            "model_names": model_names,
            "model_paths": model_paths,
            "feature_columns": feature_columns,
            "dataset_csv": str(dataset_csv),
        },
        "generation": {
            key: value
            for key, value in generation.items()
            if key != "candidates"
        },
        "applicability_domain_calibration": ad.summary(),
        "counts": {
            "generated_hard_valid": len(evaluated),
            "trusted_in_domain": len(trusted),
            "qualified_all_targets": len(qualified),
            "pareto_front": len(pareto_front),
            "recommended": len(recommendations),
            "near_miss_returned": len(near_miss_cards),
        },
        "design_cards": design_cards,
        "near_miss_candidates": near_miss_cards,
        "safety": {
            "all_objectives_gate_training_allowed": all(
                item["training_allowed"] is True
                for item in gates.values()
            ),
            "all_objectives_official_model_allowed": all(
                item["official_model_allowed"] is True
                for item in gates.values()
            ),
            "formal_design_requires_in_domain": True,
            "out_of_domain_can_be_recommended": False,
            "predictions_from_persisted_sklearn_models": True,
            "llm_generated_prediction_values": False,
            "fabricate_missing_recommendations": False,
        },
        "scientific_status": {
            "fixture_or_real_data": (
                "fixture" if is_fixture else "unknown_or_real"
            ),
            "official_scientific_recommendation_allowed": (
                False if is_fixture else None
            ),
            "note": (
                "Fixture output validates the inverse-design engineering "
                "pipeline only; it is not a materials-science conclusion."
                if is_fixture
                else (
                    "For real use, model/data governance and domain review "
                    "still apply."
                )
            ),
        },
    }
