from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from optimization import (
    ApplicabilityDomainCalibrator,
    BOConfig,
    CandidateGenerator,
    GaussianProcessBayesianOptimizer,
    filter_already_observed_candidate_indices,
    load_search_space,
    parse_inverse_design_text,
    run_inverse_design,
)
from runtime.progress import emit_progress


class V014UIError(RuntimeError):
    """User-facing V0.1.4 UI/API execution error."""


def looks_like_inverse_design(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    has_threshold = bool(re.search(r"(?:>=|<=|≥|≤|>|<)\s*[-+]?\d", text))
    has_design_intent = any(
        token in text
        for token in ("推荐", "设计", "方案", "配方", "反向设计", "逆向设计")
    )
    return has_threshold and has_design_intent


def looks_like_next_experiments(message: str) -> bool:
    text = str(message or "").strip()
    if "下一轮" not in text:
        return False
    return any(
        token in text
        for token in ("实验", "推荐", "做哪", "做什么", "测试", "验证")
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise V014UIError(f"缺少运行文件: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V014UIError(f"运行文件读取失败: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise V014UIError(f"运行文件顶层必须是 JSON object: {path}")
    return data


def _safe_metric_name(value: str) -> str:
    metric = str(value or "").strip()
    if not metric:
        raise V014UIError("target_metric 不能为空")
    if len(metric) > 80 or any(x in metric for x in ("..", "/", "\\")):
        raise V014UIError("target_metric 包含非法路径字符")
    return metric


def _condition_label(name: str) -> tuple[str, str]:
    text = str(name or "").strip()
    if "::" not in text:
        return "设计变量", text
    prefix, label = text.split("::", 1)
    group = {
        "formula": "配方",
        "process": "工艺",
        "condition": "测试条件",
    }.get(prefix, prefix or "设计变量")
    return group, label or text


def _build_experiment_conditions(
    search_space: Any,
    features: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build UI-ready design conditions without changing BO semantics."""
    feature_map = dict(features or {})
    rows: list[dict[str, Any]] = []

    for spec in search_space.variables:
        name = str(spec.name)
        if name not in feature_map:
            continue
        group, label = _condition_label(name)
        rows.append(
            {
                "name": name,
                "group": group,
                "label": label,
                "value": feature_map[name],
                "unit": spec.unit,
                "kind": spec.kind,
            }
        )

    # Fail-open for any future candidate feature not present in the
    # search-space declaration. Keep it visible rather than silently dropping it.
    declared = {str(spec.name) for spec in search_space.variables}
    for name, value in feature_map.items():
        if str(name) in declared:
            continue
        group, label = _condition_label(str(name))
        rows.append(
            {
                "name": str(name),
                "group": group,
                "label": label,
                "value": value,
                "unit": None,
                "kind": "unknown",
            }
        )

    return rows


def _matching_fixture_dir(runtime_root: Path, project_id: int) -> Path | None:
    fixture_root = runtime_root / "v014" / "fixtures"
    if not fixture_root.is_dir():
        return None

    for search_path in sorted(fixture_root.glob("*/search_space.json")):
        try:
            payload = json.loads(search_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("project_id") == project_id:
            return search_path.parent
    return None


def _resolve_search_space(runtime_root: Path, project_id: int) -> Path:
    project_dir = runtime_root / "v014" / "ui_inputs" / f"project_{project_id}"
    preferred = project_dir / "search_space.json"
    if preferred.is_file():
        return preferred

    fixture_dir = _matching_fixture_dir(runtime_root, project_id)
    if fixture_dir is not None and (fixture_dir / "search_space.json").is_file():
        return fixture_dir / "search_space.json"

    raise V014UIError(
        f"Project {project_id} 没有 V0.1.4 search_space.json。"
        f"请放到 {project_dir / 'search_space.json'}。"
    )


def _resolve_inverse_dataset(runtime_root: Path, project_id: int) -> Path:
    project_dir = runtime_root / "v014" / "ui_inputs" / f"project_{project_id}"
    for name in ("dataset.csv", "multiobjective_dataset.csv"):
        path = project_dir / name
        if path.is_file():
            return path

    fixture_dir = _matching_fixture_dir(runtime_root, project_id)
    if fixture_dir is not None:
        for name in ("multiobjective_dataset.csv", "dataset.csv"):
            path = fixture_dir / name
            if path.is_file():
                return path

    raise V014UIError(
        f"Project {project_id} 没有逆向设计数据集。"
        f"请放到 {project_dir / 'dataset.csv'}。"
    )


def _resolve_observations(runtime_root: Path, project_id: int, target_metric: str) -> Path:
    metric = _safe_metric_name(target_metric)
    project_dir = runtime_root / "v014" / "ui_inputs" / f"project_{project_id}"
    for name in (
        f"observations_{metric}.csv",
        "observations.csv",
        "initial_observations.csv",
    ):
        path = project_dir / name
        if path.is_file():
            return path

    fixture_dir = _matching_fixture_dir(runtime_root, project_id)
    if fixture_dir is not None:
        for name in ("initial_observations.csv", "observations.csv"):
            path = fixture_dir / name
            if path.is_file():
                return path

    raise V014UIError(
        f"Project {project_id} 没有“{metric}”的历史实验 CSV。"
        f"请放到 {project_dir / f'observations_{metric}.csv'}。"
    )


def available_gate_metrics(runtime_root: Path, project_id: int) -> list[str]:
    gate_dir = runtime_root / "v013" / "gates"
    prefix = f"project_{project_id}_"
    suffix = "_modeling_gate.json"
    metrics: list[str] = []
    if not gate_dir.is_dir():
        return metrics
    for path in gate_dir.glob(f"{prefix}*{suffix}"):
        name = path.name
        metric = name[len(prefix):-len(suffix)]
        if metric:
            metrics.append(metric)
    return sorted(set(metrics), key=len, reverse=True)


def infer_bo_target_metric(message: str, runtime_root: Path, project_id: int) -> str:
    text = str(message or "").strip()
    metrics = available_gate_metrics(runtime_root, project_id)

    explicit = [metric for metric in metrics if metric in text]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        raise V014UIError(
            "下一轮实验请求同时提到多个已建模指标；T18 当前是单目标 BO，请明确一个目标。"
        )

    match = re.search(r"以\s*([^，,。；;]{1,40}?)\s*为目标", text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate

    if len(metrics) == 1:
        return metrics[0]

    if not metrics:
        raise V014UIError(
            f"Project {project_id} 没有可识别的 Modeling Gate 指标，无法确定 T18 目标。"
        )

    raise V014UIError(
        "请在下一轮实验请求中明确目标指标，例如："
        "“以冲击强度为目标，下一轮推荐5组实验”。"
    )


def infer_batch_size(message: str, default: int = 5) -> int:
    text = str(message or "")
    match = re.search(r"(\d+)\s*(?:组|个|套)\s*(?:实验|方案|候选|测试)?", text)
    if not match:
        return default
    value = int(match.group(1))
    if not 1 <= value <= 20:
        raise V014UIError("单次下一轮实验推荐数量必须在 1～20 之间")
    return value


def run_inverse_design_for_ui(
    *,
    runtime_root: Path,
    project_id: int,
    message: str,
    candidate_count: int = 600,
    random_state: int = 42,
) -> dict[str, Any]:
    emit_progress(
        "optimization_inputs",
        "running",
        "读取优化输入",
        "正在校验 Search Space、历史数据集和可用 Modeling Gate。",
        project_id=project_id,
    )
    search_space_path = _resolve_search_space(runtime_root, project_id)
    dataset_path = _resolve_inverse_dataset(runtime_root, project_id)
    gate_metrics = available_gate_metrics(runtime_root, project_id)
    emit_progress(
        "optimization_inputs",
        "completed",
        "优化输入已就绪",
        f"已找到搜索空间、历史数据集和 {len(gate_metrics)} 个可用建模指标。",
        project_id=project_id,
        gate_metric_count=len(gate_metrics),
    )

    # Project number is routing metadata, not an objective metric.
    clean_message = re.sub(
        r"(?:Project|项目)\s*#?\s*\d+\s*"
        r"(?:号)?\s*(?:项目)?\s*(?:范围)?\s*"
        r"(?:中|里(?:面)?|内)?\s*(?:的)?\s*[：:，,]?\s*",
        "",
        message,
        count=1,
        flags=re.IGNORECASE,
    )

    try:
        request = parse_inverse_design_text(
            clean_message,
            project_id=project_id,
            candidate_count=candidate_count,
            random_state=random_state,
            allowed_metrics=gate_metrics,
        )
        emit_progress(
            "objective_binding",
            "completed",
            "目标字段绑定完成",
            "已将自然语言目标绑定到真实 Modeling Gate 指标："
            + "、".join(item.metric for item in request.objectives)
            + "。",
            objective_count=len(request.objectives),
            objective_metrics=[item.metric for item in request.objectives],
        )
        search_space = load_search_space(_read_json(search_space_path))
        emit_progress(
            "inverse_design",
            "running",
            "运行确定性逆向设计",
            "正在校验模型、生成候选并执行适用域、目标门槛与 Pareto 计算。",
            candidate_count=candidate_count,
        )
        report = run_inverse_design(
            request=request,
            search_space=search_space,
            dataset_csv=dataset_path,
            runtime_root=runtime_root,
            candidate_count_override=candidate_count,
            random_state_override=random_state,
        )
    except Exception as exc:
        if isinstance(exc, V014UIError):
            raise
        raise V014UIError(f"T17 逆向设计执行失败: {type(exc).__name__}: {exc}") from exc

    output = dict(report)
    output["kind"] = "v014_inverse_design"
    output["ui_inputs"] = {
        "search_space_json": str(search_space_path),
        "dataset_csv": str(dataset_path),
    }

    out_dir = runtime_root / "v014" / "ui_runs" / f"project_{project_id}" / "inverse_design"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _load_observations(
    path: Path,
    feature_columns: list[str],
    target_metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_column = f"target::{target_metric}"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing = [
                col for col in [*feature_columns, target_column]
                if col not in fieldnames
            ]
            if missing:
                raise V014UIError(f"历史实验 CSV 缺少字段: {missing}")
            rows = list(reader)
    except OSError as exc:
        raise V014UIError(f"历史实验 CSV 读取失败: {path}: {exc}") from exc

    X: list[list[float]] = []
    y: list[float] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            vector = [float(row[col]) for col in feature_columns]
            target = float(row[target_column])
        except (TypeError, ValueError) as exc:
            raise V014UIError(
                f"历史实验 CSV 第 {row_index} 行存在非数值模型字段"
            ) from exc
        if not np.isfinite(vector).all() or not np.isfinite(target):
            raise V014UIError(f"历史实验 CSV 第 {row_index} 行存在非有限数值")
        X.append(vector)
        y.append(target)

    if len(X) < 10:
        raise V014UIError("T18 至少需要 10 条历史实验")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _load_gate(runtime_root: Path, project_id: int, target_metric: str) -> tuple[Path, dict[str, Any]]:
    metric = _safe_metric_name(target_metric)
    path = (
        runtime_root
        / "v013"
        / "gates"
        / f"project_{project_id}_{metric}_modeling_gate.json"
    )
    gate = _read_json(path)
    if gate.get("stage") != "V0.1.3-B_modeling_gate":
        raise V014UIError("Modeling Gate 报告 stage 非法")
    if gate.get("project_id") != project_id or gate.get("target_metric") != metric:
        raise V014UIError("Modeling Gate 与请求项目/指标不一致")
    if gate.get("training_allowed") is not True:
        raise V014UIError(f"Modeling Gate={gate.get('decision')}，禁止 T18")
    if gate.get("official_model_allowed") is not True:
        raise V014UIError("Modeling Gate 不允许正式模型进入 T18")
    return path, gate


def run_next_experiments_for_ui(
    *,
    runtime_root: Path,
    project_id: int,
    target_metric: str,
    batch_size: int = 5,
    candidate_count: int = 900,
    random_state: int = 42,
    acquisition: str = "EI",
    allow_borderline_for_exploration: bool = True,
    soft_penalty_weight: float = 0.10,
) -> dict[str, Any]:
    metric = _safe_metric_name(target_metric)
    if not 1 <= int(batch_size) <= 20:
        raise V014UIError("batch_size 必须在 1～20 之间")

    search_space_path = _resolve_search_space(runtime_root, project_id)
    observations_path = _resolve_observations(runtime_root, project_id, metric)
    gate_path, gate = _load_gate(runtime_root, project_id, metric)

    try:
        search_space = load_search_space(_read_json(search_space_path))
    except Exception as exc:
        raise V014UIError(f"T18 Search Space 读取失败: {exc}") from exc

    if search_space.project_id is not None and search_space.project_id != project_id:
        raise V014UIError("Search Space project_id 与请求项目不一致")

    feature_columns = [
        spec.name
        for spec in search_space.variables
        if spec.kind in {"continuous", "integer"}
    ]
    if not feature_columns:
        raise V014UIError("T18 没有可用数值设计变量")

    X_obs, y_obs = _load_observations(observations_path, feature_columns, metric)
    ad = ApplicabilityDomainCalibrator(
        feature_columns=feature_columns,
        X=X_obs,
        dropped_rows=0,
    )

    try:
        generation = CandidateGenerator(
            search_space,
            random_state=random_state,
            id_prefix="V014_T18",
        ).generate(
            candidate_count=candidate_count,
            max_attempts=max(10000, candidate_count * 100),
        )
    except Exception as exc:
        raise V014UIError(f"T18 候选生成失败: {type(exc).__name__}: {exc}") from exc

    if not generation["generation_complete"]:
        raise V014UIError("T18 候选生成未完成")

    generated = generation["candidates"]
    X_generated = np.asarray(
        [
            [float(row["features"][col]) for col in feature_columns]
            for row in generated
        ],
        dtype=float,
    )
    keep_indices, duplicate_indices = filter_already_observed_candidate_indices(
        X_generated,
        X_obs,
    )

    eligible: list[dict[str, Any]] = []
    out_of_domain_excluded = 0
    borderline_kept = 0

    for index in keep_indices:
        candidate = generated[index]
        ad_report = ad.evaluate(candidate["features"])
        if ad_report["status"] == "OUT_OF_DOMAIN":
            out_of_domain_excluded += 1
            continue
        if ad_report["status"] == "BORDERLINE":
            if not allow_borderline_for_exploration:
                out_of_domain_excluded += 1
                continue
            borderline_kept += 1
        eligible.append({**candidate, "applicability_domain": ad_report})

    if len(eligible) < batch_size:
        raise V014UIError(
            f"T18 安全候选不足：eligible={len(eligible)}, batch_size={batch_size}"
        )

    X_candidates = np.asarray(
        [
            [float(row["features"][col]) for col in feature_columns]
            for row in eligible
        ],
        dtype=float,
    )
    candidate_ids = [row["candidate_id"] for row in eligible]
    penalties = np.asarray([float(row["soft_penalty"]) for row in eligible], dtype=float)

    try:
        optimizer = GaussianProcessBayesianOptimizer(
            X_obs,
            y_obs,
            config=BOConfig(
                acquisition=acquisition,
                direction="maximize",
                xi=0.01,
                kappa=2.0,
                batch_size=batch_size,
                min_batch_distance=0.20,
                random_state=random_state,
            ),
        )
        fit_summary = optimizer.fit_summary()
        bo_result = optimizer.propose_batch(
            X_candidates,
            candidate_ids,
            candidate_penalties=penalties,
            penalty_weight=soft_penalty_weight,
        )
    except Exception as exc:
        raise V014UIError(f"T18 Gaussian Process / acquisition 失败: {type(exc).__name__}: {exc}") from exc

    next_experiments: list[dict[str, Any]] = []
    for round_info in bo_result["rounds"]:
        candidate = eligible[int(round_info["candidate_index"])]
        next_experiments.append(
            {
                "round": int(round_info["round"]),
                "candidate_id": candidate["candidate_id"],
                "features": candidate["features"],
                "experiment_conditions": _build_experiment_conditions(
                    search_space,
                    candidate["features"],
                ),
                "constraint_status": candidate["constraint_status"],
                "soft_penalty": float(candidate["soft_penalty"]),
                "soft_violations": candidate["soft_violations"],
                "applicability_domain": candidate["applicability_domain"],
                "posterior_mean": float(round_info["posterior_mean"]),
                "posterior_std": float(round_info["posterior_std"]),
                "acquisition_value": float(round_info["acquisition_value"]),
                "adjusted_acquisition": float(round_info["adjusted_acquisition"]),
                "selection_reason": (
                    "GP posterior + adjusted acquisition + Kriging Believer batch update"
                ),
            }
        )

    report = {
        "kind": "v014_bayesian_optimization",
        "stage": "V0.1.4-UI_bayesian_optimization",
        "status": "SUCCESS",
        "answer": (
            f"基于 {len(X_obs)} 条历史实验，已用 Gaussian Process + "
            f"{acquisition.upper()} + Kriging Believer 推荐下一轮 {batch_size} 组实验。"
        ),
        "project_id": project_id,
        "target_metric": metric,
        "gate": {
            "path": str(gate_path),
            "decision": gate.get("decision"),
            "training_allowed": True,
            "official_model_allowed": True,
        },
        "observations": {
            "path": str(observations_path),
            "rows": int(len(X_obs)),
            "best_observed": float(fit_summary["best_observed"]),
            "feature_columns": feature_columns,
        },
        "gaussian_process": fit_summary,
        "candidate_filtering": {
            "generated_hard_valid": int(generation["generated_count"]),
            "already_observed_filtered": len(duplicate_indices),
            "out_of_domain_excluded": out_of_domain_excluded,
            "borderline_kept_for_exploration": borderline_kept,
            "eligible_for_bo": len(eligible),
        },
        "bayesian_optimization": {
            "acquisition": acquisition.upper(),
            "batch_strategy": bo_result["batch_strategy"],
            "batch_size": batch_size,
            "soft_penalty_weight": soft_penalty_weight,
            "selection_score": bo_result["selection_score"],
        },
        "next_experiments": next_experiments,
        "safety": {
            "hard_constraints_required": True,
            "already_observed_candidates_removed": True,
            "out_of_domain_candidates_excluded": True,
            "borderline_candidates_allowed_for_controlled_exploration": (
                allow_borderline_for_exploration
            ),
            "posterior_means_are_not_measured_results": True,
        },
        "ui_inputs": {
            "search_space_json": str(search_space_path),
            "observations_csv": str(observations_path),
        },
    }

    out_dir = runtime_root / "v014" / "ui_runs" / f"project_{project_id}" / "bayesian_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
