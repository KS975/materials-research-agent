from __future__ import annotations

import math
import time
import warnings as warning_module
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from engine.exceptions import ValidationError
from engine.modeling.predictor import predict_with_model
from engine.optimization.constraints import repair_and_report, variable_violation
from engine.optimization.contracts import (
    CandidateResult,
    ObjectiveOperator,
    OptimizationRequest,
)
from engine.optimization.evaluation import _objective_error
from engine.optimization.generator import (
    deduplicate_candidates,
    sample_candidates,
)
from engine.optimization.models import ModelSet
from engine.optimization.ranking import _mixed_distance, rank_and_select
from engine.optimization.search_space import SearchSpace


def recommend_next_experiments(
    *,
    request: OptimizationRequest,
    space: SearchSpace,
    model_set: ModelSet,
) -> tuple[list[CandidateResult], list[CandidateResult], list[CandidateResult], dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    valid_history = [
        item for item in request.historical_experiments
        if all(objective.target_name in item.observed_values for objective in request.objectives)
    ]
    if not valid_history:
        raise ValidationError("BO requires at least one complete observed experiment")

    d_search = len(space.free_variables)
    min_gp_samples = max(10, 2 * d_search + 1)
    if len(valid_history) < min_gp_samples:
        selected, exploratory, diagnostics, warnings = _cold_start_design(
            request, space, model_set
        )
        warnings.append({
            "code": "BO_COLD_START_FALLBACK",
            "message": "observed history is below the minimum GP sample count",
            "valid_history_count": len(valid_history),
            "min_gp_samples": min_gp_samples,
        })
        diagnostics.update({
            "valid_history_count": len(valid_history),
            "min_gp_samples": min_gp_samples,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        })
        return selected, exploratory, [], diagnostics, warnings

    X, y = _history_training_data(
        valid_history, request=request, space=space, model_set=model_set
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    surrogate = GaussianProcessRegressor(
        normalize_y=True,
        random_state=request.random_seed,
    )
    captured_warnings: list[warning_module.WarningMessage] = []
    with warning_module.catch_warnings(record=True) as captured_warnings:
        warning_module.simplefilter("always")
        surrogate.fit(X_scaled, y)
    warnings: list[dict[str, Any]] = [
        {
            "code": "GP_TRAINING_WARNING",
            "category": type(item.message).__name__,
            "message": str(item.message),
        }
        for item in captured_warnings
    ]

    pool_size = min(
        max(5000, request.top_n * 100),
        request.max_evaluations or 5000,
    )
    pool = sample_candidates(
        space,
        pool_size,
        request.random_seed,
        latin_hypercube=True,
    )
    pool_rows = np.asarray([
        [float(row[name]) for name in space.model_feature_names]
        for row in space.to_model_frame_rows(pool)
    ])
    predicted_mean, predicted_std = surrogate.predict(
        scaler.transform(pool_rows), return_std=True
    )
    acquisition_name = _select_acquisition(request, y)
    acquisition_values = _acquisition(
        acquisition_name,
        predicted_mean,
        predicted_std,
        float(np.max(y)),
    )

    evaluated = _evaluate_for_bo(
        pool,
        request=request,
        space=space,
        model_set=model_set,
    )
    by_values = {tuple(sorted(item.values.items())): item for item in evaluated}
    feasible: list[CandidateResult] = []
    for index, candidate in enumerate(pool):
        result = by_values.get(tuple(sorted(candidate.items())))
        if result is None or result.trust_level == "REJECTED":
            continue
        if result.applicability_domain == "OUT_OF_DOMAIN":
            continue
        if _is_history_duplicate(candidate, valid_history, space):
            continue
        result.acquisition_name = acquisition_name
        result.acquisition_value = float(acquisition_values[index])
        result.acquisition_mean = float(predicted_mean[index])
        result.acquisition_std = float(predicted_std[index])
        result.distance_to_nearest_history = min(
            _mixed_distance(candidate, item.values, space)
            for item in valid_history
        )
        result.exploration_score = float(
            (predicted_std[index] - np.min(predicted_std))
            / max(np.max(predicted_std) - np.min(predicted_std), 1e-12)
        )
        feasible.append(result)
    if not feasible:
        raise ValidationError("BO found no hard-feasible non-duplicate experiment candidates")

    selected: list[CandidateResult] = []
    remaining = list(feasible)
    while remaining and len(selected) < request.top_n:
        next_candidate = max(
            remaining,
            key=lambda item: (
                item.acquisition_value
                if item.acquisition_value is not None else -np.inf,
                item.exploration_score
                if item.exploration_score is not None else -np.inf,
            ),
        )
        selected.append(next_candidate)
        remaining = [
            item for item in remaining
            if _mixed_distance(item.values, next_candidate.values, space) > 0.10
        ]
    selected, exploratory, diagnostic, rank_warnings = rank_and_select(
        selected,
        objectives=request.objectives,
        soft_constraints=request.soft_constraints,
        space=space,
        history=request.historical_candidates,
        top_n=request.top_n,
    )
    warnings.extend(rank_warnings)
    for candidate in selected:
        candidate.selection_reason = (
            f"acquisition {candidate.acquisition_name} with batch diversity"
        )
    diagnostics = {
        "selected_strategy": "bo",
        "strategy_reason": "Gaussian-process Bayesian optimization over complete observed experiments",
        "valid_history_count": len(valid_history),
        "min_gp_samples": min_gp_samples,
        "generated_count": len(pool),
        "candidate_pool_size": pool_size,
        "acquisition": acquisition_name,
        "surrogate": "GaussianProcessRegressor",
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "completed_evaluations": len(evaluated),
    }
    return selected, exploratory, diagnostic, diagnostics, warnings


def _cold_start_design(
    request: OptimizationRequest,
    space: SearchSpace,
    model_set: ModelSet,
) -> tuple[list[CandidateResult], list[CandidateResult], list[CandidateResult], dict[str, Any], list[dict[str, Any]]]:
    pool_size = min(
        max(5000, request.top_n * 100),
        request.max_evaluations or 5000,
    )
    pool = sample_candidates(
        space,
        pool_size,
        request.random_seed,
        latin_hypercube=True,
    )
    evaluated = _evaluate_for_bo(
        pool,
        request=request,
        space=space,
        model_set=model_set,
    )
    selected, exploratory, diagnostics_from_ranking, warnings = rank_and_select(
        evaluated,
        objectives=request.objectives,
        soft_constraints=request.soft_constraints,
        space=space,
        history=request.historical_candidates,
        top_n=request.top_n,
    )
    for candidate in selected:
        candidate.acquisition_name = None
        candidate.acquisition_value = None
        candidate.acquisition_mean = None
        candidate.acquisition_std = None
        candidate.selection_reason = "cold-start diversity design"
    details = {
        "selected_strategy": "cold_start_design",
        "strategy_reason": "insufficient complete observed history for a stable GP surrogate",
        "generated_count": len(pool),
        "candidate_pool_size": pool_size,
        "sampling_method": "latin_hypercube_with_mixed_variables",
        "completed_evaluations": len(evaluated),
    }
    return selected, exploratory, details, warnings


def _history_training_data(
    history: list[Any],
    *,
    request: OptimizationRequest,
    space: SearchSpace,
    model_set: ModelSet,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    rewards = []
    defaults = space.to_model_frame_rows([{}])
    for experiment in history:
        source = dict(experiment.values)
        for feature_name in space.model_feature_names:
            if feature_name not in source:
                source[feature_name] = defaults[0][feature_name]
        rows.append([
            float(source[name])
            for name in space.model_feature_names
        ])
        rewards.append(_observed_reward(experiment.observed_values, request, model_set))
    return np.asarray(rows, dtype=float), np.asarray(rewards, dtype=float)


def _observed_reward(
    observed_values: dict[str, float],
    request: OptimizationRequest,
    model_set: ModelSet,
) -> float:
    total_weight = sum(objective.weight for objective in request.objectives)
    if total_weight <= 0:
        total_weight = 1.0
    score = 0.0
    for objective in request.objectives:
        scale = model_set.models[objective.target_name].metrics["cv_rmse_mean"]
        score += objective.weight * _objective_error(
            objective, float(observed_values[objective.target_name]), scale
        )
    return -float(score / total_weight)


def _select_acquisition(
    request: OptimizationRequest,
    observed_rewards: np.ndarray,
) -> str:
    if request.acquisition:
        return request.acquisition
    feasible = bool(np.isfinite(observed_rewards).any())
    if not feasible:
        return "ucb"
    if request.preference == "exploit":
        return "pi"
    return "ei"


def _acquisition(
    name: str,
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
) -> np.ndarray:
    safe_std = np.maximum(std, 1e-12)
    if name == "pi":
        z = (mean - best_observed - 1e-6) / safe_std
        return norm.cdf(z)
    if name == "ucb":
        return mean + 1.96 * safe_std
    improvement = mean - best_observed - 1e-6
    z = improvement / safe_std
    return improvement * norm.cdf(z) + safe_std * norm.pdf(z)


def _evaluate_for_bo(
    candidates: list[dict[str, Any]],
    *,
    request: OptimizationRequest,
    space: SearchSpace,
    model_set: ModelSet,
) -> list[CandidateResult]:
    repaired_candidates = []
    repaired_reports: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        repaired, reports, variable_violation_value = repair_and_report(
            candidate, space, request.hard_constraints
        )
        if (
            variable_violation(repaired, space) <= 1e-9
            and all(item["satisfied"] for item in reports)
        ):
            repaired_candidates.append(repaired)
            repaired_reports.append(reports)
    rows = space.to_model_frame_rows(repaired_candidates)
    output: list[CandidateResult] = []
    chunk_size = 128
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start:start + chunk_size]
        candidate_chunk = repaired_candidates[start:start + chunk_size]
        report_chunk = repaired_reports[start:start + chunk_size]
        target_predictions: dict[str, list[float]] = {}
        target_domains: dict[str, list[str]] = {}
        target_uncertainty: dict[str, list[float | None]] = {}
        for target_name, model in model_set.models.items():
            dataframe = pd.DataFrame(
                chunk_rows, columns=list(model.bundle["feature_names"])
            )
            predictions = predict_with_model(model.bundle, dataframe)
            target_predictions[target_name] = [
                item.predicted_value for item in predictions
            ]
            target_domains[target_name] = [
                item.applicability_domain for item in predictions
            ]
            target_uncertainty[target_name] = [
                item.prediction_uncertainty for item in predictions
            ]
        for index, candidate in enumerate(candidate_chunk):
            predicted = {
                target: float(values[index])
                for target, values in target_predictions.items()
            }
            domains = [
                values[index] for values in target_domains.values()
            ]
            uncertainties = [
                values[index]
                for values in target_uncertainty.values()
                if values[index] is not None
            ]
            objective_errors = {
                objective.target_name: _objective_error(
                    objective,
                    predicted[objective.target_name],
                    model_set.models[objective.target_name].metrics["cv_rmse_mean"],
                )
                for objective in request.objectives
            }
            reports = report_chunk[index] + _target_reports(
                predicted, request.objectives
            )
            rejected = any(not item["satisfied"] for item in reports)
            domain = (
                "OUT_OF_DOMAIN" if "OUT_OF_DOMAIN" in domains
                else "EDGE" if "EDGE" in domains
                else "IN_DOMAIN"
            )
            output.append(CandidateResult(
                candidate_id=f"bo_candidate_{len(output) + 1:06d}",
                values=candidate,
                predicted_values=predicted,
                prediction_uncertainty=max(uncertainties) if uncertainties else None,
                objective_values=predicted,
                objective_errors=objective_errors,
                hard_constraint_report=reports,
                soft_constraint_scores={},
                soft_constraint_score=0.0,
                applicability_domain=domain,
                trust_level="REJECTED" if rejected else "HIGH",
                model_refs=model_set.model_refs(),
            ))
    return output


def _target_reports(
    predicted: dict[str, float],
    objectives: list[Any],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for objective in objectives:
        value = predicted[objective.target_name]
        if objective.requirement.value != "hard":
            continue
        if objective.operator is ObjectiveOperator.equal:
            satisfied = abs(value - float(objective.value)) <= objective.tolerance
        elif objective.operator is ObjectiveOperator.greater_or_equal:
            satisfied = value >= float(objective.value) - objective.tolerance
        elif objective.operator is ObjectiveOperator.less_or_equal:
            satisfied = value <= float(objective.value) + objective.tolerance
        elif objective.operator is ObjectiveOperator.in_range:
            satisfied = (
                float(objective.lower_value) - objective.tolerance
                <= value
                <= float(objective.upper_value) + objective.tolerance
            )
        else:
            satisfied = True
        reports.append({
            "name": f"objective:{objective.target_name}",
            "kind": "target_threshold",
            "satisfied": satisfied,
            "violation": 0.0 if satisfied else 1.0,
            "tolerance": objective.tolerance,
        })
    return reports


def _is_history_duplicate(
    candidate: dict[str, Any],
    history: list[Any],
    space: SearchSpace,
) -> bool:
    return any(
        _mixed_distance(candidate, item.values, space) <= 1e-12
        for item in history
    )
