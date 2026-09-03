from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.modeling.predictor import predict_with_model
from engine.optimization.constraints import repair_and_report, variable_violation
from engine.optimization.contracts import (
    CandidateResult,
    HardConstraintSpec,
    ObjectiveOperator,
    ObjectiveSpec,
)
from engine.optimization.models import ModelSet
from engine.optimization.search_space import SearchSpace


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    *,
    space: SearchSpace,
    model_set: ModelSet,
    objectives: list[ObjectiveSpec],
    hard_constraints: list[HardConstraintSpec],
) -> list[CandidateResult]:
    repaired_candidates: list[dict[str, Any]] = []
    repaired_reports: list[list[dict[str, Any]]] = []
    variable_violations: list[float] = []
    for candidate in candidates:
        repaired, reports, violation = repair_and_report(
            candidate, space, hard_constraints
        )
        repaired_candidates.append(repaired)
        repaired_reports.append(reports)
        variable_violations.append(violation + variable_violation(repaired, space))

    rows = space.to_model_frame_rows(repaired_candidates)
    predictions = _batch_predict(model_set, rows)
    model_refs = model_set.model_refs()
    results: list[CandidateResult] = []
    for index, values in enumerate(repaired_candidates):
        predicted_values = {
            target_name: float(predictions[target_name]["values"][index])
            for target_name in model_set.models
        }
        uncertainties = [
            float(predictions[target_name]["uncertainty"][index])
            for target_name in model_set.models
            if predictions[target_name]["uncertainty"][index] is not None
        ]
        domains = [
            predictions[target_name]["domains"][index]
            for target_name in model_set.models
        ]
        applicability_domain = _worst_domain(domains)
        objective_values = {
            objective.target_name: predicted_values[objective.target_name]
            for objective in objectives
        }
        objective_errors = {
            objective.target_name: _objective_error(
                objective,
                predicted_values[objective.target_name],
                model_set.models[objective.target_name].metrics["cv_rmse_mean"],
            )
            for objective in objectives
        }
        reports = list(repaired_reports[index])
        for constraint in hard_constraints:
            if constraint.kind != "target_threshold":
                continue
            target_name = str(constraint.target_name)
            predicted = predicted_values.get(target_name)
            if predicted is None:
                satisfied = False
                violation = float("inf")
            else:
                threshold = float(constraint.constant)
                violation = (
                    threshold - predicted
                    if constraint.operator == "greater_or_equal"
                    else predicted - threshold
                )
                violation = max(0.0, violation)
                satisfied = violation <= constraint.tolerance
            reports.append({
                "name": constraint.name,
                "kind": "target_threshold",
                "satisfied": satisfied,
                "violation": violation,
                "tolerance": constraint.tolerance,
            })
        hard_requirement_satisfied = True
        for objective in objectives:
            if objective.requirement.value == "hard":
                satisfied = _hard_objective_satisfied(
                    objective,
                    predicted_values[objective.target_name],
                )
                reports.append({
                    "name": f"objective:{objective.target_name}",
                    "kind": "target_threshold",
                    "satisfied": satisfied,
                    "violation": objective_errors[objective.target_name],
                    "tolerance": objective.tolerance,
                })
                hard_requirement_satisfied = hard_requirement_satisfied and satisfied
        rejected = (
            variable_violations[index] > 1e-9
            or any(not item["satisfied"] for item in reports)
            or not hard_requirement_satisfied
        )
        results.append(CandidateResult(
            candidate_id=f"candidate_{index + 1:06d}",
            values=values,
            predicted_values=predicted_values,
            prediction_uncertainty=max(uncertainties) if uncertainties else None,
            objective_values=objective_values,
            objective_errors=objective_errors,
            hard_constraint_report=reports,
            soft_constraint_scores={},
            soft_constraint_score=0.0,
            applicability_domain=applicability_domain,
            trust_level="REJECTED" if rejected else "HIGH",
            model_refs=model_refs,
        ))
    return results


def _batch_predict(
    model_set: ModelSet,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    if not rows:
        return {
            target_name: {"values": [], "uncertainty": [], "domains": []}
            for target_name in model_set.models
        }
    output: dict[str, dict[str, list[Any]]] = {}
    chunk_size = 128
    for target_name, model in model_set.models.items():
        values: list[float] = []
        uncertainties: list[float | None] = []
        domains: list[str] = []
        feature_names = list(model.bundle["feature_names"])
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            dataframe = pd.DataFrame(chunk, columns=feature_names)
            predictions = predict_with_model(model.bundle, dataframe)
            values.extend(float(item.predicted_value) for item in predictions)
            uncertainties.extend(item.prediction_uncertainty for item in predictions)
            domains.extend(item.applicability_domain for item in predictions)
        output[target_name] = {
            "values": values,
            "uncertainty": uncertainties,
            "domains": domains,
        }
    return output


def objective_vector(
    candidate: CandidateResult,
    objectives: list[ObjectiveSpec],
) -> list[float]:
    return [
        _directional_error(objective, candidate.objective_errors[objective.target_name],
                           candidate.objective_values[objective.target_name])
        for objective in objectives
    ]


def _directional_error(objective: ObjectiveSpec, error: float, value: float) -> float:
    if objective.operator is ObjectiveOperator.maximize:
        return -float(value)
    if objective.operator is ObjectiveOperator.minimize:
        return float(value)
    return float(error)


def _objective_error(
    objective: ObjectiveSpec,
    value: float,
    scale: float,
) -> float:
    safe_scale = max(float(scale), 1e-9)
    if objective.operator is ObjectiveOperator.equal:
        return abs(value - float(objective.value)) / safe_scale
    if objective.operator is ObjectiveOperator.greater_or_equal:
        return max(0.0, float(objective.value) - value) / safe_scale
    if objective.operator is ObjectiveOperator.less_or_equal:
        return max(0.0, value - float(objective.value)) / safe_scale
    if objective.operator is ObjectiveOperator.in_range:
        lower = float(objective.lower_value)
        upper = float(objective.upper_value)
        return max(lower - value, 0.0, value - upper) / safe_scale
    return 0.0


def _hard_objective_satisfied(
    objective: ObjectiveSpec,
    value: float,
) -> bool:
    if objective.operator is ObjectiveOperator.equal:
        return abs(value - float(objective.value)) <= objective.tolerance
    if objective.operator is ObjectiveOperator.greater_or_equal:
        return value >= float(objective.value) - objective.tolerance
    if objective.operator is ObjectiveOperator.less_or_equal:
        return value <= float(objective.value) + objective.tolerance
    if objective.operator is ObjectiveOperator.in_range:
        return (
            float(objective.lower_value) - objective.tolerance
            <= value
            <= float(objective.upper_value) + objective.tolerance
        )
    return True


def _worst_domain(domains: list[str]) -> str:
    if "OUT_OF_DOMAIN" in domains:
        return "OUT_OF_DOMAIN"
    if "EDGE" in domains:
        return "EDGE"
    return "IN_DOMAIN"
