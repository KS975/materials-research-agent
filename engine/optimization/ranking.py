from __future__ import annotations

import math
from typing import Any

import numpy as np

from engine.optimization.contracts import (
    CandidateResult,
    ObjectiveSpec,
    SoftConstraintSpec,
    SoftRankingPolicy,
    VariableType,
)
from engine.optimization.evaluation import objective_vector
from engine.optimization.search_space import SearchSpace


def rank_and_select(
    candidates: list[CandidateResult],
    *,
    objectives: list[ObjectiveSpec],
    soft_constraints: list[SoftConstraintSpec],
    space: SearchSpace,
    history: list[dict[str, Any]],
    top_n: int,
) -> tuple[list[CandidateResult], list[CandidateResult], list[CandidateResult], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    valid = [item for item in candidates if item.trust_level != "REJECTED"]
    rejected = [item for item in candidates if item.trust_level == "REJECTED"]
    if not valid:
        diagnostics = _nearest_rejected(rejected, min(5, len(rejected)))
        return [], [], diagnostics, [{
            "code": "NO_FEASIBLE_CANDIDATES",
            "message": "all evaluated candidates violate hard constraints",
        }]

    _score_soft_constraints(
        valid, soft_constraints=soft_constraints, space=space, history=history
    )
    prefitered: list[CandidateResult] = []
    for candidate in valid:
        removed_by: list[str] = []
        for constraint in soft_constraints:
            if constraint.ranking_policy is SoftRankingPolicy.prefilter:
                score = candidate.soft_constraint_scores[constraint.name]
                if score > (constraint.filter_threshold or 1.0):
                    removed_by.append(constraint.name)
        if removed_by:
            candidate.trust_level = "REJECTED"
            candidate.selection_reason = f"soft prefilter: {','.join(removed_by)}"
        else:
            prefitered.append(candidate)
    valid = prefitered
    if not valid:
        return [], [], _nearest_rejected(candidates, 5), [{
            "code": "NO_FEASIBLE_CANDIDATES",
            "message": "all candidates were removed by explicit soft-constraint prefilters",
        }]

    trusted = [item for item in valid if item.applicability_domain != "OUT_OF_DOMAIN"]
    exploratory = [item for item in valid if item.applicability_domain == "OUT_OF_DOMAIN"]
    for candidate in exploratory:
        candidate.trust_level = "EXPLORATORY"
        candidate.selection_reason = "hard feasible but outside model applicability domain"
    if not trusted:
        return [], exploratory, _nearest_rejected(rejected, 5), [{
            "code": "INSUFFICIENT_TRUSTED_CANDIDATES",
            "message": "no IN_DOMAIN or EDGE candidates were found",
        }]

    vectors = {
        id(candidate): objective_vector(candidate, objectives)
        for candidate in trusted
    }
    for constraint in soft_constraints:
        if constraint.ranking_policy is SoftRankingPolicy.additional_objective:
            for candidate in trusted:
                vectors[id(candidate)].append(
                    candidate.soft_constraint_scores[constraint.name]
                )
    ranks = _pareto_ranks(list(vectors.values()))
    for candidate, rank in zip(trusted, ranks):
        candidate.pareto_rank = rank
        candidate.crowding_distance = _crowding_distance(
            vectors[id(candidate)], list(vectors.values())
        )
    ranked = sorted(
        trusted,
        key=lambda item: (
            item.pareto_rank or 9999,
            -(item.crowding_distance if item.crowding_distance is not None else -1e12),
            item.soft_constraint_score,
            sum(item.objective_errors.values()),
        ),
    )
    selected = _diverse_selection(ranked, space, min(top_n, len(ranked)))
    rank_one_count = sum(item.pareto_rank == 1 for item in selected)
    if rank_one_count < min(top_n, len(trusted)):
        warnings.append({
            "code": "INSUFFICIENT_PARETO_FRONT_CANDIDATES",
            "message": "candidates beyond Pareto rank 1 were used to fill top_n",
            "rank_one_selected": rank_one_count,
        })
    for candidate in selected:
        if candidate.applicability_domain == "EDGE":
            candidate.trust_level = "MEDIUM"
        elif (candidate.pareto_rank or 9999) > 1:
            candidate.trust_level = "MEDIUM"
        elif candidate.soft_constraint_score > 0:
            candidate.trust_level = "MEDIUM"
        else:
            candidate.trust_level = "HIGH"
        candidate.diversity_score = _diversity_score(candidate, selected, space)
    return selected, exploratory, _nearest_rejected(rejected, min(5, len(rejected))), warnings


def _score_soft_constraints(
    candidates: list[CandidateResult],
    *,
    soft_constraints: list[SoftConstraintSpec],
    space: SearchSpace,
    history: list[dict[str, Any]],
) -> None:
    for constraint in soft_constraints:
        scores = [_soft_score(candidate, constraint, space, history) for candidate in candidates]
        for candidate, score in zip(candidates, scores):
            candidate.soft_constraint_scores[constraint.name] = float(np.clip(score, 0, 1))
    total_weight = sum(constraint.weight for constraint in soft_constraints)
    if total_weight <= 0 and soft_constraints:
        for candidate in candidates:
            candidate.soft_constraint_score = 0.0
        return
    for candidate in candidates:
        if total_weight > 0:
            candidate.soft_constraint_score = sum(
                constraint.weight * candidate.soft_constraint_scores[constraint.name]
                for constraint in soft_constraints
            ) / total_weight
        else:
            candidate.soft_constraint_score = 0.0


def _soft_score(
    candidate: CandidateResult,
    constraint: SoftConstraintSpec,
    space: SearchSpace,
    history: list[dict[str, Any]],
) -> float:
    if constraint.kind in {"minimize_expression", "maximize_expression"}:
        coefficients = constraint.params.get("coefficients", [1.0] * len(constraint.variables))
        constant = float(constraint.params.get("constant", 0.0))
        raw = sum(
            coefficient * float(candidate.values[name])
            for name, coefficient in zip(constraint.variables, coefficients)
        ) + constant
        lower, upper = _expression_bounds(constraint, space)
        normalized = (raw - lower) / max(upper - lower, 1e-12)
        if constraint.kind == "maximize_expression":
            normalized = 1.0 - normalized
        return normalized
    if constraint.kind == "history_distance" and history:
        distance = min(_mixed_distance(candidate.values, item, space) for item in history)
        maximum = math.sqrt(len(space.free_variables))
        return distance / max(maximum, 1e-12)
    if constraint.kind == "process_stability":
        centers = constraint.params.get("center", {})
        differences = [
            abs(float(candidate.values[name]) - float(centers[name]))
            / max(
                float(space.variable(name).upper) - float(space.variable(name).lower),
                1e-12,
            )
            for name in constraint.variables
            if name in centers
        ]
        return sum(differences) / max(len(differences), 1)
    return 0.0


def _expression_bounds(
    constraint: SoftConstraintSpec,
    space: SearchSpace,
) -> tuple[float, float]:
    if constraint.normalization is not None and constraint.normalization.lower is not None:
        return float(constraint.normalization.lower), float(constraint.normalization.upper)
    coefficients = constraint.params.get("coefficients", [1.0] * len(constraint.variables))
    constant = float(constraint.params.get("constant", 0.0))
    raw_values: list[list[float]] = []
    for name, coefficient in zip(constraint.variables, coefficients):
        variable = space.variable(name)
        if variable.type is VariableType.categorical:
            raw_values.append([coefficient * index for index in range(len(variable.categories or []))])
        else:
            raw_values.append([
                coefficient * float(variable.lower),
                coefficient * float(variable.upper),
            ])
    combinations = np.array(np.meshgrid(*raw_values)).T.reshape(-1, len(raw_values))
    totals = combinations.sum(axis=1) + constant
    return float(totals.min()), float(totals.max())


def _pareto_ranks(vectors: list[list[float]]) -> list[int]:
    ranks = [0] * len(vectors)
    remaining = set(range(len(vectors)))
    rank = 1
    while remaining:
        front: set[int] = set()
        for i in remaining:
            dominated = any(
                _dominates(vectors[j], vectors[i])
                for j in remaining if j != i
            )
            if not dominated:
                front.add(i)
        if not front:
            front = set(remaining)
        for index in front:
            ranks[index] = rank
        remaining -= front
        rank += 1
    return ranks


def _dominates(left: list[float], right: list[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _crowding_distance(vector: list[float], vectors: list[list[float]]) -> float:
    if len(vectors) <= 2:
        return float("inf")
    matrix = np.asarray(vectors, dtype=float)
    normalized = (matrix - matrix.min(axis=0)) / np.maximum(
        matrix.max(axis=0) - matrix.min(axis=0), 1e-12
    )
    target = np.asarray(vector, dtype=float)
    distances = np.sqrt(((normalized - target) ** 2).sum(axis=1))
    order = np.argsort(distances)
    neighbors = normalized[order[1:3]]
    return float(np.linalg.norm(neighbors[0] - neighbors[1]))


def _diverse_selection(
    ranked: list[CandidateResult],
    space: SearchSpace,
    count: int,
) -> list[CandidateResult]:
    selected = ranked[:1]
    remaining = ranked[1:]
    while len(selected) < count and remaining:
        def choice_score(candidate: CandidateResult) -> tuple[float, int, float]:
            distance = min(_mixed_distance(candidate.values, item.values, space) for item in selected)
            base_index = ranked.index(candidate)
            return (
                -(candidate.pareto_rank or 9999),
                distance,
                candidate.crowding_distance
                if candidate.crowding_distance is not None else -1e12,
                -candidate.soft_constraint_score,
                -base_index,
            )
        next_candidate = max(remaining, key=choice_score)
        selected.append(next_candidate)
        remaining.remove(next_candidate)
    return selected


def _mixed_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    space: SearchSpace,
) -> float:
    differences: list[float] = []
    for variable in space.free_variables:
        if variable.type is VariableType.categorical:
            differences.append(0.0 if left[variable.name] == right[variable.name] else 1.0)
        else:
            span = max(float(variable.upper) - float(variable.lower), 1e-12)
            differences.append(abs(float(left[variable.name]) - float(right[variable.name])) / span)
    return float(math.sqrt(sum(value * value for value in differences) / max(len(differences), 1)))


def _diversity_score(
    candidate: CandidateResult,
    selected: list[CandidateResult],
    space: SearchSpace,
) -> float:
    distances = [
        _mixed_distance(candidate.values, item.values, space)
        for item in selected if item is not candidate
    ]
    return min(distances) if distances else 1.0


def _nearest_rejected(
    rejected: list[CandidateResult],
    count: int,
) -> list[CandidateResult]:
    ranked = sorted(
        rejected,
        key=lambda item: (
            sum(abs(value) for value in item.objective_errors.values()),
            sum(
                abs(report["violation"])
                for report in item.hard_constraint_report
            ),
        ),
    )
    diagnostics = ranked[:count]
    for candidate in diagnostics:
        candidate.trust_level = "REJECTED"
        candidate.selection_reason = "diagnostic only: closest infeasible candidate"
    return diagnostics
