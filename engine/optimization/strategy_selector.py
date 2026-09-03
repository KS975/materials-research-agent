from __future__ import annotations

from typing import Any

from engine.optimization.contracts import (
    ObjectiveOperator,
    ObjectiveSpec,
    OptimizationRequest,
    VariableType,
)
from engine.optimization.generator import estimate_candidate_count
from engine.optimization.models import ModelSet
from engine.optimization.search_space import SearchSpace


def select_strategy(
    request: OptimizationRequest,
    *,
    space: SearchSpace,
    model_set: ModelSet,
) -> tuple[str, str, dict[str, Any]]:
    if request.algorithm_override:
        return request.algorithm_override, "user algorithm override", {
            "override": request.algorithm_override
        }
    thresholds = request.strategy_thresholds
    objectives = request.objectives
    free = space.free_variables
    d_free = len(free)
    candidate_count = estimate_candidate_count(
        space,
        max_count=thresholds.candidate_rank_max_count,
        max_points_per_continuous_dimension=(
            thresholds.candidate_rank_max_points_per_continuous_dimension
        ),
    )
    can_candidate_rank = (
        d_free <= thresholds.candidate_rank_max_free_dimensions
        and candidate_count <= thresholds.candidate_rank_max_count
    )
    mixed = any(
        variable.type in {VariableType.integer, VariableType.categorical}
        for variable in free
    )
    has_hard_target = any(
        objective.requirement.value == "hard"
        and objective.operator is not ObjectiveOperator.equal
        for objective in objectives
    )
    exact_only = bool(objectives) and all(
        objective.operator is ObjectiveOperator.equal for objective in objectives
    )
    history_count = len(request.historical_candidates)
    active_history = _historical_active_dimension(request, space)
    coverage = active_history / d_free if d_free else 0.0

    if can_candidate_rank:
        return "candidate_rank", "search space is low-dimensional and enumerable", {
            "candidate_count_estimate": candidate_count,
            "free_dimension": d_free,
        }
    if mixed and (len(objectives) >= 2 or has_hard_target):
        return "mixed_nsga2", "mixed variables require mixed evolutionary operators", {
            "candidate_count_estimate": candidate_count,
            "free_dimension": d_free,
        }
    if exact_only:
        if (
            history_count >= thresholds.de_rag_min_history_count
            and active_history >= thresholds.de_rag_min_active_dimension
            and coverage >= thresholds.de_rag_min_coverage_ratio
        ):
            return "de_rag", "historical active variables sufficiently cover the search space", {
                "history_count": history_count,
                "active_history_dimension": active_history,
                "coverage_ratio": coverage,
            }
        if (
            d_free > thresholds.active_set_min_search_dimension
            and model_set.sample_feature_ratio < 1
        ):
            return "active_set_de", "high-dimensional small-sample exact-target problem", {
                "sample_feature_ratio": model_set.sample_feature_ratio,
                "free_dimension": d_free,
            }
        return "de", "continuous exact-target matching problem", {
            "free_dimension": d_free
        }
    return "nsga2", "multi-objective or threshold optimization requires Pareto ranking", {
        "objective_count": len(objectives),
        "has_hard_target": has_hard_target,
    }


def _historical_active_dimension(
    request: OptimizationRequest,
    space: SearchSpace,
) -> int:
    active: set[str] = set()
    for candidate in request.historical_candidates:
        if not isinstance(candidate, dict):
            continue
        for variable in space.free_variables:
            value = candidate.get(variable.name)
            if variable.type is VariableType.categorical:
                if value not in {None, ""}:
                    active.add(variable.name)
            elif value is not None and abs(float(value)) > 1e-12:
                active.add(variable.name)
    return len(active)


def objective_conflict_exists(objectives: list[ObjectiveSpec]) -> bool:
    return False
