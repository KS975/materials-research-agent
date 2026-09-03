from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.optimization.constraints import repair_and_report
from engine.optimization.evaluation import evaluate_candidates
from engine.optimization.generator import (
    deduplicate_candidates,
    grid_candidates,
    inject_history,
    sample_candidates,
)
from engine.optimization.ranking import rank_and_select
from engine.optimization.contracts import (
    CandidateResult,
    ObjectiveSpec,
    OptimizationRequest,
    VariableSpec,
    VariableType,
)
from engine.optimization.models import ModelSet
from engine.optimization.search_space import SearchSpace


@dataclass
class StrategyRun:
    candidates: list[CandidateResult]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = "COMPLETE"
    stop_reason: str | None = None
    completed_evaluations: int = 0


class StrategyContext:
    def __init__(
        self,
        request: OptimizationRequest,
        space: SearchSpace,
        model_set: ModelSet,
    ) -> None:
        self.request = request
        self.space = space
        self.model_set = model_set
        self.started_at = time.perf_counter()
        self.evaluations = 0
        self.cache: dict[str, CandidateResult] = {}

    def evaluate(self, candidates: list[dict[str, Any]]) -> list[CandidateResult]:
        output: list[CandidateResult] = []
        missing: list[dict[str, Any]] = []
        for candidate in candidates:
            key = _candidate_key(candidate)
            if key in self.cache:
                output.append(self.cache[key])
            else:
                missing.append(candidate)
                output.append(None)  # type: ignore[arg-type]
        if missing:
            evaluated = evaluate_candidates(
                missing,
                space=self.space,
                model_set=self.model_set,
                objectives=self.request.objectives,
                hard_constraints=self.request.hard_constraints,
            )
            evaluation_by_key = {
                _candidate_key(item.values): item for item in evaluated
            }
            self.evaluations += len(missing)
            for original, result in zip(missing, evaluated):
                original_key = _candidate_key(original)
                repaired_key = _candidate_key(result.values)
                self.cache[original_key] = result
                self.cache[repaired_key] = result
            output = [
                self.cache[_candidate_key(candidate)]
                for candidate in candidates
            ]
        return output

    def timed_out(self) -> bool:
        if self.request.time_limit is None:
            return False
        return time.perf_counter() - self.started_at >= self.request.time_limit

    def evaluation_budget_reached(self) -> bool:
        return (
            self.request.max_evaluations is not None
            and self.evaluations >= self.request.max_evaluations
        )


def run_strategy(
    strategy: str,
    *,
    request: OptimizationRequest,
    space: SearchSpace,
    model_set: ModelSet,
) -> StrategyRun:
    context = StrategyContext(request, space, model_set)
    if strategy == "candidate_rank":
        run = _run_candidate_rank(request, space, context)
    elif strategy == "de":
        run = _run_de(request, space, context, search_variables=space.free_variables)
    elif strategy == "de_rag":
        run = _run_de_rag(request, space, context)
    elif strategy == "active_set_de":
        run = _run_active_set_de(request, space, model_set, context)
    elif strategy in {"nsga2", "mixed_nsga2"}:
        run = _run_nsga2(request, space, context, mixed=strategy == "mixed_nsga2")
    else:
        raise ValueError(f"unsupported optimization strategy: {strategy}")
    run.completed_evaluations = context.evaluations
    run.diagnostics["elapsed_ms"] = int(
        (time.perf_counter() - context.started_at) * 1000
    )
    if run.stop_reason is None and context.timed_out():
        run.status = "PARTIAL"
        run.stop_reason = "time_limit"
    return run


def _run_candidate_rank(
    request: OptimizationRequest,
    space: SearchSpace,
    context: StrategyContext,
) -> StrategyRun:
    thresholds = request.strategy_thresholds
    candidates = grid_candidates(
        space,
        max_count=thresholds.candidate_rank_max_count,
        max_points_per_continuous_dimension=(
            thresholds.candidate_rank_max_points_per_continuous_dimension
        ),
    )
    evaluated = context.evaluate(candidates)
    return StrategyRun(
        candidates=evaluated,
        diagnostics={
            "generated_count": len(candidates),
            "algorithm_parameters": {
                "max_count": thresholds.candidate_rank_max_count,
                "max_points_per_continuous_dimension": (
                    thresholds.candidate_rank_max_points_per_continuous_dimension
                ),
            },
        },
    )


def _run_de(
    request: OptimizationRequest,
    space: SearchSpace,
    context: StrategyContext,
    *,
    search_variables: list[VariableSpec],
    stage_name: str = "de",
) -> StrategyRun:
    if not search_variables or any(
        variable.type is not VariableType.continuous for variable in search_variables
    ):
        raise ValueError("DE search variables must be continuous")
    dimension = len(search_variables)
    population_size = int(np.clip(5 * dimension, 30, 120))
    if dimension < 3:
        population_size = max(population_size, 100)
    if request.max_evaluations is not None:
        population_size = min(
            population_size, max(5, request.max_evaluations // 2)
        )
    generations = 50
    rng = np.random.default_rng(request.random_seed)
    population = sample_candidates(
        space, population_size, request.random_seed, latin_hypercube=True
    )
    history = inject_history(request.historical_candidates, space)
    if history:
        population = deduplicate_candidates(history[:population_size] + population)
        population = population[:population_size]
    evaluated = context.evaluate(population)
    convergence: list[float] = []
    mutation_factor = 0.7
    crossover_probability = 0.85
    stop_reason: str | None = None

    for generation in range(generations):
        if context.timed_out() or context.evaluation_budget_reached():
            stop_reason = (
                "time_limit" if context.timed_out() else "max_evaluations"
            )
            break
        next_population: list[dict[str, Any]] = []
        next_evaluated: list[CandidateResult] = []
        for base_index, base in enumerate(population):
            if context.timed_out() or context.evaluation_budget_reached():
                stop_reason = (
                    "time_limit" if context.timed_out() else "max_evaluations"
                )
                next_population.append(base)
                next_evaluated.append(evaluated[base_index])
                continue
            indices = [index for index in range(len(population)) if index != base_index]
            if len(indices) < 3:
                break
            a_index, b_index, c_index = rng.choice(indices, size=3, replace=False)
            mutant: dict[str, Any] = dict(base)
            for variable in search_variables:
                lower = float(variable.lower)
                upper = float(variable.upper)
                value = (
                    float(population[a_index][variable.name])
                    + mutation_factor * (
                        float(population[b_index][variable.name])
                        - float(population[c_index][variable.name])
                    )
                )
                value = min(max(value, lower), upper)
                if rng.random() > crossover_probability:
                    value = float(base[variable.name])
                mutant[variable.name] = value
            trial = context.evaluate([mutant])[0]
            base_result = evaluated[base_index]
            if _scalar_objective(trial, request.objectives) <= _scalar_objective(
                base_result, request.objectives
            ):
                next_population.append(mutant)
                next_evaluated.append(trial)
            else:
                next_population.append(base)
                next_evaluated.append(base_result)
        population = next_population
        evaluated = next_evaluated
        convergence.append(float(np.mean([
            _scalar_objective(item, request.objectives) for item in evaluated
        ])))

    return StrategyRun(
        candidates=evaluated,
        status="PARTIAL" if stop_reason else "COMPLETE",
        stop_reason=stop_reason,
        diagnostics={
            "generated_count": len(population),
            "stage": stage_name,
            "search_variables": [item.name for item in search_variables],
            "algorithm_parameters": {
                "population_size": population_size,
                "generations": generations,
                "mutation_factor": mutation_factor,
                "crossover_probability": crossover_probability,
            },
            "convergence_history": convergence,
        },
    )


def _run_de_rag(
    request: OptimizationRequest,
    space: SearchSpace,
    context: StrategyContext,
) -> StrategyRun:
    active = _historical_active_variables(request, space)
    search_variables = [space.variable(name) for name in active]
    run = _run_de(
        request, space, context, search_variables=search_variables, stage_name="de_rag"
    )
    run.diagnostics.update({
        "historical_candidate_count": len(request.historical_candidates),
        "historical_active_dimension": len(active),
        "coverage_ratio": len(active) / max(len(space.free_variables), 1),
    })
    return run


def _run_active_set_de(
    request: OptimizationRequest,
    space: SearchSpace,
    model_set: ModelSet,
    context: StrategyContext,
) -> StrategyRun:
    stage1 = _run_de(
        request, space, context,
        search_variables=space.free_variables,
        stage_name="active_set_de_stage1",
    )
    if context.evaluation_budget_reached() or context.timed_out():
        stage1.diagnostics.update({
            "sparse_method": "active_set_de_stage1_only",
            "active_variables": [item.name for item in space.free_variables],
            "removed_variables": [],
        })
        stage1.status = "PARTIAL"
        stage1.stop_reason = (
            "time_limit" if context.timed_out() else "max_evaluations"
        )
        return stage1
    feasible = [
        item for item in stage1.candidates if item.trust_level != "REJECTED"
    ]
    feasible.sort(
        key=lambda item: _scalar_objective(item, request.objectives)
    )
    top_count = max(
        1,
        int(np.ceil(
            len(feasible) * request.strategy_thresholds.active_set_stage1_top_ratio
        )),
    )
    top = feasible[:top_count]
    active_names: list[str] = []
    active_records: list[dict[str, Any]] = []
    for variable in space.free_variables:
        ratios = []
        for candidate in top:
            effective_range = max(
                float(variable.upper) - float(variable.lower), 1e-12
            )
            ratios.append(float(candidate.values[variable.name]) / effective_range)
        usage_ratio = float(np.median(ratios))
        threshold, source = _active_threshold(variable, model_set)
        keep = usage_ratio >= threshold
        if keep:
            active_names.append(variable.name)
        active_records.append({
            "variable": variable.name,
            "usage_ratio": usage_ratio,
            "normalized_active_threshold": threshold,
            "threshold_source": source,
            "kept": keep,
        })
    if not active_names:
        active_records.sort(key=lambda item: item["usage_ratio"], reverse=True)
        fallback_count = max(1, int(np.ceil(len(active_records) * 0.10)))
        active_names = [item["variable"] for item in active_records[:fallback_count]]
        for record in active_records:
            record["kept"] = record["variable"] in active_names

    search_variables = [space.variable(name) for name in active_names]
    stage2 = _run_de(
        request, space, context,
        search_variables=search_variables,
        stage_name="active_set_de_stage2",
    )
    stage2.diagnostics.update({
        "sparse_method": "active_set_de",
        "active_variables": active_names,
        "removed_variables": [
            variable.name for variable in space.free_variables
            if variable.name not in active_names
        ],
        "variable_activity": active_records,
        "stage1_metrics": stage1.diagnostics,
    })
    if stage1.stop_reason:
        stage2.status = "PARTIAL"
        stage2.stop_reason = stage1.stop_reason
    return stage2


def _run_nsga2(
    request: OptimizationRequest,
    space: SearchSpace,
    context: StrategyContext,
    *,
    mixed: bool,
) -> StrategyRun:
    free = space.free_variables
    if not free:
        raise ValueError("optimization requires at least one free variable")
    dimension = len(free)
    population_size = int(np.clip(5 * dimension, 50, 200))
    if request.max_evaluations is not None:
        population_size = min(
            population_size, max(5, request.max_evaluations // 2)
        )
    generations = min(100, 60)
    rng = np.random.default_rng(request.random_seed)
    population = sample_candidates(
        space, population_size, request.random_seed, latin_hypercube=True
    )
    history = inject_history(request.historical_candidates, space)
    if history:
        population = deduplicate_candidates(history + population)[:population_size]
    evaluated = context.evaluate(population)
    convergence: list[float] = []
    stop_reason: str | None = None

    for generation in range(generations):
        if context.timed_out() or context.evaluation_budget_reached():
            stop_reason = (
                "time_limit" if context.timed_out() else "max_evaluations"
            )
            break
        parents = _tournament_parents(evaluated, request.objectives, rng)
        offspring: list[dict[str, Any]] = []
        for left_result, right_result in zip(parents[::2], parents[1::2]):
            child_a, child_b = _crossover(
                left_result.values, right_result.values, space, rng, mixed=mixed
            )
            offspring.append(_mutate(child_a, space, rng, mixed=mixed))
            offspring.append(_mutate(child_b, space, rng, mixed=mixed))
        random_count = max(2, population_size // 10)
        offspring.extend(sample_candidates(
            space,
            random_count,
            request.random_seed + generation + 1,
            latin_hypercube=False,
        ))
        offspring = deduplicate_candidates(offspring)[:population_size]
        offspring_evaluated = context.evaluate(offspring)
        combined = deduplicate_candidates(
            [item.values for item in evaluated + offspring_evaluated]
        )
        selected, _, _, _ = rank_and_select(
            evaluated + offspring_evaluated,
            objectives=request.objectives,
            soft_constraints=[],
            space=space,
            history=request.historical_candidates,
            top_n=population_size,
        )
        evaluated = selected if selected else evaluated[:population_size]
        convergence.append(float(np.mean([
            min(sum(item.objective_errors.values()) for item in evaluated),
        ])))

    return StrategyRun(
        candidates=evaluated,
        status="PARTIAL" if stop_reason else "COMPLETE",
        stop_reason=stop_reason,
        diagnostics={
            "generated_count": len(population),
            "algorithm_parameters": {
                "population_size": population_size,
                "generations": generations,
                "mixed_variables": mixed,
            },
            "convergence_history": convergence,
        },
    )


def _tournament_parents(
    candidates: list[CandidateResult],
    objectives: list[ObjectiveSpec],
    rng: np.random.Generator,
) -> list[CandidateResult]:
    if not candidates:
        return []
    parents: list[CandidateResult] = []
    tournament_size = min(3, len(candidates))
    while len(parents) < len(candidates):
        contenders = rng.choice(len(candidates), size=tournament_size, replace=False)
        winner = min(
            (candidates[index] for index in contenders),
            key=lambda item: _scalar_objective(item, objectives),
        )
        parents.append(winner)
    return parents


def _crossover(
    left: dict[str, Any],
    right: dict[str, Any],
    space: SearchSpace,
    rng: np.random.Generator,
    *,
    mixed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    child_a: dict[str, Any] = {}
    child_b: dict[str, Any] = {}
    for variable in space.variables:
        if variable.type is VariableType.categorical or (
            mixed and rng.random() < 0.5
        ):
            child_a[variable.name] = left[variable.name]
            child_b[variable.name] = right[variable.name]
        else:
            alpha = rng.random()
            left_value = float(left[variable.name]) if variable.type is not VariableType.categorical else 0.0
            right_value = float(right[variable.name]) if variable.type is not VariableType.categorical else 0.0
            a = alpha * left_value + (1 - alpha) * right_value
            b = alpha * right_value + (1 - alpha) * left_value
            child_a[variable.name] = _bounded_numeric(space, variable, a)
            child_b[variable.name] = _bounded_numeric(space, variable, b)
    return child_a, child_b


def _mutate(
    values: dict[str, Any],
    space: SearchSpace,
    rng: np.random.Generator,
    *,
    mixed: bool,
) -> dict[str, Any]:
    mutated = dict(values)
    for variable in space.free_variables:
        probability = 0.2 if mixed else 0.1
        if rng.random() > probability:
            continue
        if variable.type is VariableType.categorical:
            categories = variable.categories or []
            mutated[variable.name] = categories[int(rng.integers(0, len(categories)))]
        elif variable.type is VariableType.integer:
            span = max(int(variable.upper - variable.lower), 1)
            delta = int(rng.integers(-max(span // 10, 1), max(span // 10, 1) + 1))
            mutated[variable.name] = _bounded_numeric(
                space, variable, int(mutated[variable.name]) + delta
            )
        else:
            span = float(variable.upper) - float(variable.lower)
            mutated[variable.name] = _bounded_numeric(
                space,
                variable,
                float(mutated[variable.name]) + rng.normal(0, span * 0.08),
            )
    return mutated


def _bounded_numeric(
    space: SearchSpace,
    variable: VariableSpec,
    value: float,
) -> Any:
    value = min(max(float(value), float(variable.lower)), float(variable.upper))
    if variable.type is VariableType.integer:
        return int(round(value))
    return float(value)


def _scalar_objective(
    candidate: CandidateResult,
    objectives: list[ObjectiveSpec],
) -> float:
    total_weight = sum(objective.weight for objective in objectives)
    if total_weight <= 0:
        return sum(candidate.objective_errors.values())
    return sum(
        objective.weight * candidate.objective_errors[objective.target_name]
        for objective in objectives
    ) / total_weight


def _historical_active_variables(
    request: OptimizationRequest,
    space: SearchSpace,
) -> list[str]:
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
    return sorted(active)


def _active_threshold(
    variable: VariableSpec,
    model_set: ModelSet,
) -> tuple[float, str]:
    effective_range = max(
        float(variable.upper) - float(variable.lower), 1e-12
    )
    if variable.min_effective_value is not None:
        return float(variable.min_effective_value) / effective_range, "min_effective_value"
    raw_values: list[float] = []
    for model in model_set.models.values():
        if variable.name not in model.bundle["feature_names"]:
            continue
        lower, upper = model.training_bounds(variable.name)
        column = model.raw_training_matrix()[:, list(model.bundle["feature_names"]).index(variable.name)]
        nonzero = column[column > max(lower, 0) + (upper - lower) * 1e-9]
        if nonzero.size:
            raw_values.append(float(np.quantile(nonzero, 0.25) / effective_range))
    if raw_values:
        return float(np.median(raw_values)), "training_nonzero_q25"
    return 0.05, "engine_default"


def _candidate_key(candidate: dict[str, Any]) -> str:
    return repr(sorted((key, str(value)) for key, value in candidate.items()))
