from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

from engine.exceptions import ValidationError
from engine.optimization.contracts import VariableSpec, VariableType
from engine.optimization.search_space import SearchSpace


def sample_candidates(
    space: SearchSpace,
    count: int,
    random_seed: int,
    *,
    latin_hypercube: bool = True,
) -> list[dict[str, Any]]:
    free = space.free_variables
    if not free or count <= 0:
        return []
    rng = np.random.default_rng(random_seed)
    columns: dict[str, list[Any]] = {}
    for variable in free:
        if variable.type is VariableType.categorical:
            indexes = rng.integers(0, len(variable.categories or []), size=count)
            columns[variable.name] = [variable.categories[int(index)] for index in indexes]
        elif variable.type is VariableType.integer:
            lower = int(math.ceil(float(variable.lower)))
            upper = int(math.floor(float(variable.upper)))
            if latin_hypercube:
                raw = (np.arange(count) + rng.random(count)) / count
            else:
                raw = rng.random(count)
            columns[variable.name] = [
                int(round(lower + value * (upper - lower))) for value in raw
            ]
        else:
            if latin_hypercube:
                raw = (np.arange(count) + rng.random(count)) / count
            else:
                raw = rng.random(count)
            columns[variable.name] = [
                float(variable.lower + value * (float(variable.upper) - float(variable.lower)))
                for value in raw
            ]
    candidates: list[dict[str, Any]] = []
    for row_index in range(count):
        candidate = {variable.name: variable.fixed_value for variable in space.variables}
        for variable in free:
            candidate[variable.name] = _valid_value(
                space, variable, columns[variable.name][row_index]
            )
        candidates.append(candidate)
    return candidates


def grid_candidates(
    space: SearchSpace,
    *,
    max_count: int,
    max_points_per_continuous_dimension: int,
) -> list[dict[str, Any]]:
    free = space.free_variables
    if not free:
        return []
    continuous_count = sum(
        1 for item in free if item.type is VariableType.continuous
    )
    dimension_limit = max_count ** (1.0 / max(continuous_count, 1))
    values_by_variable: list[list[Any]] = []
    for variable in free:
        if variable.type is VariableType.categorical:
            values = list(variable.categories or [])
        elif variable.type is VariableType.integer:
            lower = int(math.ceil(float(variable.lower)))
            upper = int(math.floor(float(variable.upper)))
            step = max(1, int(math.ceil((upper - lower + 1) / dimension_limit)))
            values = list(range(lower, upper + 1, step))
            if values[-1] != upper:
                values.append(upper)
        else:
            point_count = max(
                2,
                min(
                    max_points_per_continuous_dimension,
                    int(math.floor(dimension_limit)),
                ),
            )
            raw = np.linspace(float(variable.lower), float(variable.upper), point_count)
            values = [float(value) for value in raw]
        values_by_variable.append(values)
    estimated = int(math.prod([len(item) for item in values_by_variable]))
    if estimated > max_count:
        raise ValidationError(
            f"candidate rank grid estimate {estimated} exceeds {max_count}"
        )
    fixed = {item.name: item.fixed_value for item in space.variables}
    candidates: list[dict[str, Any]] = []
    for combination in product(*values_by_variable):
        candidate = dict(fixed)
        candidate.update(dict(zip([item.name for item in free], combination)))
        candidates.append(candidate)
    return candidates


def estimate_candidate_count(
    space: SearchSpace,
    *,
    max_count: int,
    max_points_per_continuous_dimension: int,
) -> int:
    free = space.free_variables
    if not free:
        return 1
    continuous_count = sum(
        1 for item in free if item.type is VariableType.continuous
    )
    points_per_dimension = min(
        max_points_per_continuous_dimension,
        int(math.floor(max_count ** (1.0 / max(continuous_count, 1)))),
    )
    count = 1
    for variable in free:
        if variable.type is VariableType.categorical:
            count *= len(variable.categories or [])
        elif variable.type is VariableType.integer:
            lower = int(math.ceil(float(variable.lower)))
            upper = int(math.floor(float(variable.upper)))
            count *= upper - lower + 1
        else:
            count *= max(2, points_per_dimension)
    return int(count)


def inject_history(
    history: list[dict[str, Any]],
    space: SearchSpace,
) -> list[dict[str, Any]]:
    injected: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        candidate: dict[str, Any] = {}
        valid = True
        for variable in space.variables:
            if variable.name not in row:
                valid = False
                break
            candidate[variable.name] = row[variable.name]
        if valid:
            injected.append(candidate)
    return injected


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = repr(sorted((key, str(value)) for key, value in candidate.items()))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _valid_value(
    space: SearchSpace,
    variable: VariableSpec,
    value: Any,
) -> Any:
    if variable.type is VariableType.categorical:
        return value
    if variable.type is VariableType.integer:
        return int(round(float(value)))
    return float(min(max(float(value), float(variable.lower)), float(variable.upper)))
