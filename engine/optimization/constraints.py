from __future__ import annotations

from typing import Any

import numpy as np

from engine.optimization.contracts import HardConstraintSpec
from engine.optimization.search_space import SearchSpace


def repair_and_report(
    values: dict[str, Any],
    space: SearchSpace,
    constraints: list[HardConstraintSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    repaired = dict(values)
    reports: list[dict[str, Any]] = []
    total_violation = 0.0

    for constraint in constraints:
        if constraint.kind == "bound":
            variable = space.variable(constraint.variables[0])
            lower = constraint.lower if constraint.lower is not None else variable.lower
            upper = constraint.upper if constraint.upper is not None else variable.upper
            original = float(repaired[variable.name])
            clipped = min(max(original, float(lower)), float(upper))
            repaired[variable.name] = _value_for_variable(space, variable, clipped)
            violation = max(float(lower) - original, original - float(upper), 0.0)
            total_violation += violation
            reports.append(_report(constraint, violation <= constraint.tolerance, violation))

    for constraint in constraints:
        if constraint.kind in {"mutex", "categorical_incompatibility"}:
            active = _active_variables(repaired, space, constraint.variables)
            if len(active) <= 1:
                satisfied = True
                violation = 0.0
            else:
                repaired, violation = _repair_mutex(repaired, space, constraint)
                satisfied = violation <= constraint.tolerance
            total_violation += violation
            reports.append(_report(constraint, satisfied, violation))

    for constraint in constraints:
        if constraint.kind == "linear_sum":
            repaired, violation = _repair_linear_sum(repaired, space, constraint)
            total_violation += violation
            reports.append(_report(constraint, violation <= constraint.tolerance, violation))

    for variable in space.variables:
        if variable.fixed_value is not None:
            repaired[variable.name] = variable.fixed_value
    return repaired, reports, float(total_violation)


def variable_violation(
    values: dict[str, Any],
    space: SearchSpace,
) -> float:
    violation = 0.0
    for variable in space.variables:
        value = values[variable.name]
        if variable.type is None:
            continue
        if str(variable.type.value) == "categorical":
            if value not in (variable.categories or []):
                violation += 1.0
            continue
        numeric = float(value)
        violation += max(
            float(variable.lower) - numeric,
            numeric - float(variable.upper),
            0.0,
        )
    return float(violation)


def _repair_linear_sum(
    values: dict[str, Any],
    space: SearchSpace,
    constraint: HardConstraintSpec,
) -> tuple[dict[str, Any], float]:
    repaired = dict(values)
    names = constraint.variables
    coefficients = constraint.coefficients or [1.0] * len(names)
    if any(coefficient <= 0 for coefficient in coefficients):
        current = _linear_sum(repaired, names, coefficients)
        target = _linear_target(current, constraint)
        if target is None:
            return repaired, 0.0
        return repaired, abs(current - target)

    for _ in range(len(names) * 4 + 8):
        current = _linear_sum(repaired, names, coefficients)
        target = _linear_target(current, constraint)
        if target is None or abs(current - target) <= constraint.tolerance:
            break
        delta = target - current
        movable: list[tuple[str, float, float]] = []
        for name, coefficient in zip(names, coefficients):
            variable = space.variable(name)
            if variable.fixed_value is not None or str(variable.type.value) == "categorical":
                continue
            value = float(repaired[name])
            if delta > 0:
                room = (float(variable.upper) - value) * coefficient
            else:
                room = (value - float(variable.lower)) * coefficient
            if room > 1e-12:
                movable.append((name, coefficient, value))
        if not movable:
            break
        capacity = sum(
            (float(space.variable(name).upper) - value) * coefficient
            if delta > 0 else
            (value - float(space.variable(name).lower)) * coefficient
            for name, coefficient, value in movable
        )
        fraction = min(1.0, abs(delta) / capacity) if capacity > 0 else 1.0
        for name, coefficient, value in movable:
            variable = space.variable(name)
            room = (
                float(variable.upper) - value
                if delta > 0 else value - float(variable.lower)
            )
            change = (room * fraction) / coefficient
            new_value = value + change if delta > 0 else value - change
            if str(variable.type.value) == "integer":
                new_value = round(new_value)
            repaired[name] = _value_for_variable(space, variable, new_value)

    final = _linear_sum(repaired, names, coefficients)
    target = _linear_target(final, constraint)
    violation = 0.0 if target is None else abs(final - target)
    return repaired, float(violation)


def _linear_target(current: float, constraint: HardConstraintSpec) -> float | None:
    if constraint.constant is not None:
        return float(constraint.constant)
    if constraint.lower is not None and current < float(constraint.lower):
        return float(constraint.lower)
    if constraint.upper is not None and current > float(constraint.upper):
        return float(constraint.upper)
    return None


def _linear_sum(
    values: dict[str, Any],
    names: list[str],
    coefficients: list[float],
) -> float:
    return float(sum(
        coefficient * float(values[name])
        for name, coefficient in zip(names, coefficients)
    ))


def _repair_mutex(
    values: dict[str, Any],
    space: SearchSpace,
    constraint: HardConstraintSpec,
) -> tuple[dict[str, Any], float]:
    repaired = dict(values)
    active = _active_variables(repaired, space, constraint.variables)
    if len(active) <= 1:
        return repaired, 0.0
    numeric_active = [name for name in active if str(space.variable(name).type.value) != "categorical"]
    if numeric_active:
        keep = max(numeric_active, key=lambda name: float(repaired[name]))
        for name in numeric_active:
            if name == keep:
                continue
            variable = space.variable(name)
            if float(variable.lower) <= 0 <= float(variable.upper):
                repaired[name] = _value_for_variable(space, variable, 0.0)
    final_active = _active_variables(repaired, space, constraint.variables)
    violation = max(0, len(final_active) - 1)
    return repaired, float(violation)


def _active_variables(
    values: dict[str, Any],
    space: SearchSpace,
    names: list[str],
) -> list[str]:
    active: list[str] = []
    for name in names:
        variable = space.variable(name)
        value = values[name]
        if str(variable.type.value) == "categorical":
            if value not in {None, "", "__none__"}:
                active.append(name)
        elif abs(float(value)) > 1e-12:
            active.append(name)
    return active


def _value_for_variable(
    space: SearchSpace,
    variable: Any,
    numeric_value: float,
) -> Any:
    if str(variable.type.value) == "integer":
        return int(round(numeric_value))
    if str(variable.type.value) == "categorical":
        return space.original_value(variable, numeric_value)
    return float(numeric_value)


def _report(
    constraint: HardConstraintSpec,
    satisfied: bool,
    violation: float,
) -> dict[str, Any]:
    return {
        "name": constraint.name,
        "kind": constraint.kind,
        "satisfied": bool(satisfied and violation <= constraint.tolerance),
        "violation": float(violation),
        "tolerance": float(constraint.tolerance),
    }
