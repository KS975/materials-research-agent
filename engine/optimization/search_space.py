from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.exceptions import ValidationError
from engine.optimization.contracts import VariableSpec, VariableType
from engine.optimization.models import ModelSet, default_variables


@dataclass(frozen=True)
class SearchSpace:
    variables: list[VariableSpec]
    defaults: dict[str, float]
    model_feature_names: list[str]

    @property
    def free_variables(self) -> list[VariableSpec]:
        return [item for item in self.variables if item.fixed_value is None]

    def variable(self, name: str) -> VariableSpec:
        for item in self.variables:
            if item.name == name:
                return item
        raise ValidationError(f"unknown search variable: {name}")

    def encoded_value(self, variable: VariableSpec, value: Any) -> float:
        if variable.type is VariableType.categorical:
            try:
                return float(variable.categories.index(value))
            except ValueError as exc:
                raise ValidationError(
                    f"unknown category {value!r} for variable {variable.name}"
                ) from exc
        return float(value)

    def original_value(self, variable: VariableSpec, encoded: float) -> Any:
        if variable.type is VariableType.categorical:
            index = int(round(encoded))
            if index < 0 or index >= len(variable.categories or []):
                raise ValidationError(f"encoded category is out of range: {encoded}")
            return variable.categories[index]
        return encoded

    def to_model_frame_rows(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            row: dict[str, Any] = {}
            for feature_name in self.model_feature_names:
                if feature_name in candidate:
                    row[feature_name] = self.encoded_value(
                        self.variable(feature_name), candidate[feature_name]
                    )
                else:
                    row[feature_name] = self.defaults.get(
                        feature_name, self.defaults.get(feature_name, 0.0)
                    )
            rows.append(row)
        return rows


def build_search_space(
    variables: list[VariableSpec] | None,
    model_set: ModelSet,
) -> SearchSpace:
    source = variables if variables else default_variables(model_set)
    names = [item.name for item in source]
    if len(names) != len(set(names)):
        raise ValidationError("search variable names must be unique")
    missing = set(model_set.feature_names) - set(names)
    if missing:
        raise ValidationError(
            f"search space missing model features: {sorted(missing)}"
        )
    resolved: list[VariableSpec] = []
    defaults = {
        feature_name: model_set.default_value(feature_name)
        for feature_name in model_set.feature_names
    }
    for variable in source:
        if variable.name not in model_set.feature_names:
            raise ValidationError(
                f"search variable {variable.name} is not used by selected models"
            )
        if variable.type in {VariableType.continuous, VariableType.integer}:
            if not variable.allow_exploration:
                bounds = []
                for model in model_set.models.values():
                    if variable.name in model.bundle["feature_names"]:
                        bounds.append(model.training_bounds(variable.name))
                training_lower = max(item[0] for item in bounds)
                training_upper = min(item[1] for item in bounds)
                lower = max(float(variable.lower), training_lower)
                upper = min(float(variable.upper), training_upper)
                if lower >= upper:
                    if variable.allow_exploration:
                        lower, upper = float(variable.lower), float(variable.upper)
                    else:
                        raise ValidationError(
                            f"search bounds for {variable.name} have an empty "
                            "intersection with model training bounds"
                        )
            else:
                lower, upper = float(variable.lower), float(variable.upper)
            if variable.type is VariableType.integer:
                lower = float(np.ceil(lower))
                upper = float(np.floor(upper))
                if lower >= upper:
                    raise ValidationError(
                        f"integer variable {variable.name} has no valid integer"
                    )
            resolved.append(_replace(variable, lower=lower, upper=upper))
        else:
            resolved.append(variable)
    return SearchSpace(
        variables=resolved,
        defaults=defaults,
        model_feature_names=list(model_set.feature_names),
    )


def _replace(variable: VariableSpec, **changes: Any) -> VariableSpec:
    payload = variable.__dict__.copy()
    payload.update(changes)
    return VariableSpec(**payload)
