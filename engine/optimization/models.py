from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.exceptions import ArtifactError, ValidationError
from engine.modeling.registry import load_registered_model
from engine.optimization.contracts import ModelQualityGate, ObjectiveSpec, VariableSpec


@dataclass(frozen=True)
class OptimizationModel:
    target_name: str
    bundle: dict[str, Any]
    registry_entry: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.bundle["model_id"])

    @property
    def model_version(self) -> str:
        return str(self.bundle["version"])

    @property
    def metrics(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.bundle.get("metrics", {}).items()
        }

    def reference(self) -> np.ndarray:
        values = np.asarray(self.bundle.get("applicability_domain_reference", []))
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValidationError(
                f"model {self.model_id} is missing applicability-domain reference data"
            )
        return values

    def raw_training_matrix(self) -> np.ndarray:
        reference = self.reference()
        scaler = self.bundle["pipeline"].named_steps["scaler"]
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            return reference
        return reference * np.asarray(scaler.scale_) + np.asarray(scaler.mean_)

    def training_bounds(self, feature_name: str) -> tuple[float, float]:
        names = list(self.bundle["feature_names"])
        if feature_name not in names:
            raise ValidationError(f"feature {feature_name} is not in model {self.model_id}")
        column = self.raw_training_matrix()[:, names.index(feature_name)]
        return float(np.nanmin(column)), float(np.nanmax(column))

    def training_median(self, feature_name: str) -> float:
        names = list(self.bundle["feature_names"])
        if feature_name not in names:
            raise ValidationError(f"feature {feature_name} is not in model {self.model_id}")
        column = self.raw_training_matrix()[:, names.index(feature_name)]
        return float(np.nanmedian(column))


@dataclass(frozen=True)
class ModelSet:
    models: dict[str, OptimizationModel]

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for model in self.models.values():
            for feature_name in model.bundle["feature_names"]:
                if feature_name not in names:
                    names.append(feature_name)
        return names

    @property
    def sample_feature_ratio(self) -> float:
        ratios = [
            model.reference().shape[0] / max(len(model.bundle["feature_names"]), 1)
            for model in self.models.values()
        ]
        return float(min(ratios)) if ratios else 0.0

    def default_value(self, feature_name: str) -> float:
        values = [
            model.training_median(feature_name)
            for model in self.models.values()
            if feature_name in model.bundle["feature_names"]
        ]
        if not values:
            raise ValidationError(f"feature {feature_name} has no model default value")
        return float(np.median(values))

    def model_refs(self) -> list[dict[str, Any]]:
        return [
            {
                "target_name": model.target_name,
                "model_id": model.model_id,
                "model_version": model.model_version,
                "algorithm": model.bundle.get("algorithm"),
                "dataset_artifact_id": model.bundle.get("dataset_artifact_id"),
            }
            for model in self.models.values()
        ]


def load_optimization_models(
    *,
    registry_path: str,
    objectives: list[ObjectiveSpec],
    model_selection: Any,
) -> ModelSet:
    models: dict[str, OptimizationModel] = {}
    for objective in objectives:
        mapping = model_selection.target_mappings.get(objective.target_name, {})
        loaded = load_registered_model(
            registry_path,
            model_id=mapping.get("model_id") or objective.model_id,
            target_name=objective.target_name,
            dataset_artifact_id=mapping.get("dataset_artifact_id"),
            version=mapping.get("version") or objective.model_version,
        )
        if str(loaded.registry_entry.get("status", "CANDIDATE")).upper() == "DEPRECATED":
            raise ArtifactError(
                f"model for {objective.target_name} is deprecated and cannot be optimized"
            )
        metrics = loaded.bundle.get("metrics")
        if not metrics or float(metrics.get("cv_rmse_mean", 0)) <= 0:
            raise ValidationError(
                f"model for {objective.target_name} has no usable cv_rmse_mean"
            )
        if not loaded.bundle.get("applicability_domain"):
            raise ValidationError(
                f"model for {objective.target_name} is missing applicability domain"
            )
        models[objective.target_name] = OptimizationModel(
            target_name=objective.target_name,
            bundle=loaded.bundle,
            registry_entry=loaded.registry_entry,
        )
    dataset_ids = {
        str(model.bundle.get("dataset_artifact_id"))
        for model in models.values()
        if model.bundle.get("dataset_artifact_id") is not None
    }
    if len(dataset_ids) > 1:
        raise ValidationError(
            "optimization targets must use models from the same dataset artifact; "
            "provide explicit model_selection.target_mappings"
        )
    return ModelSet(models=models)


def evaluate_model_quality(
    model_set: ModelSet,
    quality_gate: ModelQualityGate,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for target_name, model in model_set.models.items():
        metrics = model.metrics
        failures: list[str] = []
        if (
            quality_gate.min_cv_r2 is not None
            and metrics.get("cv_r2_mean", float("inf")) < quality_gate.min_cv_r2
        ):
            failures.append("min_cv_r2")
        if (
            quality_gate.min_test_r2 is not None
            and metrics.get("r2", float("inf")) < quality_gate.min_test_r2
        ):
            failures.append("min_test_r2")
        if (
            quality_gate.max_cv_rmse is not None
            and metrics.get("cv_rmse_mean", float("-inf")) > quality_gate.max_cv_rmse
        ):
            failures.append("max_cv_rmse")
        if quality_gate.max_rmse_to_target_range_ratio is not None:
            target_values = [
                float(item["y_true"])
                for item in model.bundle.get("evaluation_records", [])
                if isinstance(item, dict) and "y_true" in item
            ]
            if target_values:
                target_range = max(target_values) - min(target_values)
                ratio = metrics.get("rmse", float("inf")) / max(target_range, 1e-12)
                if ratio > quality_gate.max_rmse_to_target_range_ratio:
                    failures.append("max_rmse_to_target_range_ratio")
        if failures:
            findings.append({
                "code": "LOW_MODEL_QUALITY",
                "target_name": target_name,
                "model_id": model.model_id,
                "failed_thresholds": failures,
                "mode": quality_gate.mode,
            })
    return findings


def default_variables(model_set: ModelSet) -> list[VariableSpec]:
    variables: list[VariableSpec] = []
    for feature_name in model_set.feature_names:
        bounds = []
        for model in model_set.models.values():
            if feature_name in model.bundle["feature_names"]:
                bounds.append(model.training_bounds(feature_name))
        lower = max(item[0] for item in bounds)
        upper = min(item[1] for item in bounds)
        allow_exploration = False
        if lower >= upper:
            lower = min(item[0] for item in bounds)
            upper = max(item[1] for item in bounds)
            allow_exploration = True
        variables.append(VariableSpec(
            name=feature_name,
            lower=lower,
            upper=upper,
            allow_exploration=allow_exploration,
        ))
    return variables
