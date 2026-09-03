from __future__ import annotations

from inspect import signature
from typing import Any

import numpy as np
import pandas as pd

from engine.contracts import PredictionResult
from engine.exceptions import ValidationError


def predict_with_model(
    bundle: dict[str, Any],
    dataframe: pd.DataFrame,
) -> list[PredictionResult]:
    """Predict with model-version metadata and standardized kNN applicability domain."""
    feature_names = list(bundle["feature_names"])
    missing = set(feature_names) - set(dataframe.columns)
    if missing:
        raise ValidationError(f"prediction inputs missing features: {sorted(missing)}")
    inputs = dataframe.loc[:, feature_names].copy()
    if inputs.isna().any().any():
        invalid_fields = inputs.columns[inputs.isna().any()].tolist()
        raise ValidationError(f"prediction inputs contain missing values: {invalid_fields}")
    non_numeric = [
        column for column in feature_names
        if not pd.api.types.is_numeric_dtype(inputs[column])
    ]
    if non_numeric:
        raise ValidationError(
            f"prediction features must be numeric in current engine version: {non_numeric}"
        )

    pipeline = bundle["pipeline"]
    transform = str(bundle.get("target_transform", "none"))
    estimator = pipeline.named_steps["estimator"]
    scaler = pipeline.named_steps["scaler"]
    standardized = scaler.transform(inputs)
    uncertainty: float | None = None
    if "return_std" in signature(estimator.predict).parameters:
        transformed_prediction, transformed_std = estimator.predict(
            standardized, return_std=True
        )
        uncertainty = _inverse_uncertainty(
            transformed_prediction, np.asarray(transformed_std), transform
        )
    else:
        transformed_prediction = pipeline.predict(inputs)
    prediction = _inverse_transform(transformed_prediction, transform)
    domains, domain_warnings = _applicability_domains(bundle, standardized)

    results: list[PredictionResult] = []
    for row_number, (_, row) in enumerate(inputs.iterrows()):
        row_prediction = float(prediction[row_number])
        row_uncertainty = (
            float(uncertainty[row_number]) if uncertainty is not None else None
        )
        results.append(PredictionResult(
            model_id=str(bundle["model_id"]),
            model_version=str(bundle["version"]),
            target_name=str(bundle["target_name"]),
            input_values={key: row[key] for key in feature_names},
            predicted_value=row_prediction,
            prediction_uncertainty=row_uncertainty,
            applicability_domain=domains[row_number],
            warnings=domain_warnings[row_number],
        ))
    return results


def _applicability_domains(
    bundle: dict[str, Any],
    standardized_inputs: np.ndarray,
) -> tuple[list[str], list[list[str]]]:
    record = bundle.get("applicability_domain", {})
    reference = np.asarray(bundle.get("applicability_domain_reference", []))
    k = int(record.get("k_neighbors", 5))
    q75 = float(record.get("distance_q75", 0.0))
    q95 = float(record.get("distance_q95", 0.0))
    if reference.size == 0:
        raise ValidationError("model bundle is missing applicability-domain reference data")
    k = min(k, len(reference))
    distances = np.stack([
        np.sqrt(((reference - candidate) ** 2).sum(axis=1))
        for candidate in standardized_inputs
    ])
    mean_distances = np.sort(distances, axis=1)[:, :k].mean(axis=1)
    domains: list[str] = []
    warnings: list[list[str]] = []
    for distance in mean_distances:
        if distance <= q75:
            domain = "IN_DOMAIN"
        elif distance <= q95:
            domain = "EDGE"
        else:
            domain = "OUT_OF_DOMAIN"
        domain_warnings = [] if domain == "IN_DOMAIN" else [
            f"applicability domain is {domain}"
        ]
        domains.append(domain)
        warnings.append(domain_warnings)
    return domains, warnings


def _inverse_transform(values: Any, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(values)
    if transform != "none":
        raise ValidationError(f"unsupported target transform: {transform}")
    return np.asarray(values)


def _inverse_uncertainty(
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    transform: str,
) -> np.ndarray:
    if transform == "none":
        return uncertainty
    if transform == "log1p":
        return np.abs(np.expm1(prediction + uncertainty) - np.expm1(prediction))
    raise ValidationError(f"unsupported target transform: {transform}")
