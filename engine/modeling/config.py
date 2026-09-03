from __future__ import annotations

from typing import Any

import pandas as pd

from engine.contracts import TrainingConfig
from engine.exceptions import ValidationError


CONFIG_KEYS = {
    "target_names",
    "feature_names",
    "reserved_test_ratio",
    "cv_mode",
    "random_seed",
    "algorithms",
    "disabled_algorithms",
    "optuna_enabled",
    "optuna_trials",
    "transform_config",
    "primary_metric",
    "profile_name",
}


def resolve_training_config(
    dataframe: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
    user_config: dict[str, Any] | None = None,
) -> tuple[TrainingConfig, dict[str, Any]]:
    """Resolve preset modeling fields with explicit user overrides."""
    meta = dict(metadata or {})
    user = dict(user_config or {})
    aliases = {"target_fields": "target_names", "feature_fields": "feature_names"}
    normalized: dict[str, Any] = {}
    for key, value in user.items():
        normalized[aliases.get(key, key)] = value
    unknown = set(normalized) - CONFIG_KEYS
    if unknown:
        raise ValidationError(f"unknown training config keys: {sorted(unknown)}")

    target_names = _strings(normalized.get("target_names")) or _strings(
        meta.get("target_fields") or meta.get("target_cols")
    )
    if not target_names:
        raise ValidationError(
            "no target fields found; provide target_names or dataset metadata"
        )
    missing_targets = set(target_names) - set(dataframe.columns)
    if missing_targets:
        raise ValidationError(
            f"target fields missing from dataset: {sorted(missing_targets)}"
        )

    feature_names = _strings(normalized.get("feature_names")) or _strings(
        meta.get("feature_fields")
    )
    if not feature_names:
        feature_names = [
            column for column in dataframe.columns
            if column not in set(target_names)
        ]
    missing_features = set(feature_names) - set(dataframe.columns)
    if missing_features:
        raise ValidationError(
            f"feature fields missing from dataset: {sorted(missing_features)}"
        )
    overlap = set(target_names) & set(feature_names)
    if overlap:
        raise ValidationError(
            f"target and feature fields overlap: {sorted(overlap)}"
        )

    config = TrainingConfig(
        target_names=target_names,
        feature_names=feature_names,
        reserved_test_ratio=float(normalized.get("reserved_test_ratio", 0.2)),
        cv_mode=str(normalized.get("cv_mode", "auto")),
        random_seed=int(normalized.get("random_seed", 42)),
        algorithms=(
            _strings(normalized.get("algorithms"))
            if normalized.get("algorithms") is not None
            else None
        ),
        disabled_algorithms=_strings(normalized.get("disabled_algorithms", [])),
        optuna_enabled=bool(normalized.get("optuna_enabled", False)),
        optuna_trials=normalized.get("optuna_trials"),
        transform_config=dict(normalized.get("transform_config", {})),
        primary_metric=str(normalized.get("primary_metric", "cv_rmse_mean")),
        profile_name=str(normalized.get("profile_name", "default_modeling_v1")),
    )
    if config.optuna_trials is not None:
        config.optuna_trials = int(config.optuna_trials)
    config.validate()
    return config, {
        "profile_name": config.profile_name,
        "metadata_source": "dataset_artifact" if not normalized.get("target_names") else "user_config",
        "target_names": config.target_names,
        "feature_names": config.feature_names,
        "user_overrides": normalized,
    }


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]
