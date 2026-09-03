from __future__ import annotations

from engine.contracts import (
    ALLOWED_MODELING_ALGORITHMS,
    ModelingStrategyRecord,
    TrainingConfig,
)
from engine.exceptions import ValidationError


TIER_ALGORITHMS: dict[int, tuple[str, ...]] = {
    1: (
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "bayesian_ridge",
        "pls",
    ),
    2: (
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "bayesian_ridge",
        "pls",
        "gaussian_process",
        "random_forest",
        "xgboost",
    ),
    3: (
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "bayesian_ridge",
        "pls",
        "gaussian_process",
        "random_forest",
        "xgboost",
        "lightgbm",
        "gradient_boosting",
        "svr",
    ),
}


def select_modeling_strategy(
    *,
    target_name: str,
    sample_count: int,
    feature_count: int,
    config: TrainingConfig,
    user_overrides: dict | None = None,
) -> ModelingStrategyRecord:
    config.validate()
    if sample_count <= 0 or feature_count <= 0:
        raise ValidationError("sample_count and feature_count must be positive")
    ratio = sample_count / feature_count
    tier = 1 if ratio < 1 else 2 if ratio < 3 else 3
    cv_mode = config.cv_mode
    if cv_mode == "auto":
        cv_mode = "loocv" if ratio < 1 else "kfold"

    preset = list(TIER_ALGORITHMS[tier])
    selected = list(config.algorithms) if config.algorithms else preset
    selected = [
        algorithm for algorithm in selected
        if algorithm not in set(config.disabled_algorithms)
    ]
    unknown = set(selected) - ALLOWED_MODELING_ALGORITHMS
    if unknown:
        raise ValidationError(f"unknown modeling algorithms: {sorted(unknown)}")
    if not selected:
        raise ValidationError("no modeling algorithms remain after filtering")

    if config.algorithms:
        reason = (
            f"ratio={ratio:.3f} suggests Tier {tier}; "
            "user explicitly supplied the algorithm set"
        )
    else:
        reason = (
            f"ratio={ratio:.3f} selected Tier {tier}; "
            f"default_modeling_v1 uses {cv_mode} and the Tier {tier} algorithm set"
        )

    return ModelingStrategyRecord(
        stage="model_training.strategy",
        target_name=target_name,
        profile_name=config.profile_name,
        tier=tier,
        cv_mode=cv_mode,
        selected_algorithms=selected,
        sample_count=sample_count,
        feature_count=feature_count,
        sample_feature_ratio=ratio,
        strategy_reason=reason,
        user_overrides=dict(user_overrides or {}),
    )
