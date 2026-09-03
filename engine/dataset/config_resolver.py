from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from engine.contracts import (
    CleaningConfig,
    ClosureConfig,
    LeakageConfig,
    QualityThresholdConfig,
    TestConsistencySpec,
)
from engine.exceptions import ValidationError


DEFAULT_PROFILE_NAME = "default_safe_v1"


class MetadataRequiredError(ValidationError):
    """Raised when a modeling dataset has no discoverable target metadata."""


@dataclass
class ResolvedPreprocessingConfig:
    target_fields: list[str]
    feature_fields: list[str]
    identifier_fields: list[str]
    thresholds: QualityThresholdConfig
    closure_config: ClosureConfig
    leakage_config: LeakageConfig
    consistency_specs: list[TestConsistencySpec]
    cleaning_config: CleaningConfig
    user_overrides: dict[str, Any]
    metadata_source: str
    resolution_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": DEFAULT_PROFILE_NAME,
            "target_fields": self.target_fields,
            "feature_fields": self.feature_fields,
            "identifier_fields": self.identifier_fields,
            "thresholds": vars(self.thresholds),
            "closure_config": vars(self.closure_config),
            "leakage_config": vars(self.leakage_config),
            "consistency_specs": [vars(item) for item in self.consistency_specs],
            "cleaning_config": vars(self.cleaning_config),
            "user_overrides": self.user_overrides,
            "metadata_source": self.metadata_source,
            "resolution_reasons": self.resolution_reasons,
        }


def resolve_preprocessing_config(
    dataframe: pd.DataFrame,
    *,
    user_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> ResolvedPreprocessingConfig:
    user = dict(user_config or {})
    meta = dict(metadata or {})
    if not meta and source_path is not None:
        meta = _load_sidecar_metadata(Path(source_path))

    metadata_source = "explicit"
    if not meta:
        metadata_source = "none"
    elif metadata is None and source_path is not None:
        metadata_source = "sidecar"

    target_fields = _strings(user.get("target_fields")) or _strings(
        meta.get("target_fields") or meta.get("target_cols")
    )
    if not target_fields:
        raise MetadataRequiredError(
            "no target fields found; provide target_fields or dataset metadata"
        )
    missing_targets = set(target_fields) - set(dataframe.columns)
    if missing_targets:
        raise ValidationError(f"target fields missing from dataset: {sorted(missing_targets)}")

    identifier_fields = _strings(user.get("identifier_fields")) or _strings(
        meta.get("identifier_fields") or meta.get("id_fields")
    )
    inferred_identifiers = [] if identifier_fields else _infer_identifier_fields(
        dataframe,
        target_fields,
    )
    identifier_fields = identifier_fields or inferred_identifiers
    missing_identifiers = set(identifier_fields) - set(dataframe.columns)
    if missing_identifiers:
        raise ValidationError(
            f"identifier fields missing from dataset: {sorted(missing_identifiers)}"
        )

    feature_fields = _strings(user.get("feature_fields")) or [
        column for column in dataframe.columns
        if column not in set(target_fields) | set(identifier_fields)
    ]
    missing_features = set(feature_fields) - set(dataframe.columns)
    if missing_features:
        raise ValidationError(
            f"feature fields missing from dataset: {sorted(missing_features)}"
        )

    reasons: list[str] = []
    if user.get("target_fields"):
        reasons.append("target fields came from user configuration")
    else:
        reasons.append("target fields came from dataset metadata")
    if user.get("identifier_fields"):
        reasons.append("identifier fields came from user configuration")
    elif identifier_fields:
        reasons.append("identifier fields were inferred from uniqueness and type")
    else:
        reasons.append("no identifier fields were configured or inferred")

    threshold_payload = dict(user.get("thresholds") or {})
    thresholds = QualityThresholdConfig.from_dict(threshold_payload)
    if threshold_payload:
        reasons.append("quality thresholds were overridden by user configuration")

    closure_payload = dict(user.get("closure") or {})
    closure_required = _strings(
        closure_payload.get("required_fields") or identifier_fields
    )
    closure_config = ClosureConfig(
        identifier_fields=identifier_fields,
        required_fields=closure_required,
        min_closure_ratio=float(
            closure_payload.get("min_closure_ratio", 0.95)
        ),
        max_ambiguous_identifier_count=int(
            closure_payload.get("max_ambiguous_identifier_count", 0)
        ),
    )
    if closure_payload:
        reasons.append("closure configuration was overridden by user configuration")

    leakage_payload = dict(user.get("leakage") or {})
    leakage_config = LeakageConfig(
        post_experiment_fields=_strings(
            leakage_payload.get("post_experiment_fields")
        ),
        forbidden_fields=_strings(leakage_payload.get("forbidden_fields")),
        target_derivation_correlation=float(
            leakage_payload.get("target_derivation_correlation", 0.999)
        ),
    )
    if leakage_payload:
        reasons.append("leakage configuration was overridden by user configuration")

    consistency_payload = list(user.get("consistency_specs") or [])
    consistency_specs = [TestConsistencySpec(**item) for item in consistency_payload]
    if consistency_specs:
        reasons.append("test consistency specifications were supplied by user")

    cleaning_payload = dict(user.get("cleaning") or {})
    drop_fields = _strings(cleaning_payload.get("drop_fields"))
    cleaning_config = CleaningConfig(
        drop_missing_target_rows=bool(
            cleaning_payload.get("drop_missing_target_rows", True)
        ),
        drop_exact_duplicates=bool(
            cleaning_payload.get("drop_exact_duplicates", True)
        ),
        drop_fields=drop_fields,
        drop_high_missing_fields=bool(
            cleaning_payload.get("drop_high_missing_fields", True)
        ),
        max_feature_missing_ratio=float(
            cleaning_payload.get("max_feature_missing_ratio", 0.50)
        ),
        impute_missing_features=bool(
            cleaning_payload.get("impute_missing_features", True)
        ),
        add_missing_indicators=bool(
            cleaning_payload.get("add_missing_indicators", True)
        ),
        outlier_strategy=str(cleaning_payload.get("outlier_strategy", "mark")),
    )
    cleaning_config.validate()
    if cleaning_payload:
        reasons.append("cleaning strategy was overridden by user configuration")

    return ResolvedPreprocessingConfig(
        target_fields=target_fields,
        feature_fields=feature_fields,
        identifier_fields=identifier_fields,
        thresholds=thresholds,
        closure_config=closure_config,
        leakage_config=leakage_config,
        consistency_specs=consistency_specs,
        cleaning_config=cleaning_config,
        user_overrides=user,
        metadata_source=metadata_source,
        resolution_reasons=reasons,
    )


def _load_sidecar_metadata(source: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    if source.name:
        candidates.extend([
            source.with_name(f"{source.stem}.metadata.json"),
            source.with_name("metadata.json"),
        ])
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        target_fields = _strings(payload.get("target_fields") or payload.get("target_cols"))
        if target_fields:
            return payload
    return {}


def _infer_identifier_fields(
    dataframe: pd.DataFrame,
    target_fields: list[str],
) -> list[str]:
    candidates: list[str] = []
    for column in dataframe.columns:
        if column in set(target_fields):
            continue
        series = dataframe[column]
        uniqueness = series.nunique(dropna=False) / len(series) if len(series) else 0
        if uniqueness < 0.95:
            continue
        lowered = column.lower()
        looks_like_id = lowered.endswith("id") or lowered.endswith("_id")
        object_like = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        integer_like = pd.api.types.is_integer_dtype(series)
        if looks_like_id or object_like or integer_like:
            candidates.append(column)
    return candidates[:1]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]
