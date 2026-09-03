from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FieldRole(str, Enum):
    identifier = "identifier"
    feature = "feature"
    target = "target"
    group = "group"
    timestamp = "timestamp"
    condition = "condition"
    ignored = "ignored"


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class GateDecision(str, Enum):
    passed = "PASS"
    conditional = "CONDITIONAL_PASS"
    failed = "FAIL"


@dataclass(frozen=True)
class FieldMetadata:
    name: str
    role: FieldRole
    dtype: str | None = None
    unit: str | None = None
    allowed_values: list[str] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    is_post_experiment: bool = False


@dataclass(frozen=True)
class QualityFinding:
    check: str
    severity: Severity
    affected_fields: list[str]
    affected_rows: list[Any] = field(default_factory=list)
    metric_value: float | None = None
    threshold: float | None = None
    reason: str = ""
    suggestion: str = ""
    suggested_action: str | None = None
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass(frozen=True)
class PreprocessingRuleRecord:
    """Traceable record of one preprocessing rule and its result."""

    stage: str
    rule_name: str
    rule_version: str
    rule_definition: dict[str, Any]
    input_scope: list[str]
    status: str
    summary_text: str
    findings: list[QualityFinding] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "preprocessing_rule",
            "stage": self.stage,
            "rule_name": self.rule_name,
            "rule_version": self.rule_version,
            "rule_definition": self.rule_definition,
            "input_scope": self.input_scope,
            "status": self.status,
            "summary_text": self.summary_text,
            "findings": [finding.to_dict() for finding in self.findings],
            "details": self.details,
        }


@dataclass(frozen=True)
class QualityThresholdConfig:
    min_total_samples: int = 10
    min_samples_per_target: int = 10
    min_feature_count: int = 1
    min_target_count: int = 1
    max_target_missing_ratio: float = 0.20
    max_feature_missing_ratio: float = 0.50
    max_duplicate_ratio: float = 0.10
    max_target_conflict_count: int = 0
    max_global_outlier_ratio: float = 0.10
    max_single_feature_outlier_ratio: float = 0.10
    min_sample_feature_ratio: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> QualityThresholdConfig:
        if not payload:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        unknown = set(payload) - set(allowed)
        if unknown:
            raise ValueError(f"unknown quality thresholds: {sorted(unknown)}")
        return cls(**payload)


@dataclass(frozen=True)
class ClosureConfig:
    identifier_fields: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    min_closure_ratio: float = 0.95
    max_ambiguous_identifier_count: int = 0


@dataclass(frozen=True)
class TestConsistencySpec:
    target_field: str
    test_field: str | None = None
    expected_test: str | None = None
    unit_field: str | None = None
    expected_unit: str | None = None
    method_field: str | None = None
    expected_method: str | None = None
    required_condition_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LeakageConfig:
    post_experiment_fields: list[str] = field(default_factory=list)
    forbidden_fields: list[str] = field(default_factory=list)
    target_derivation_correlation: float = 0.999


@dataclass(frozen=True)
class CleaningConfig:
    drop_missing_target_rows: bool = True
    drop_exact_duplicates: bool = True
    drop_fields: list[str] = field(default_factory=list)
    drop_high_missing_fields: bool = True
    max_feature_missing_ratio: float = 0.50
    impute_missing_features: bool = True
    impute_numeric_fields: list[str] = field(default_factory=list)
    impute_categorical_fields: list[str] = field(default_factory=list)
    add_missing_indicators: bool = True
    missing_indicator_fields: list[str] = field(default_factory=list)
    winsorize_numeric_outliers: bool = False
    outlier_strategy: str = "mark"
    profile_name: str = "default_safe_v1"

    def validate(self) -> None:
        if self.outlier_strategy not in {"mark", "winsorize"}:
            raise ValueError(f"unsupported outlier strategy: {self.outlier_strategy}")
        if not 0 < self.max_feature_missing_ratio <= 1:
            raise ValueError("max_feature_missing_ratio must be in (0, 1]")


@dataclass
class DataQualityReport:
    row_count: int
    feature_count: int
    target_count: int
    sample_feature_ratio: float
    findings: list[QualityFinding]
    details: dict[str, Any] = field(default_factory=dict)
    rule_records: list[PreprocessingRuleRecord] = field(default_factory=list)

    @property
    def max_severity(self) -> Severity:
        if any(item.severity is Severity.error for item in self.findings):
            return Severity.error
        if any(item.severity is Severity.warning for item in self.findings):
            return Severity.warning
        return Severity.info

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "target_count": self.target_count,
            "sample_feature_ratio": self.sample_feature_ratio,
            "max_severity": self.max_severity.value,
            "findings": [item.to_dict() for item in self.findings],
            "details": self.details,
            "rule_records": [record.to_dict() for record in self.rule_records],
            "technical_summary": self.technical_summary(),
        }

    def technical_summary(self) -> list[dict[str, Any]]:
        """Derive user-safe stage messages from rule records, not raw internals."""
        return [
            {
                "stage": record.stage,
                "rule_name": record.rule_name,
                "severity": record.status,
                "message": record.summary_text,
            }
            for record in self.rule_records
        ]


@dataclass(frozen=True)
class ModelingGateResult:
    decision: GateDecision
    reasons: list[str]
    recommended_tier: int
    blocking_items: list[str]
    warning_items: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "recommended_tier": self.recommended_tier,
            "blocking_items": self.blocking_items,
            "warning_items": self.warning_items,
        }


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_id: str
    version: str
    artifact_dir: str
    file_path: str
    source_uri: str
    source_hash: str
    data_hash: str
    parent_dataset_id: str | None
    feature_fields: list[str]
    target_fields: list[str]
    identifier_fields: list[str]
    cleaning_config: CleaningConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cleaning_config"] = asdict(self.cleaning_config)
        return payload


@dataclass(frozen=True)
class CleaningStep:
    action: str
    fields: list[str]
    reason: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningExecutionReport:
    input_row_count: int
    output_row_count: int
    input_column_count: int
    output_column_count: int
    dropped_duplicate_count: int
    dropped_missing_target_count: int
    removed_fields: list[str]
    added_fields: list[str]
    imputed_fields: list[str]
    imputed_value_count: int
    winsorized_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleaningOperationRecord:
    """Traceable record of one executed cleaning operation."""

    stage: str
    operation: str
    source: str
    status: str
    parameters: dict[str, Any]
    affected_fields: list[str]
    affected_row_count: int
    affected_row_indices: list[Any]
    input_count: int
    output_count: int
    summary_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "cleaning_operation",
            **asdict(self),
        }


@dataclass(frozen=True)
class GateDecisionRecord:
    stage: str
    decision: GateDecision
    summary_text: str
    blocking_items: list[str]
    warning_items: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "modeling_gate",
            "stage": self.stage,
            "decision": self.decision.value,
            "summary_text": self.summary_text,
            "blocking_items": self.blocking_items,
            "warning_items": self.warning_items,
        }


@dataclass(frozen=True)
class TrainingConfig:
    target_names: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    reserved_test_ratio: float = 0.2
    cv_mode: str = "auto"
    random_seed: int = 42
    algorithms: list[str] | None = None
    disabled_algorithms: list[str] = field(default_factory=list)
    optuna_enabled: bool = False
    optuna_trials: int | None = None
    transform_config: dict[str, str] = field(default_factory=dict)
    primary_metric: str = "cv_rmse_mean"
    profile_name: str = "default_modeling_v1"

    def validate(self) -> None:
        if not self.target_names:
            raise ValueError("target_names must not be empty")
        if not 0 < self.reserved_test_ratio < 1:
            raise ValueError("reserved_test_ratio must be in (0, 1)")
        if self.cv_mode not in {"auto", "loocv", "kfold"}:
            raise ValueError(f"unsupported cv_mode: {self.cv_mode}")
        if self.primary_metric != "cv_rmse_mean":
            raise ValueError("primary_metric currently supports cv_rmse_mean")
        unknown_disabled = set(self.disabled_algorithms) - ALLOWED_MODELING_ALGORITHMS
        if unknown_disabled:
            raise ValueError(
                f"unknown disabled algorithms: {sorted(unknown_disabled)}"
            )


ALLOWED_MODELING_ALGORITHMS = frozenset({
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
})


@dataclass(frozen=True)
class ModelingStrategyRecord:
    stage: str
    target_name: str
    profile_name: str
    tier: int
    cv_mode: str
    selected_algorithms: list[str]
    sample_count: int
    feature_count: int
    sample_feature_ratio: float
    strategy_reason: str
    user_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "modeling_strategy",
            **asdict(self),
        }


@dataclass(frozen=True)
class CandidateMetrics:
    r2: float
    mae: float
    rmse: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_r2_mean: float
    cv_r2_std: float
    cv_fold_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateTrainingRecord:
    target_name: str
    algorithm: str
    status: str
    metrics: CandidateMetrics | None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    interpretability: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: int = 0
    selection_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "candidate_model",
            "target_name": self.target_name,
            "algorithm": self.algorithm,
            "status": self.status,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "hyperparameters": self.hyperparameters,
            "interpretability": self.interpretability,
            "warnings": self.warnings,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "selection_rank": self.selection_rank,
        }


@dataclass(frozen=True)
class ApplicabilityDomainRecord:
    method: str
    k_neighbors: int
    distance_q75: float
    distance_q95: float
    training_sample_count: int
    feature_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelArtifactRecord:
    model_id: str
    version: str
    target_name: str
    algorithm: str
    dataset_artifact_id: str | None
    dataset_data_hash: str
    artifact_dir: str
    file_path: str
    status: str
    metrics: CandidateMetrics
    applicability_domain: ApplicabilityDomainRecord
    feature_names: list[str]
    target_transform: str
    created_at: str
    evaluation_records: list[EvaluationSampleRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        payload["applicability_domain"] = self.applicability_domain.to_dict()
        return payload


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    model_version: str
    target_name: str
    input_values: dict[str, Any]
    predicted_value: float
    prediction_uncertainty: float | None
    applicability_domain: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "prediction",
            **asdict(self),
        }


@dataclass(frozen=True)
class EvaluationSampleRecord:
    """Reserved-test observation exposed for frontend diagnostics."""

    row_index: int
    target_name: str
    y_true: float
    y_pred: float
    residual: float
    model_id: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "evaluation_sample",
            **asdict(self),
        }


@dataclass(frozen=True)
class VisualizationColumn:
    name: str
    label: str
    data_type: str = "auto"
    format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualizationDataset:
    """UI-neutral data contract; rendering is owned by the host frontend."""

    dataset_id: str
    dataset_kind: str
    title: str
    columns: list[VisualizationColumn]
    records: list[dict[str, Any]]
    description: str = ""
    chart_type: str | None = None
    x_field: str | None = None
    y_fields: list[str] = field(default_factory=list)
    series_field: str | None = None
    source_artifact: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        valid_kinds = {"table", "chart", "chart_with_table"}
        valid_charts = {
            "bar", "horizontal_bar", "scatter", "line",
            "heatmap", "parallel_coordinates",
        }
        if self.dataset_kind not in valid_kinds:
            raise ValueError(f"unsupported visualization dataset_kind: {self.dataset_kind}")
        if self.dataset_kind == "table":
            if self.chart_type is not None:
                raise ValueError("a table visualization must not define chart_type")
        else:
            if self.chart_type not in valid_charts:
                raise ValueError(
                    f"unsupported visualization chart_type: {self.chart_type}"
                )
            if not self.x_field:
                raise ValueError("chart visualizations require x_field")
            if not self.y_fields:
                raise ValueError("chart visualizations require y_fields")
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("visualization columns must be unique")
        record_keys: set[str] = set()
        for record in self.records:
            record_keys.update(record)
        missing = set(names) - record_keys
        if missing:
            raise ValueError(f"records missing visualization columns: {sorted(missing)}")
        chart_fields = {self.x_field, *self.y_fields, self.series_field}
        missing_chart_fields = {
            field for field in chart_fields if field is not None
        } - set(names)
        if missing_chart_fields:
            raise ValueError(
                "columns missing chart fields: "
                f"{sorted(missing_chart_fields)}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "record_type": "visualization_dataset",
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_kind": self.dataset_kind,
            "chart_type": self.chart_type,
            "title": self.title,
            "description": self.description,
            "columns": [column.to_dict() for column in self.columns],
            "records": self.records,
            "x_field": self.x_field,
            "y_fields": self.y_fields,
            "series_field": self.series_field,
            "source_artifact": self.source_artifact,
        }


@dataclass(frozen=True)
class VisualizationBundle:
    source_kind: str
    source_uri: str
    datasets: list[VisualizationDataset]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "visualization_bundle",
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "source_uri": self.source_uri,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }
