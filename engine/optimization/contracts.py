from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from engine.exceptions import ValidationError


class VariableType(str, Enum):
    continuous = "continuous"
    integer = "integer"
    categorical = "categorical"


class ObjectiveOperator(str, Enum):
    equal = "equal"
    greater_or_equal = "greater_or_equal"
    less_or_equal = "less_or_equal"
    in_range = "in_range"
    maximize = "maximize"
    minimize = "minimize"


class ObjectiveRequirement(str, Enum):
    hard = "hard"
    preferred = "preferred"


class SoftRankingPolicy(str, Enum):
    tie_breaker = "tie_breaker"
    additional_objective = "additional_objective"
    prefilter = "prefilter"


@dataclass(frozen=True)
class StrategyThresholds:
    candidate_rank_max_count: int = 5000
    candidate_rank_max_free_dimensions: int = 3
    candidate_rank_max_points_per_continuous_dimension: int = 100
    de_rag_min_history_count: int = 5
    de_rag_min_active_dimension: int = 3
    de_rag_min_coverage_ratio: float = 0.20
    active_set_min_search_dimension: int = 20
    active_set_stage1_top_ratio: float = 0.20
    mixed_nsga2_categorical_pool_multiplier: int = 3
    soft_conflict_zero_discrimination_tolerance: float = 1e-6

    def validate(self) -> None:
        positive_integers = [
            self.candidate_rank_max_count,
            self.candidate_rank_max_free_dimensions,
            self.candidate_rank_max_points_per_continuous_dimension,
            self.de_rag_min_history_count,
            self.de_rag_min_active_dimension,
            self.active_set_min_search_dimension,
            self.mixed_nsga2_categorical_pool_multiplier,
        ]
        if any(value <= 0 for value in positive_integers):
            raise ValidationError("integer strategy thresholds must be positive")
        ratios = [
            self.de_rag_min_coverage_ratio,
            self.active_set_stage1_top_ratio,
        ]
        if any(not 0 < value <= 1 for value in ratios):
            raise ValidationError("ratio strategy thresholds must be in (0, 1]")
        if self.soft_conflict_zero_discrimination_tolerance <= 0:
            raise ValidationError(
                "soft_conflict_zero_discrimination_tolerance must be positive"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> StrategyThresholds:
        if not payload:
            return cls()
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValidationError(f"unknown strategy thresholds: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveSpec:
    target_name: str
    operator: ObjectiveOperator
    value: float | None = None
    lower_value: float | None = None
    upper_value: float | None = None
    tolerance: float = 1e-9
    weight: float = 1.0
    requirement: ObjectiveRequirement = ObjectiveRequirement.preferred
    model_id: str | None = None
    model_version: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operator", ObjectiveOperator(self.operator)
        )
        object.__setattr__(
            self,
            "requirement",
            ObjectiveRequirement(self.requirement),
        )

    def validate(self) -> None:
        if self.weight < 0:
            raise ValidationError("objective weight must be non-negative")
        if self.tolerance < 0:
            raise ValidationError("objective tolerance must be non-negative")
        if self.operator in {ObjectiveOperator.equal, ObjectiveOperator.greater_or_equal,
                             ObjectiveOperator.less_or_equal} and self.value is None:
            raise ValidationError(f"objective {self.target_name} requires value")
        if self.operator is ObjectiveOperator.in_range:
            if self.lower_value is None or self.upper_value is None:
                raise ValidationError(f"objective {self.target_name} requires bounds")
            if self.lower_value >= self.upper_value:
                raise ValidationError(f"objective {self.target_name} has invalid bounds")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObjectiveSpec:
        data = dict(payload)
        data["operator"] = ObjectiveOperator(data["operator"])
        data["requirement"] = ObjectiveRequirement(data.get("requirement", "preferred"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["operator"] = self.operator.value
        payload["requirement"] = self.requirement.value
        return payload


@dataclass(frozen=True)
class VariableSpec:
    name: str
    type: VariableType = VariableType.continuous
    role: str = "other"
    lower: float | None = None
    upper: float | None = None
    categories: list[Any] | None = None
    unit: str | None = None
    min_effective_value: float | None = None
    fixed_value: Any = None
    allow_exploration: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", VariableType(self.type))

    def validate(self) -> None:
        if self.type in {VariableType.continuous, VariableType.integer}:
            if self.lower is None or self.upper is None:
                raise ValidationError(f"variable {self.name} requires bounds")
            if self.lower >= self.upper:
                raise ValidationError(f"variable {self.name} has invalid bounds")
        if self.type is VariableType.categorical:
            if not self.categories:
                raise ValidationError(f"variable {self.name} requires categories")
            if len(set(map(str, self.categories))) != len(self.categories):
                raise ValidationError(f"variable {self.name} has duplicate categories")
        if self.fixed_value is not None:
            if self.type is VariableType.categorical:
                if self.fixed_value not in (self.categories or []):
                    raise ValidationError(f"variable {self.name} has invalid fixed_value")
            elif self.lower is not None and self.upper is not None:
                if not self.lower <= float(self.fixed_value) <= self.upper:
                    raise ValidationError(f"variable {self.name} fixed_value is out of bounds")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VariableSpec:
        data = dict(payload)
        data["type"] = VariableType(data.get("type", "continuous"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload


@dataclass(frozen=True)
class HardConstraintSpec:
    name: str
    kind: str
    variables: list[str] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    constant: float | None = None
    lower: float | None = None
    upper: float | None = None
    target_name: str | None = None
    operator: str | None = None
    tolerance: float = 1e-9

    def validate(self) -> None:
        supported = {
            "linear_sum", "bound", "mutex", "categorical_incompatibility",
            "target_threshold",
        }
        if self.kind not in supported:
            raise ValidationError(f"unsupported hard constraint kind: {self.kind}")
        if self.kind == "linear_sum":
            if not self.variables:
                raise ValidationError(f"constraint {self.name} requires variables")
            if self.coefficients and len(self.coefficients) != len(self.variables):
                raise ValidationError(f"constraint {self.name} coefficient count mismatch")
            if self.constant is None and self.lower is None and self.upper is None:
                raise ValidationError(f"constraint {self.name} requires a sum limit")
        elif self.kind == "bound":
            if len(self.variables) != 1 or (self.lower is None and self.upper is None):
                raise ValidationError(f"constraint {self.name} requires one variable and a bound")
        elif self.kind in {"mutex", "categorical_incompatibility"}:
            if len(self.variables) < 2:
                raise ValidationError(f"constraint {self.name} requires at least two variables")
        elif self.kind == "target_threshold":
            if not self.target_name or self.operator not in {
                "greater_or_equal", "less_or_equal"
            } or self.constant is None:
                raise ValidationError(f"constraint {self.name} is incomplete")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardConstraintSpec:
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationSpec:
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class SoftConstraintSpec:
    name: str
    kind: str
    variables: list[str] = field(default_factory=list)
    weight: float = 1.0
    ranking_policy: SoftRankingPolicy = SoftRankingPolicy.tie_breaker
    filter_threshold: float | None = None
    normalization: NormalizationSpec | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ranking_policy",
            SoftRankingPolicy(self.ranking_policy),
        )

    def validate(self) -> None:
        supported = {
            "minimize_expression", "maximize_expression", "history_distance",
            "process_stability", "custom",
        }
        if self.kind not in supported:
            raise ValidationError(f"unsupported soft constraint kind: {self.kind}")
        if self.weight < 0:
            raise ValidationError(f"soft constraint {self.name} weight must be non-negative")
        if self.ranking_policy is SoftRankingPolicy.prefilter:
            if self.filter_threshold is None or not 0 <= self.filter_threshold <= 1:
                raise ValidationError(
                    f"soft constraint {self.name} requires a [0,1] filter_threshold"
                )
        if self.kind in {"minimize_expression", "maximize_expression"}:
            if not self.variables:
                raise ValidationError(f"soft constraint {self.name} requires variables")
            if self.params.get("coefficients") and len(
                self.params["coefficients"]
            ) != len(self.variables):
                raise ValidationError(
                    f"soft constraint {self.name} coefficient count mismatch"
                )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SoftConstraintSpec:
        data = dict(payload)
        data["ranking_policy"] = SoftRankingPolicy(
            data.get("ranking_policy", "tie_breaker")
        )
        if data.get("normalization"):
            data["normalization"] = NormalizationSpec(**data["normalization"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranking_policy"] = self.ranking_policy.value
        return payload


@dataclass(frozen=True)
class ModelSelection:
    strategy: str = "latest_valid"
    target_mappings: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.strategy not in {"latest_valid", "explicit_model_id"}:
            raise ValidationError(f"unsupported model selection strategy: {self.strategy}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ModelSelection:
        if not payload:
            return cls()
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelQualityGate:
    mode: str = "warn"
    min_cv_r2: float | None = None
    min_test_r2: float | None = None
    max_cv_rmse: float | None = None
    max_rmse_to_target_range_ratio: float | None = None

    def validate(self) -> None:
        if self.mode not in {"warn", "block"}:
            raise ValidationError(f"unsupported model quality gate mode: {self.mode}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ModelQualityGate:
        if not payload:
            return cls()
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValidationError(f"unknown model quality gate fields: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalExperiment:
    experiment_id: str
    values: dict[str, Any]
    observed_values: dict[str, float]
    constraints_report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoricalExperiment:
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationRequest:
    request_id: str | None = None
    mode: str = "recommend_recipe"
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    variables: list[VariableSpec] = field(default_factory=list)
    hard_constraints: list[HardConstraintSpec] = field(default_factory=list)
    soft_constraints: list[SoftConstraintSpec] = field(default_factory=list)
    model_registry_path: str = "engine/artifacts/models/model-registry.json"
    model_selection: ModelSelection = field(default_factory=ModelSelection)
    historical_candidates: list[dict[str, Any]] = field(default_factory=list)
    historical_experiments: list[HistoricalExperiment] = field(default_factory=list)
    top_n: int = 5
    random_seed: int = 42
    max_evaluations: int | None = None
    time_limit: float | None = None
    strategy_thresholds: StrategyThresholds = field(default_factory=StrategyThresholds)
    model_quality_gate: ModelQualityGate = field(default_factory=ModelQualityGate)
    algorithm_override: str | None = None
    acquisition: str | None = None
    preference: str | None = None

    def validate(self) -> None:
        if self.mode not in {"recommend_recipe", "recommend_next_experiments"}:
            raise ValidationError(f"unsupported optimization mode: {self.mode}")
        if not self.objectives:
            raise ValidationError("optimization request requires objectives")
        if self.top_n <= 0:
            raise ValidationError("top_n must be positive")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValidationError("time_limit must be positive")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValidationError("max_evaluations must be positive")
        if self.algorithm_override is not None and self.algorithm_override not in {
            "candidate_rank", "de", "de_rag", "active_set_de", "nsga2", "mixed_nsga2",
        }:
            raise ValidationError(
                f"unsupported algorithm_override: {self.algorithm_override}"
            )
        if self.acquisition is not None and self.acquisition not in {"ei", "pi", "ucb"}:
            raise ValidationError(f"unsupported acquisition: {self.acquisition}")
        self.strategy_thresholds.validate()
        self.model_selection.validate()
        self.model_quality_gate.validate()
        for objective in self.objectives:
            objective.validate()
        for variable in self.variables:
            variable.validate()
        for constraint in self.hard_constraints:
            constraint.validate()
        for constraint in self.soft_constraints:
            constraint.validate()
        _validate_objective_conflicts(self.objectives)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OptimizationRequest:
        data = dict(payload)
        data["objectives"] = [
            ObjectiveSpec.from_dict(item) for item in data.get("objectives", [])
        ]
        data["variables"] = [
            VariableSpec.from_dict(item) for item in data.get("variables", [])
        ]
        data["hard_constraints"] = [
            HardConstraintSpec.from_dict(item)
            for item in data.get("hard_constraints", [])
        ]
        data["soft_constraints"] = [
            SoftConstraintSpec.from_dict(item)
            for item in data.get("soft_constraints", [])
        ]
        data["model_selection"] = ModelSelection.from_dict(
            data.get("model_selection")
        )
        data["strategy_thresholds"] = StrategyThresholds.from_dict(
            data.get("strategy_thresholds")
        )
        data["model_quality_gate"] = ModelQualityGate.from_dict(
            data.get("model_quality_gate")
        )
        data["historical_experiments"] = [
            HistoricalExperiment.from_dict(item)
            for item in data.get("historical_experiments", [])
        ]
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValidationError(f"unknown optimization request fields: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "objectives": [item.to_dict() for item in self.objectives],
            "variables": [item.to_dict() for item in self.variables],
            "hard_constraints": [item.to_dict() for item in self.hard_constraints],
            "soft_constraints": [item.to_dict() for item in self.soft_constraints],
            "model_registry_path": self.model_registry_path,
            "model_selection": self.model_selection.to_dict(),
            "historical_candidates": self.historical_candidates,
            "historical_experiments": [
                item.to_dict() for item in self.historical_experiments
            ],
            "top_n": self.top_n,
            "random_seed": self.random_seed,
            "max_evaluations": self.max_evaluations,
            "time_limit": self.time_limit,
            "strategy_thresholds": self.strategy_thresholds.to_dict(),
            "model_quality_gate": self.model_quality_gate.to_dict(),
            "algorithm_override": self.algorithm_override,
            "acquisition": self.acquisition,
            "preference": self.preference,
        }


@dataclass
class CandidateResult:
    candidate_id: str
    values: dict[str, Any]
    predicted_values: dict[str, float]
    prediction_uncertainty: float | None
    objective_values: dict[str, float]
    objective_errors: dict[str, float]
    hard_constraint_report: list[dict[str, Any]]
    soft_constraint_scores: dict[str, float]
    soft_constraint_score: float
    applicability_domain: str
    pareto_rank: int | None = None
    crowding_distance: float | None = None
    diversity_score: float | None = None
    trust_level: str = "HIGH"
    model_refs: list[dict[str, Any]] = field(default_factory=list)
    selection_reason: str | None = None
    acquisition_name: str | None = None
    acquisition_value: float | None = None
    acquisition_mean: float | None = None
    acquisition_std: float | None = None
    distance_to_nearest_history: float | None = None
    exploration_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    request_id: str
    status: str
    selected_candidates: list[CandidateResult] = field(default_factory=list)
    exploratory_candidates: list[CandidateResult] = field(default_factory=list)
    diagnostic_candidates: list[CandidateResult] = field(default_factory=list)
    rejected_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    artifact_ids: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "optimization_result",
            "request_id": self.request_id,
            "status": self.status,
            "selected_candidates": [
                item.to_dict() for item in self.selected_candidates
            ],
            "exploratory_candidates": [
                item.to_dict() for item in self.exploratory_candidates
            ],
            "diagnostic_candidates": [
                item.to_dict() for item in self.diagnostic_candidates
            ],
            "rejected_summary": self.rejected_summary,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "artifact_ids": self.artifact_ids,
        }


def _validate_objective_conflicts(objectives: list[ObjectiveSpec]) -> None:
    groups: dict[str, list[ObjectiveSpec]] = {}
    for objective in objectives:
        groups.setdefault(objective.target_name, []).append(objective)
    for target_name, group in groups.items():
        lowers = [
            item.value for item in group
            if item.operator is ObjectiveOperator.greater_or_equal and item.value is not None
        ]
        uppers = [
            item.value for item in group
            if item.operator is ObjectiveOperator.less_or_equal and item.value is not None
        ]
        if lowers and uppers and max(lowers) > min(uppers):
            raise ValidationError(f"objective constraints conflict for {target_name}")
