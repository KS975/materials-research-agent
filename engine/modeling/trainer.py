from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_validate, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from engine.contracts import (
    ApplicabilityDomainRecord,
    CandidateMetrics,
    CandidateTrainingRecord,
    EvaluationSampleRecord,
    ModelArtifactRecord,
    ModelingStrategyRecord,
    TrainingConfig,
)
from engine.exceptions import ValidationError
from engine.modeling.strategy import select_modeling_strategy


@dataclass(frozen=True)
class TrainedCandidate:
    algorithm: str
    pipeline: Pipeline
    metrics: CandidateMetrics
    hyperparameters: dict[str, Any]
    interpretability: dict[str, Any]
    warnings: list[dict[str, Any]]
    evaluation_records: list[EvaluationSampleRecord]
    applicability_domain: ApplicabilityDomainRecord
    applicability_domain_reference: np.ndarray
    elapsed_ms: int


@dataclass
class TrainingRunResult:
    strategies: list[ModelingStrategyRecord]
    candidate_records: list[CandidateTrainingRecord]
    model_artifacts: list[ModelArtifactRecord]
    technical_summary: list[dict[str, Any]]

    @property
    def strategy(self) -> ModelingStrategyRecord:
        return self.strategies[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": [record.to_dict() for record in self.strategies],
            "candidate_records": [
                record.to_dict() for record in self.candidate_records
            ],
            "model_artifacts": [
                record.to_dict() for record in self.model_artifacts
            ],
            "technical_summary": self.technical_summary,
        }


def train_models(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
    *,
    dataset_artifact_id: str | None = None,
    dataset_data_hash: str | None = None,
    source_uri: str = "dataframe",
    output_dir: str | Path = "engine/artifacts/models",
    registry_path: str | Path | None = None,
    user_overrides: dict[str, Any] | None = None,
) -> TrainingRunResult:
    config.validate()
    feature_names = list(config.feature_names) or [
        column for column in dataframe.columns
        if column not in set(config.target_names)
    ]
    missing_columns = (
        set(config.target_names + feature_names) - set(dataframe.columns)
    )
    if missing_columns:
        raise ValidationError(
            f"configured columns missing from dataset: {sorted(missing_columns)}"
        )

    effective_config = TrainingConfig(
        target_names=list(config.target_names),
        feature_names=feature_names,
        reserved_test_ratio=config.reserved_test_ratio,
        cv_mode=config.cv_mode,
        random_seed=config.random_seed,
        algorithms=config.algorithms,
        disabled_algorithms=config.disabled_algorithms,
        optuna_enabled=config.optuna_enabled,
        optuna_trials=config.optuna_trials,
        transform_config=dict(config.transform_config),
        primary_metric=config.primary_metric,
        profile_name=config.profile_name,
    )
    candidate_records: list[CandidateTrainingRecord] = []
    model_artifacts: list[ModelArtifactRecord] = []
    strategies: list[ModelingStrategyRecord] = []
    for target_name in effective_config.target_names:
        target_rows = dataframe.dropna(subset=[target_name])
        strategy = select_modeling_strategy(
            target_name=target_name,
            sample_count=len(target_rows),
            feature_count=len(feature_names),
            config=effective_config,
            user_overrides=user_overrides,
        )
        strategies.append(strategy)
        trained: list[TrainedCandidate] = []
        for algorithm in strategy.selected_algorithms:
            started = time.perf_counter()
            try:
                candidate = _train_candidate(
                    dataframe=target_rows,
                    feature_names=feature_names,
                    target_name=target_name,
                    algorithm=algorithm,
                    config=effective_config,
                    cv_mode=strategy.cv_mode,
                )
                trained.append(candidate)
                candidate_records.append(CandidateTrainingRecord(
                    target_name=target_name,
                    algorithm=algorithm,
                    status="success",
                    metrics=candidate.metrics,
                    hyperparameters=candidate.hyperparameters,
                    interpretability=candidate.interpretability,
                    warnings=candidate.warnings,
                    elapsed_ms=candidate.elapsed_ms,
                ))
            except Exception as exc:
                candidate_records.append(CandidateTrainingRecord(
                    target_name=target_name,
                    algorithm=algorithm,
                    status="failed",
                    metrics=None,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                ))

        if not trained:
            raise RuntimeError(
                f"all candidate models failed for target {target_name}"
            )
        ranked = sorted(
            trained,
            key=lambda item: (
                item.metrics.cv_rmse_mean,
                item.metrics.cv_rmse_std,
                item.metrics.rmse,
                item.metrics.mae,
            ),
        )
        for rank, candidate in enumerate(ranked, start=1):
            for record in candidate_records:
                if (
                    record.target_name == target_name
                    and record.algorithm == candidate.algorithm
                ):
                    object.__setattr__(record, "selection_rank", rank)

        best = ranked[0]
        artifact_record = _save_model_artifact(
            candidate=best,
            config=effective_config,
            dataset_artifact_id=dataset_artifact_id,
            dataset_data_hash=dataset_data_hash or _dataframe_hash(dataframe),
            source_uri=source_uri,
            output_dir=output_dir,
            registry_path=registry_path,
            target_name=target_name,
        )
        model_artifacts.append(artifact_record)

    return TrainingRunResult(
        strategies=strategies,
        candidate_records=candidate_records,
        model_artifacts=model_artifacts,
        technical_summary=_technical_summary(
            strategies, candidate_records, model_artifacts
        ),
    )


def _train_candidate(
    *,
    dataframe: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    algorithm: str,
    config: TrainingConfig,
    cv_mode: str,
) -> TrainedCandidate:
    started = time.perf_counter()
    X = dataframe[feature_names]
    y_raw = dataframe[target_name]
    transform = config.transform_config.get(target_name, "none")
    if transform not in {"none", "log1p"}:
        raise ValidationError(f"unsupported target transform: {transform}")
    if transform == "log1p" and (y_raw <= 0).any():
        raise ValidationError(f"log1p target {target_name} contains non-positive values")
    y = np.log1p(y_raw) if transform == "log1p" else y_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.reserved_test_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    pipeline = _build_pipeline(algorithm, config.random_seed)
    captured_warnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        if config.optuna_enabled and algorithm in {
            "random_forest", "xgboost", "lightgbm", "gradient_boosting", "svr"
        }:
            pipeline = _tune_pipeline(
                algorithm=algorithm,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                config=config,
                cv_mode=cv_mode,
            )
        else:
            pipeline.fit(X_train, y_train)

        y_pred = _inverse_transform(pipeline.predict(X_test), transform)
        y_test_raw = _inverse_transform(y_test, transform)
        metrics = _evaluate(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test_raw,
            y_pred=y_pred,
            cv_mode=cv_mode,
            random_seed=config.random_seed,
        )
    hyperparameters = _safe_params(pipeline)
    interpretability = _interpretability(pipeline, feature_names)
    warning_records = _warning_records(captured_warnings)
    evaluation_records = [
        EvaluationSampleRecord(
            row_index=int(row_index),
            target_name=target_name,
            y_true=_finite_float(actual),
            y_pred=_finite_float(predicted),
            residual=_finite_float(actual - predicted),
            model_id="pending",
            model_version="pending",
        )
        for row_index, actual, predicted in zip(
            y_test_raw.index, y_test_raw.to_numpy(), y_pred
        )
    ]
    ad, ad_reference = _fit_applicability_domain(pipeline, X_train)
    return TrainedCandidate(
        algorithm=algorithm,
        pipeline=pipeline,
        metrics=metrics,
        hyperparameters=hyperparameters,
        interpretability=interpretability,
        warnings=warning_records,
        evaluation_records=evaluation_records,
        applicability_domain=ad,
        applicability_domain_reference=ad_reference,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def _build_pipeline(algorithm: str, random_seed: int) -> Pipeline:
    estimator: Any
    if algorithm == "linear_regression":
        estimator = LinearRegression()
    elif algorithm == "ridge":
        estimator = RidgeCV(alphas=np.logspace(-3, 3, 50))
    elif algorithm == "lasso":
        estimator = LassoCV(
            alphas=np.logspace(-3, 2, 100),
            max_iter=10000,
            random_state=random_seed,
        )
    elif algorithm == "elastic_net":
        estimator = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-3, 2, 30),
            max_iter=10000,
            random_state=random_seed,
        )
    elif algorithm == "bayesian_ridge":
        estimator = BayesianRidge()
    elif algorithm == "pls":
        estimator = PLSRegression(n_components=2)
    elif algorithm == "gaussian_process":
        estimator = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
            normalize_y=True,
            random_state=random_seed,
        )
    elif algorithm == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        )
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            random_state=random_seed,
            verbosity=0,
            n_jobs=-1,
        )
    elif algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_seed,
            verbose=-1,
            n_jobs=-1,
        )
    elif algorithm == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=random_seed,
        )
    elif algorithm == "svr":
        estimator = SVR(C=10.0, epsilon=0.05)
    else:
        raise ValidationError(f"unsupported algorithm: {algorithm}")
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])


def _tune_pipeline(
    *,
    algorithm: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
    cv_mode: str,
) -> Pipeline:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feature_count = len(X_train.columns)
    trials = config.optuna_trials or (30 if feature_count < 20 else 50)
    splitter = _cv_splitter(cv_mode, len(X_train), config.random_seed)

    def objective(trial: optuna.Trial) -> float:
        estimator: Any
        if algorithm == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 12),
                random_state=config.random_seed,
                n_jobs=-1,
            )
        elif algorithm == "xgboost":
            from xgboost import XGBRegressor
            estimator = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                random_state=config.random_seed,
                verbosity=0,
                n_jobs=-1,
            )
        elif algorithm == "lightgbm":
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                random_state=config.random_seed,
                verbose=-1,
                n_jobs=-1,
            )
        elif algorithm == "gradient_boosting":
            estimator = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 5),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                random_state=config.random_seed,
            )
        elif algorithm == "svr":
            estimator = SVR(
                C=trial.suggest_float("C", 0.01, 100.0, log=True),
                epsilon=trial.suggest_float("epsilon", 0.001, 0.5),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            )
        else:
            raise ValidationError(f"algorithm is not tunable: {algorithm}")
        candidate = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ])
        scores = cross_validate(
            candidate,
            X_train,
            y_train,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        return float(np.sqrt(-scores["test_score"].mean()))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    tuned = Pipeline([
        ("scaler", StandardScaler()),
        ("estimator", _build_pipeline(algorithm, config.random_seed)[1]),
    ])
    tuned.named_steps["estimator"].set_params(**study.best_params)
    return tuned.fit(X_train, y_train)


def _evaluate(
    *,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    cv_mode: str,
    random_seed: int,
) -> CandidateMetrics:
    splitter = _cv_splitter(cv_mode, len(X_train), random_seed)
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
    )
    cv_rmse = np.sqrt(-cv["test_neg_rmse"])
    cv_mae = -cv["test_neg_mae"]
    cv_r2 = cv["test_r2"]
    return CandidateMetrics(
        r2=_finite_float(r2_score(y_test, y_pred)),
        mae=_finite_float(mean_absolute_error(y_test, y_pred)),
        rmse=_finite_float(mean_squared_error(y_test, y_pred) ** 0.5),
        cv_rmse_mean=_finite_float(cv_rmse.mean()),
        cv_rmse_std=_finite_float(cv_rmse.std(ddof=1)),
        cv_mae_mean=_finite_float(cv_mae.mean()),
        cv_mae_std=_finite_float(cv_mae.std(ddof=1)),
        cv_r2_mean=_finite_float(cv_r2.mean()),
        cv_r2_std=_finite_float(cv_r2.std(ddof=1)),
        cv_fold_count=len(cv_rmse),
    )


def _cv_splitter(cv_mode: str, sample_count: int, random_seed: int):
    if cv_mode == "loocv":
        return LeaveOneOut()
    splits = min(5, sample_count)
    if splits < 2:
        raise ValidationError("at least two training samples are required for CV")
    return KFold(n_splits=splits, shuffle=True, random_state=random_seed)


def _fit_applicability_domain(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
) -> tuple[ApplicabilityDomainRecord, np.ndarray]:
    scaler = pipeline.named_steps["scaler"]
    standardized = scaler.transform(X_train)
    k = min(5, max(1, len(X_train) - 1))
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(standardized)
    distances, _ = neighbors.kneighbors(standardized)
    mean_distances = distances.mean(axis=1)
    record = ApplicabilityDomainRecord(
        method="standardized_knn_distance",
        k_neighbors=k,
        distance_q75=_finite_float(np.quantile(mean_distances, 0.75)),
        distance_q95=_finite_float(np.quantile(mean_distances, 0.95)),
        training_sample_count=len(X_train),
        feature_count=len(X_train.columns),
    )
    return record, standardized


def _save_model_artifact(
    *,
    candidate: TrainedCandidate,
    config: TrainingConfig,
    dataset_artifact_id: str | None,
    dataset_data_hash: str,
    source_uri: str,
    output_dir: str | Path,
    registry_path: str | Path | None,
    target_name: str,
) -> ModelArtifactRecord:
    from datetime import datetime, timezone

    target_token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", target_name)
    identity = {
        "dataset_data_hash": dataset_data_hash,
        "target": target_name,
        "algorithm": candidate.algorithm,
        "hyperparameters": candidate.hyperparameters,
        "feature_names": config.feature_names,
        "transform": config.transform_config,
        "seed": config.random_seed,
    }
    identity_hash = _hash_object(identity)
    model_id = f"model_{dataset_data_hash[:12]}_{target_token}_{candidate.algorithm}"
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    version_number = 1
    artifact_dir = root / model_id / f"v{version_number:03d}"
    while artifact_dir.exists():
        version_number += 1
        artifact_dir = root / model_id / f"v{version_number:03d}"
    artifact_dir.mkdir(parents=True)
    model_path = artifact_dir / "model.joblib"
    model_version = f"v{version_number:03d}"
    evaluation_records = [
        replace(
            item,
            model_id=model_id,
            model_version=model_version,
        )
        for item in candidate.evaluation_records
    ]

    bundle = {
        "schema_version": 1,
        "model_id": model_id,
        "version": model_version,
        "algorithm": candidate.algorithm,
        "target_name": target_name,
        "feature_names": config.feature_names,
        "target_transform": config.transform_config.get(target_name, "none"),
        "pipeline": candidate.pipeline,
        "metrics": candidate.metrics.to_dict(),
        "applicability_domain": candidate.applicability_domain.to_dict(),
        "interpretability": candidate.interpretability,
        "training_warnings": candidate.warnings,
        "evaluation_records": [
            item.to_dict() for item in evaluation_records
        ],
        "applicability_domain_reference": candidate.applicability_domain_reference,
        "training_config": {
            **vars(config),
            "target_names": [target_name],
        },
        "dataset_artifact_id": dataset_artifact_id,
        "dataset_data_hash": dataset_data_hash,
        "identity_hash": identity_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, model_path)

    record = ModelArtifactRecord(
        model_id=model_id,
        version=model_version,
        target_name=target_name,
        algorithm=candidate.algorithm,
        dataset_artifact_id=dataset_artifact_id,
        dataset_data_hash=dataset_data_hash,
        artifact_dir=str(artifact_dir),
        file_path=str(model_path),
        status="CANDIDATE",
        metrics=candidate.metrics,
        applicability_domain=candidate.applicability_domain,
        feature_names=config.feature_names,
        target_transform=bundle["target_transform"],
        created_at=bundle["created_at"],
        evaluation_records=evaluation_records,
    )
    metadata = record.to_dict()
    metadata.update({
        "source_uri": source_uri,
        "hyperparameters": candidate.hyperparameters,
        "interpretability": candidate.interpretability,
        "training_warnings": candidate.warnings,
        "identity_hash": identity_hash,
    })
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(candidate.metrics.to_dict(), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {"features": config.feature_names, "target": target_name},
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "evaluation.json").write_text(
        json.dumps(
            [item.to_dict() for item in evaluation_records],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _register_model(
        metadata,
        registry_path or root / "model-registry.json",
    )
    return record


def _register_model(metadata: dict[str, Any], registry_path: str | Path) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any]
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": 1, "models": []}
    duplicate = any(
        item.get("model_id") == metadata.get("model_id")
        and item.get("version") == metadata.get("version")
        for item in registry.get("models", [])
    )
    if duplicate:
        raise ValidationError(
            "model registry entry already exists and must not be overwritten"
        )
    registry["models"] = list(registry.get("models", []))
    registry["models"].append(metadata)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _technical_summary(
    strategies: list[ModelingStrategyRecord],
    candidates: list[CandidateTrainingRecord],
    artifacts: list[ModelArtifactRecord],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for strategy in strategies:
        summaries.append({
            "stage": strategy.stage,
            "severity": "info",
            "message": (
                f"目标 {strategy.target_name} 选择 Tier {strategy.tier}，"
                f"CV={strategy.cv_mode}，候选算法 "
                f"{len(strategy.selected_algorithms)} 个。{strategy.strategy_reason}"
            ),
        })
    for artifact in artifacts:
        summaries.append({
            "stage": "model_training.model_selection",
            "severity": "info",
            "message": (
                f"目标 {artifact.target_name} 选择 {artifact.algorithm}，"
                f"CV RMSE Mean={artifact.metrics.cv_rmse_mean:.6f}，"
                f"CV RMSE Std={artifact.metrics.cv_rmse_std:.6f}，"
                f"模型版本 {artifact.model_id}/{artifact.version}。"
            ),
        })
    failures = [item for item in candidates if item.status == "failed"]
    if failures:
        summaries.append({
            "stage": "model_training.candidate_failure",
            "severity": "warning",
            "message": f"{len(failures)} 个候选模型训练失败，已保留错误记录。",
        })
    warning_candidates = [item for item in candidates if item.warnings]
    for item in warning_candidates:
        warning_count = sum(record["count"] for record in item.warnings)
        summaries.append({
            "stage": "model_training.candidate_warning",
            "severity": "warning",
            "message": (
                f"目标 {item.target_name} 的 {item.algorithm} 记录到 "
                f"{len(item.warnings)} 类、{warning_count} 条训练警告。"
            ),
        })
    return summaries


def _interpretability(pipeline: Pipeline, feature_names: list[str]) -> dict[str, Any]:
    estimator = pipeline.named_steps["estimator"]
    result: dict[str, Any] = {}
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is not None:
        values = np.asarray(coefficients).reshape(-1)
        if len(values) == len(feature_names):
            result["coefficient_fields"] = [
                {"feature_name": name, "coefficient": _finite_float(value)}
                for name, value in zip(feature_names, values)
            ]
    intercept = getattr(estimator, "intercept_", None)
    if intercept is not None:
        result["intercept"] = _finite_float(np.asarray(intercept).reshape(-1)[0])
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        values = np.asarray(importances).reshape(-1)
        if len(values) == len(feature_names):
            result["feature_importance"] = [
                {"feature_name": name, "importance": _finite_float(value)}
                for name, value in zip(feature_names, values)
            ]
    return result


def _warning_records(
    captured: list[warnings.WarningMessage],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for item in captured:
        if item.category is None or not issubclass(item.category, ConvergenceWarning):
            continue
        key = (item.category.__name__, str(item.message).strip())
        counts[key] = counts.get(key, 0) + 1
    return [
        {"category": category, "message": message, "count": count}
        for (category, message), count in sorted(counts.items())
    ]


def _inverse_transform(values: Any, transform: str) -> Any:
    if transform == "log1p":
        return np.expm1(values)
    return values


def _safe_params(pipeline: Pipeline) -> dict[str, Any]:
    try:
        return dict(pipeline.named_steps["estimator"].get_params(deep=True))
    except Exception:
        return {}


def _finite_float(value: Any) -> float:
    result = float(value)
    return result if np.isfinite(result) else 0.0


def _dataframe_hash(dataframe: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(map(str, dataframe.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(dataframe, index=True).values.tobytes())
    return digest.hexdigest()


def _hash_object(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
