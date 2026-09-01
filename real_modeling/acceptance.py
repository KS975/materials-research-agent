from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from company_data import (
    CompanyDataRepository,
    CompanyDataValidationError,
    resolve_company_data_runtime_root,
)


STAGE = "V0.3-real_data_modeling_acceptance_v1"
SCHEMA_VERSION = 1


class RealModelingAcceptanceError(RuntimeError):
    pass


def _safe(value: Any) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(value or "")).strip()
    return text[:120] or "value"


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _unit_metadata_available(repo: CompanyDataRepository) -> bool:
    path = repo.import_dir() / "performances_long.csv"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    normalized = {str(x or "").strip().casefold() for x in header}
    return bool(
        normalized
        & {"unit", "units", "单位", "量纲", "test_unit"}
    )


def _target_regime_diagnostic(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(x) for x in values if math.isfinite(float(x)))
    if len(ordered) < 10:
        return {
            "suspected": False,
            "reason": "INSUFFICIENT_TARGET_ROWS",
            "count": len(ordered),
        }

    gaps = sorted(
        (
            (b - a, a, b)
            for a, b in zip(ordered[:-1], ordered[1:])
        ),
        reverse=True,
    )
    largest_gap, left_edge, right_edge = gaps[0]
    second_gap = gaps[1][0] if len(gaps) > 1 else 0.0
    left_count = sum(x <= left_edge for x in ordered)
    right_count = sum(x >= right_edge for x in ordered)
    gap_ratio = (
        largest_gap / second_gap
        if second_gap > 1e-12 else None
    )
    balanced = min(left_count, right_count) >= max(5, int(len(ordered) * 0.10))
    suspected = bool(
        balanced
        and second_gap > 0
        and largest_gap >= second_gap * 3.0
    )

    return {
        "suspected": suspected,
        "count": len(ordered),
        "largest_gap": float(largest_gap),
        "largest_gap_left": float(left_edge),
        "largest_gap_right": float(right_edge),
        "second_largest_gap": float(second_gap),
        "largest_to_second_gap_ratio": (
            float(gap_ratio) if gap_ratio is not None else None
        ),
        "left_regime_count": int(left_count),
        "right_regime_count": int(right_count),
        "interpretation": (
            "TARGET_SCALE_OR_CONDITION_REGIME_SUSPECTED"
            if suspected
            else "NO_STRONG_TWO_REGIME_GAP_DETECTED"
        ),
        "warning": (
            "该诊断只能提示目标值可能混有不同量纲/测试条件/数据制度，"
            "不能在缺少 unit/test-condition 元数据时自动断言原因。"
        ),
    }


def _feature_selection(
    rows: list[dict[str, Any]],
    feature_cols: list[str],
    train_indices: list[int],
    *,
    minimum_coverage: float,
    minimum_unique_numeric: int,
    maximum_features: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    stats: list[dict[str, Any]] = []
    train_n = len(train_indices)

    for col in feature_cols:
        values = [
            _num(rows[idx].get(col))
            for idx in train_indices
        ]
        numeric = [x for x in values if x is not None]
        unique = len(set(numeric))
        coverage = len(numeric) / train_n if train_n else 0.0
        stats.append({
            "feature": col,
            "material_id": col.removeprefix("formula::"),
            "train_numeric_count": len(numeric),
            "train_coverage": coverage,
            "train_unique_numeric": unique,
            "selected": (
                coverage >= minimum_coverage
                and unique >= minimum_unique_numeric
            ),
        })

    eligible = [
        item for item in stats if item["selected"]
    ]
    eligible.sort(
        key=lambda item: (
            -item["train_coverage"],
            -item["train_unique_numeric"],
            item["feature"],
        )
    )
    selected = [
        item["feature"]
        for item in eligible[:maximum_features]
    ]
    selected_set = set(selected)
    for item in stats:
        item["selected"] = item["feature"] in selected_set

    stats.sort(
        key=lambda item: (
            not item["selected"],
            -item["train_coverage"],
            -item["train_unique_numeric"],
            item["feature"],
        )
    )
    return selected, stats


def _matrix(
    rows: list[dict[str, Any]],
    indices: list[int],
    features: list[str],
):
    import numpy as np

    matrix = []
    for idx in indices:
        matrix.append([
            (
                float(value)
                if (value := _num(rows[idx].get(col))) is not None
                else np.nan
            )
            for col in features
        ])
    return np.asarray(matrix, dtype=float)


def _target(rows: list[dict[str, Any]], indices: list[int], target_col: str):
    import numpy as np

    return np.asarray(
        [float(_num(rows[idx][target_col])) for idx in indices],
        dtype=float,
    )


def _pipelines(random_state: int) -> dict[str, Any]:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def imputer():
        # We intentionally do NOT convert blanks to zero. In the imported
        # source, a blank formula cell may mean "ingredient absent" or
        # "value not recorded"; the canonical normalized layer does not
        # currently distinguish those meanings.
        return SimpleImputer(
            strategy="median",
            add_indicator=True,
        )

    return {
        "DummyMedian": Pipeline([
            ("imputer", imputer()),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "Ridge": Pipeline([
            ("imputer", imputer()),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForestRegressor": Pipeline([
            ("imputer", imputer()),
            ("model", RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "ExtraTreesRegressor": Pipeline([
            ("imputer", imputer()),
            ("model", ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "GradientBoostingRegressor": Pipeline([
            ("imputer", imputer()),
            ("model", GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.04,
                max_depth=2,
                loss="huber",
                random_state=random_state,
            )),
        ]),
    }


def _cv_leaderboard(
    X_train,
    y_train,
    *,
    random_state: int,
) -> tuple[list[dict[str, Any]], str, Any]:
    import numpy as np
    from sklearn.model_selection import RepeatedKFold, cross_validate

    candidates = _pipelines(random_state)
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=random_state,
    )
    leaderboard: list[dict[str, Any]] = []

    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            return_train_score=False,
            n_jobs=1,
        )
        leaderboard.append({
            "model_name": name,
            "cv_folds": int(len(scores["test_mae"])),
            "cv_mae_mean": float(-np.mean(scores["test_mae"])),
            "cv_mae_std": float(np.std(-scores["test_mae"])),
            "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
            "cv_r2_mean": float(np.mean(scores["test_r2"])),
            "is_baseline": name == "DummyMedian",
        })

    baseline = next(
        item for item in leaderboard
        if item["model_name"] == "DummyMedian"
    )
    model_rows = [
        item for item in leaderboard
        if not item["is_baseline"]
    ]
    model_rows.sort(
        key=lambda item: (
            item["cv_mae_mean"],
            item["cv_rmse_mean"],
            -item["cv_r2_mean"],
            item["model_name"],
        )
    )
    selected = model_rows[0]["model_name"]

    leaderboard.sort(
        key=lambda item: (
            item["is_baseline"],
            item["cv_mae_mean"],
            item["model_name"],
        )
    )
    for rank, item in enumerate(
        [x for x in leaderboard if not x["is_baseline"]],
        start=1,
    ):
        item["model_rank"] = rank

    selected_row = next(
        item for item in leaderboard
        if item["model_name"] == selected
    )
    selected_row["cv_mae_improvement_vs_dummy"] = float(
        baseline["cv_mae_mean"] - selected_row["cv_mae_mean"]
    )
    selected_row["cv_mae_improvement_ratio_vs_dummy"] = (
        float(
            (
                baseline["cv_mae_mean"]
                - selected_row["cv_mae_mean"]
            )
            / baseline["cv_mae_mean"]
        )
        if baseline["cv_mae_mean"] > 0 else None
    )
    return leaderboard, selected, candidates[selected]


def _ad_calibration(X_train, X_holdout) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )
    scaler = StandardScaler()

    train_imp = imputer.fit_transform(X_train)
    hold_imp = imputer.transform(X_holdout)
    train_z = scaler.fit_transform(train_imp)
    hold_z = scaler.transform(hold_imp)

    train_nn = NearestNeighbors(n_neighbors=2).fit(train_z)
    train_distances, _ = train_nn.kneighbors(train_z)
    leave_one_out = train_distances[:, 1]
    q95 = float(np.quantile(leave_one_out, 0.95))
    q99 = float(np.quantile(leave_one_out, 0.99))

    hold_nn = NearestNeighbors(n_neighbors=1).fit(train_z)
    distances, indices = hold_nn.kneighbors(hold_z)

    rows: list[dict[str, Any]] = []
    for distance, index in zip(distances[:, 0], indices[:, 0]):
        value = float(distance)
        if value <= q95:
            status = "IN_DOMAIN"
        elif value <= q99:
            status = "BORDERLINE"
        else:
            status = "OUT_OF_DOMAIN"
        rows.append({
            "nearest_neighbor_distance": value,
            "nearest_train_position": int(index),
            "domain_status": status,
        })

    return {
        "method": "standardized_nearest_neighbor",
        "imputer": "median_plus_missing_indicator",
        "fit_scope": "TRAIN_ONLY",
        "train_loo_distance_q95": q95,
        "train_loo_distance_q99": q99,
    }, rows


def _exact_transformed_collisions(
    X_train,
    y_train,
    train_sample_ids: list[str],
    train_sample_names: list[str],
) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )
    transformed = imputer.fit_transform(X_train)
    groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, vector in enumerate(transformed):
        key = tuple(float(x) for x in np.round(vector, 12))
        groups[key].append(index)

    collisions: list[dict[str, Any]] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        targets = [float(y_train[idx]) for idx in indices]
        if max(targets) - min(targets) <= 1e-12:
            continue
        collisions.append({
            "count": len(indices),
            "sample_ids": [train_sample_ids[idx] for idx in indices],
            "sample_names": [train_sample_names[idx] for idx in indices],
            "target_values": targets,
            "target_range": float(max(targets) - min(targets)),
        })
    collisions.sort(
        key=lambda item: (-item["target_range"], item["sample_ids"][0])
    )
    return collisions


class RealDataModelingAcceptance:
    """Exploratory real-company-data modeling acceptance.

    This service is deliberately isolated from the official model registry and
    Bayesian-optimization pipeline. It may train a DIAGNOSTIC model even while
    the formal Modeling Gate remains FAIL, because its only purpose is to
    measure whether the historical data currently contains predictive signal.

    It never sets official_model_allowed=true and never registers a model for
    autonomous optimization.
    """

    def __init__(
        self,
        runtime_root: str | Path | None = None,
    ) -> None:
        self.runtime_root = resolve_company_data_runtime_root(
            runtime_root
        )
        self.repo = CompanyDataRepository(self.runtime_root)

    def _output_dir(
        self,
        project_id: int,
        product_name: str,
        target_metric: str,
    ) -> Path:
        return (
            self.runtime_root
            / "v030"
            / "real_modeling_acceptance"
            / f"project_{project_id}_{_safe(product_name)}_{_safe(target_metric)}"
        )

    def run(
        self,
        *,
        product_name: str,
        target_metric: str,
        test_size: float = 0.20,
        random_state: int = 42,
        minimum_feature_coverage: float = 0.20,
        minimum_unique_numeric: int = 2,
        maximum_features: int = 40,
    ) -> dict[str, Any]:
        try:
            import joblib
            import numpy as np
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RealModelingAcceptanceError(
                "缺少真实建模依赖；请安装 requirements-v013-ml.txt"
            ) from exc

        if not (0.10 <= test_size <= 0.40):
            raise RealModelingAcceptanceError(
                "test_size 必须在 0.10~0.40"
            )
        if not (0.05 <= minimum_feature_coverage <= 1.0):
            raise RealModelingAcceptanceError(
                "minimum_feature_coverage 必须在 0.05~1.0"
            )

        manifest = self.repo.manifest()
        available_metrics = {
            str(item.get("metric") or "")
            for item in manifest.get("performance_coverage") or []
        }
        if target_metric not in available_metrics:
            close = sorted(
                metric for metric in available_metrics
                if target_metric in metric or metric in target_metric
            )
            suffix = f"；可能匹配: {close}" if close else ""
            raise RealModelingAcceptanceError(
                f"目标性能必须使用真实库精确字段名: {target_metric}{suffix}"
            )

        export = self.repo.export_modeling_dataset(
            product_name=product_name,
            target_metric=target_metric,
        )
        dataset_path = Path(export["dataset_csv"])
        reality_path = Path(export["reality_json"])
        reality = json.loads(
            reality_path.read_text(encoding="utf-8")
        )

        with dataset_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            rows_all = list(reader)

        target_col = f"target::{target_metric}"
        feature_cols = [
            col for col in fields
            if col.startswith("formula::")
        ]
        valid_row_positions = [
            idx for idx, row in enumerate(rows_all)
            if _num(row.get(target_col)) is not None
        ]
        if len(valid_row_positions) < 20:
            raise RealModelingAcceptanceError(
                f"有效目标样本不足 20: {len(valid_row_positions)}"
            )

        valid_rows = [
            rows_all[idx] for idx in valid_row_positions
        ]
        positions = np.arange(len(valid_rows))
        train_pos, hold_pos = train_test_split(
            positions,
            test_size=test_size,
            random_state=random_state,
        )
        train_pos = sorted(int(x) for x in train_pos)
        hold_pos = sorted(int(x) for x in hold_pos)

        selected_features, feature_stats = _feature_selection(
            valid_rows,
            feature_cols,
            train_pos,
            minimum_coverage=minimum_feature_coverage,
            minimum_unique_numeric=minimum_unique_numeric,
            maximum_features=maximum_features,
        )
        if len(selected_features) < 2:
            raise RealModelingAcceptanceError(
                "训练集筛选后可用配方特征少于 2 个"
            )

        X_train = _matrix(valid_rows, train_pos, selected_features)
        X_hold = _matrix(valid_rows, hold_pos, selected_features)
        y_train = _target(valid_rows, train_pos, target_col)
        y_hold = _target(valid_rows, hold_pos, target_col)

        train_sample_ids = [
            str(valid_rows[idx].get("sample_id") or "")
            for idx in train_pos
        ]
        hold_sample_ids = [
            str(valid_rows[idx].get("sample_id") or "")
            for idx in hold_pos
        ]
        train_sample_names = [
            str(valid_rows[idx].get("sample_name") or "")
            for idx in train_pos
        ]
        hold_sample_names = [
            str(valid_rows[idx].get("sample_name") or "")
            for idx in hold_pos
        ]

        # Strict split audit.
        overlap = sorted(set(train_sample_ids) & set(hold_sample_ids))
        if overlap:
            raise RealModelingAcceptanceError(
                f"Train/Holdout sample_id 泄漏: {overlap[:10]}"
            )

        leaderboard, selected_model_name, selected_pipeline = (
            _cv_leaderboard(
                X_train,
                y_train,
                random_state=random_state,
            )
        )

        selected_pipeline.fit(X_train, y_train)
        hold_prediction = selected_pipeline.predict(X_hold)
        hold_metrics = _metrics(y_hold, hold_prediction)

        dummy = _pipelines(random_state)["DummyMedian"]
        dummy.fit(X_train, y_train)
        dummy_prediction = dummy.predict(X_hold)
        dummy_metrics = _metrics(y_hold, dummy_prediction)

        holdout_mae_improvement = (
            dummy_metrics["mae"] - hold_metrics["mae"]
        )
        holdout_mae_improvement_ratio = (
            holdout_mae_improvement / dummy_metrics["mae"]
            if dummy_metrics["mae"] > 0 else None
        )
        holdout_beats_baseline = (
            hold_metrics["mae"] < dummy_metrics["mae"]
        )

        ad_calibration, ad_rows = _ad_calibration(
            X_train,
            X_hold,
        )

        prediction_rows: list[dict[str, Any]] = []
        for offset, idx in enumerate(hold_pos):
            actual = float(y_hold[offset])
            predicted = float(hold_prediction[offset])
            nearest_train_position = int(
                ad_rows[offset]["nearest_train_position"]
            )
            prediction_rows.append({
                "sample_id": hold_sample_ids[offset],
                "sample_name": hold_sample_names[offset],
                "actual": actual,
                "predicted": predicted,
                "residual": predicted - actual,
                "absolute_error": abs(predicted - actual),
                "domain_status": ad_rows[offset]["domain_status"],
                "nearest_neighbor_distance": ad_rows[offset][
                    "nearest_neighbor_distance"
                ],
                "nearest_train_sample_id": train_sample_ids[
                    nearest_train_position
                ],
                "nearest_train_target": float(
                    y_train[nearest_train_position]
                ),
            })
        prediction_rows.sort(
            key=lambda item: (
                -item["absolute_error"],
                item["sample_id"],
            )
        )

        target_regime = _target_regime_diagnostic(
            [float(x) for x in y_train]
        )
        collisions = _exact_transformed_collisions(
            X_train,
            y_train,
            train_sample_ids,
            train_sample_names,
        )
        unit_available = _unit_metadata_available(self.repo)

        official_boundary = {
            "formal_reality_core_closed_samples": int(
                (reality.get("summary") or {}).get(
                    "core_closed_formula_process_target", 0
                )
            ),
            "process_parameter_rows": int(
                export.get("process_parameter_rows") or 0
            ),
            "explicit_test_condition_rows": int(
                export.get("condition_rows") or 0
            ),
            "official_model_allowed": False,
            "bo_allowed": False,
            "autonomous_model_use_allowed": False,
            "model_registry_write": False,
            "reason": (
                "当前真实数据缺少材料工艺参数和显式测试条件；"
                "本次训练仅为 DIAGNOSTIC / EXPLORATORY acceptance，"
                "不进入正式模型注册表，也不进入 BO。"
            ),
        }

        review_reasons: list[str] = []
        if not unit_available:
            review_reasons.append("TARGET_UNIT_METADATA_UNAVAILABLE")
        if target_regime.get("suspected"):
            review_reasons.append(
                "TARGET_SCALE_OR_CONDITION_REGIME_SUSPECTED"
            )
        if collisions:
            review_reasons.append(
                "IDENTICAL_SELECTED_FEATURES_WITH_DIFFERENT_TARGETS"
            )
        if not holdout_beats_baseline:
            review_reasons.append(
                "SELECTED_MODEL_DOES_NOT_BEAT_HOLDOUT_MEDIAN_BASELINE"
            )
        if hold_metrics["r2"] < 0:
            review_reasons.append("NEGATIVE_HOLDOUT_R2")

        status = (
            "REVIEW_REQUIRED"
            if review_reasons
            else "EXPLORATORY_PASS"
        )

        out_dir = self._output_dir(
            int(export["product"]["local_project_id"]),
            product_name,
            target_metric,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_manifest = {
            "stage": STAGE,
            "schema_version": SCHEMA_VERSION,
            "source_kind": "company_real_data",
            "canonical_source": (
                manifest.get("source") or {}
            ).get("canonical_source"),
            "company_dataset_id": export["dataset_id"],
            "company_source_sha256": (
                manifest.get("source") or {}
            ).get("sha256"),
            "modeling_export_csv": str(dataset_path),
            "modeling_export_sha256": _sha256_file(dataset_path),
            "project_id": int(export["product"]["local_project_id"]),
            "product_type": product_name,
            "target_metric": target_metric,
            "source_rows": len(rows_all),
            "target_numeric_rows": len(valid_rows),
            "simulator_rows": 0,
            "train_rows": len(train_pos),
            "holdout_rows": len(hold_pos),
            "train_sample_ids": train_sample_ids,
            "holdout_sample_ids": hold_sample_ids,
            "split_overlap_count": 0,
            "split_sha256": _sha256_json({
                "train": train_sample_ids,
                "holdout": hold_sample_ids,
                "random_state": random_state,
                "test_size": test_size,
            }),
            "split_policy": (
                "random_holdout_before_feature_selection; "
                "holdout target not used for model selection"
            ),
        }

        feature_manifest = {
            "selection_fit_scope": "TRAIN_ONLY",
            "all_active_formula_features": len(feature_cols),
            "minimum_train_coverage": minimum_feature_coverage,
            "minimum_unique_numeric": minimum_unique_numeric,
            "maximum_features": maximum_features,
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "feature_stats": feature_stats,
            "missing_value_policy": (
                "median imputation + missing indicators fitted on training only; "
                "blank formula cells are NOT silently converted to zero"
            ),
        }

        cv_report = {
            "selection_scope": "TRAIN_CV_ONLY",
            "selection_rule": (
                "lowest repeated-CV MAE; holdout is not used to choose model"
            ),
            "cv": "RepeatedKFold(n_splits=5,n_repeats=2,random_state=42)",
            "leaderboard": leaderboard,
            "selected_model": selected_model_name,
        }

        model_path = out_dir / "diagnostic_model.joblib"
        joblib.dump({
            "stage": STAGE,
            "use_classification": "DIAGNOSTIC_ONLY",
            "project_id": int(export["product"]["local_project_id"]),
            "product_type": product_name,
            "target_metric": target_metric,
            "selected_features": selected_features,
            "model_name": selected_model_name,
            "model": selected_pipeline,
            "official_model_allowed": False,
            "bo_allowed": False,
        }, model_path)

        predictions_path = out_dir / "holdout_predictions.csv"
        _atomic_csv(
            predictions_path,
            [
                "sample_id",
                "sample_name",
                "actual",
                "predicted",
                "residual",
                "absolute_error",
                "domain_status",
                "nearest_neighbor_distance",
                "nearest_train_sample_id",
                "nearest_train_target",
            ],
            prediction_rows,
        )

        report = {
            "stage": STAGE,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "mode": "EXPLORATORY_REAL_MODEL_DIAGNOSTIC",
            "project_id": int(export["product"]["local_project_id"]),
            "product_type": product_name,
            "target_metric": target_metric,
            "source": dataset_manifest,
            "feature_selection": feature_manifest,
            "model_comparison": cv_report,
            "holdout": {
                "used_once_after_model_selection": True,
                "rows": len(hold_pos),
                "selected_model_metrics": hold_metrics,
                "median_baseline_metrics": dummy_metrics,
                "selected_model_beats_median_baseline": holdout_beats_baseline,
                "mae_improvement_vs_median_baseline": float(
                    holdout_mae_improvement
                ),
                "mae_improvement_ratio_vs_median_baseline": (
                    float(holdout_mae_improvement_ratio)
                    if holdout_mae_improvement_ratio is not None
                    else None
                ),
                "domain_counts": {
                    status_name: sum(
                        row["domain_status"] == status_name
                        for row in prediction_rows
                    )
                    for status_name in (
                        "IN_DOMAIN",
                        "BORDERLINE",
                        "OUT_OF_DOMAIN",
                    )
                },
            },
            "applicability_domain": ad_calibration,
            "data_quality_diagnostics": {
                "target_unit_metadata_available": unit_available,
                "target_regime": target_regime,
                "identical_selected_feature_collision_count": len(collisions),
                "identical_selected_feature_collisions": collisions[:20],
            },
            "review_reasons": review_reasons,
            "official_boundary": official_boundary,
            "artifacts": {
                "dataset_manifest_json": str(
                    out_dir / "dataset_manifest.json"
                ),
                "feature_manifest_json": str(
                    out_dir / "feature_manifest.json"
                ),
                "cv_report_json": str(
                    out_dir / "cv_report.json"
                ),
                "holdout_predictions_csv": str(predictions_path),
                "diagnostic_model_joblib": str(model_path),
                "acceptance_report_json": str(
                    out_dir / "acceptance_report.json"
                ),
            },
        }

        _atomic_json(
            out_dir / "dataset_manifest.json",
            dataset_manifest,
        )
        _atomic_json(
            out_dir / "feature_manifest.json",
            feature_manifest,
        )
        _atomic_json(
            out_dir / "cv_report.json",
            cv_report,
        )
        _atomic_json(
            out_dir / "acceptance_report.json",
            report,
        )

        return report
