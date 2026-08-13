from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EDGE_FRACTION = 0.05
NN_IN_QUANTILE = 0.90
NN_OUT_QUANTILE = 0.99
CENTROID_IN_QUANTILE = 0.95
CENTROID_OUT_QUANTILE = 0.99


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: expected JSON object: {path}")

    return data


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V0.1.3-F Applicability Domain evaluator."
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--sample-json", required=True)
    parser.add_argument(
        "--best-model",
        default=None,
        help="Optional T11 best_model.joblib for prediction.",
    )
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise SystemExit(
            "ERROR: ML dependencies are missing.\n"
            "Install them with:\n"
            "  python -m pip install -r requirements-v013-ml.txt\n"
            f"Original import error: {exc}"
        )

    project_id = args.project_id
    target_metric = args.target
    dataset_path = Path(args.dataset_csv)
    gate_path = Path(args.gate_json)
    sample_path = Path(args.sample_json)

    print("V0.1.3-F APPLICABILITY DOMAIN")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print(f"sample_json: {sample_path}")
    print()

    # ------------------------------------------------------------
    # Gate dependency
    # ------------------------------------------------------------
    gate = load_json(gate_path)

    if gate.get("stage") != "V0.1.3-B_modeling_gate":
        raise SystemExit(
            "ERROR: gate JSON is not a V0.1.3-B Modeling Gate report."
        )

    if gate.get("project_id") != project_id:
        raise SystemExit(
            "ERROR: project_id mismatch between command and gate report."
        )

    if gate.get("target_metric") != target_metric:
        raise SystemExit(
            "ERROR: target mismatch between command and gate report."
        )

    if gate.get("training_allowed") is not True:
        raise SystemExit(
            "APPLICABILITY DOMAIN BLOCKED BY MODELING GATE\n"
            f"decision: {gate.get('decision')}\n"
            "training_allowed: false\n"
            "No applicability-domain judgment was produced."
        )

    # ------------------------------------------------------------
    # Training dataset
    # ------------------------------------------------------------
    if not dataset_path.exists():
        raise SystemExit(f"ERROR: dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise SystemExit("ERROR: dataset CSV has no header.")

        inferred_feature_cols = [
            col
            for col in reader.fieldnames
            if col.startswith("formula::")
            or col.startswith("process::")
        ]

        rows = list(reader)

    if not inferred_feature_cols:
        raise SystemExit(
            "ERROR: no feature columns found in training dataset."
        )

    feature_cols = list(inferred_feature_cols)
    model_bundle = None
    model = None
    model_name = None

    # ------------------------------------------------------------
    # Optional best model. Its feature order becomes authoritative.
    # ------------------------------------------------------------
    if args.best_model:
        model_path = Path(args.best_model)

        if not model_path.exists():
            raise SystemExit(f"ERROR: best model not found: {model_path}")

        model_bundle = joblib.load(model_path)

        if not isinstance(model_bundle, dict):
            raise SystemExit("ERROR: invalid best-model bundle.")

        if model_bundle.get("project_id") != project_id:
            raise SystemExit(
                "ERROR: project_id mismatch between command and best model."
            )

        if model_bundle.get("target_metric") != target_metric:
            raise SystemExit(
                "ERROR: target mismatch between command and best model."
            )

        model_feature_cols = model_bundle.get("feature_columns")

        if not isinstance(model_feature_cols, list) or not model_feature_cols:
            raise SystemExit(
                "ERROR: best-model bundle has no valid feature_columns."
            )

        missing_in_dataset = [
            col for col in model_feature_cols
            if col not in inferred_feature_cols
        ]

        if missing_in_dataset:
            raise SystemExit(
                "ERROR: best model expects features missing from dataset: "
                f"{missing_in_dataset}"
            )

        feature_cols = list(model_feature_cols)
        model = model_bundle.get("model")
        model_name = model_bundle.get("model_name")

        if model is None:
            raise SystemExit("ERROR: best-model bundle contains no model.")

    # ------------------------------------------------------------
    # Numeric training matrix
    # ------------------------------------------------------------
    X_rows: list[list[float]] = []
    dropped_rows = 0

    for row in rows:
        vector: list[float] = []
        valid = True

        for col in feature_cols:
            value = parse_float(row.get(col))

            if value is None:
                valid = False
                break

            vector.append(value)

        if valid:
            X_rows.append(vector)
        else:
            dropped_rows += 1

    if len(X_rows) < 10:
        raise SystemExit(
            "ERROR: fewer than 10 usable training rows for AD calibration."
        )

    X = np.asarray(X_rows, dtype=float)

    # ------------------------------------------------------------
    # Sample
    # Accepted forms:
    # {"features": {"formula::ABS": 30, ...}}
    # or {"formula::ABS": 30, ...}
    # ------------------------------------------------------------
    sample_doc = load_json(sample_path)

    raw_features = sample_doc.get("features", sample_doc)

    if not isinstance(raw_features, dict):
        raise SystemExit(
            "ERROR: sample JSON must be a feature object "
            "or contain a 'features' object."
        )

    missing_features = [
        col for col in feature_cols
        if col not in raw_features
    ]

    extra_features = [
        key for key in raw_features.keys()
        if (
            key.startswith("formula::")
            or key.startswith("process::")
        )
        and key not in feature_cols
    ]

    if missing_features:
        raise SystemExit(
            "ERROR: sample is missing required model features: "
            f"{missing_features}"
        )

    sample_values: list[float] = []

    for col in feature_cols:
        value = parse_float(raw_features.get(col))

        if value is None:
            raise SystemExit(
                f"ERROR: sample feature is non-numeric: {col}"
            )

        sample_values.append(value)

    x = np.asarray(sample_values, dtype=float)

    # ------------------------------------------------------------
    # Standardization from training data
    # ------------------------------------------------------------
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    # Avoid divide-by-zero. A constant training feature has no
    # geometric separation power; range checking still catches change.
    safe_stds = np.where(stds > 1e-12, stds, 1.0)

    Xz = (X - means) / safe_stds
    xz = (x - means) / safe_stds

    # ------------------------------------------------------------
    # 1) Single-feature range checks
    # ------------------------------------------------------------
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    range_checks: list[dict[str, Any]] = []
    outside_features: list[str] = []
    edge_features: list[str] = []

    for idx, feature in enumerate(feature_cols):
        value = float(x[idx])
        min_value = float(mins[idx])
        max_value = float(maxs[idx])
        width = max_value - min_value

        if width <= 1e-12:
            in_range = abs(value - min_value) <= 1e-12
            edge_fraction = 0.0 if in_range else None
            normalized_exceedance = (
                0.0 if in_range else float("inf")
            )
        else:
            in_range = min_value <= value <= max_value

            if in_range:
                edge_fraction = min(
                    (value - min_value) / width,
                    (max_value - value) / width,
                )
                normalized_exceedance = 0.0
            else:
                edge_fraction = None

                if value < min_value:
                    normalized_exceedance = (
                        (min_value - value) / width
                    )
                else:
                    normalized_exceedance = (
                        (value - max_value) / width
                    )

        if not in_range:
            outside_features.append(feature)
        elif edge_fraction is not None and edge_fraction < EDGE_FRACTION:
            edge_features.append(feature)

        range_checks.append(
            {
                "feature": feature,
                "value": value,
                "train_min": min_value,
                "train_max": max_value,
                "in_range": bool(in_range),
                "edge_fraction": (
                    float(edge_fraction)
                    if edge_fraction is not None
                    else None
                ),
                "normalized_exceedance": (
                    float(normalized_exceedance)
                    if math.isfinite(normalized_exceedance)
                    else None
                ),
            }
        )

    # ------------------------------------------------------------
    # 2) Standardized distance to training centroid
    # ------------------------------------------------------------
    training_centroid_distances = np.linalg.norm(Xz, axis=1)
    sample_centroid_distance = float(np.linalg.norm(xz))

    centroid_in_threshold = float(
        np.quantile(
            training_centroid_distances,
            CENTROID_IN_QUANTILE,
        )
    )
    centroid_out_threshold = float(
        np.quantile(
            training_centroid_distances,
            CENTROID_OUT_QUANTILE,
        )
    )

    # ------------------------------------------------------------
    # 3) Standardized nearest-neighbor distance
    # Calibrate with leave-one-out nearest-neighbor distances.
    # ------------------------------------------------------------
    nn = NearestNeighbors(
        n_neighbors=2,
        metric="euclidean",
    )
    nn.fit(Xz)

    training_distances, _ = nn.kneighbors(Xz)
    training_loo_nn = training_distances[:, 1]

    nn_in_threshold = float(
        np.quantile(
            training_loo_nn,
            NN_IN_QUANTILE,
        )
    )
    nn_out_threshold = float(
        np.quantile(
            training_loo_nn,
            NN_OUT_QUANTILE,
        )
    )

    sample_nn_model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )
    sample_nn_model.fit(Xz)

    sample_distances, sample_indices = sample_nn_model.kneighbors(
        xz.reshape(1, -1)
    )

    sample_nn_distance = float(sample_distances[0, 0])
    nearest_training_row_index = int(sample_indices[0, 0])

    # ------------------------------------------------------------
    # 4) Final AD decision
    # ------------------------------------------------------------
    reasons: list[str] = []

    if outside_features:
        status = "OUT_OF_DOMAIN"
        risk = "HIGH"

        for feature in outside_features:
            reasons.append(
                f"{feature} 超出训练数据单特征范围"
            )

    elif (
        sample_nn_distance > nn_out_threshold
        or sample_centroid_distance > centroid_out_threshold
    ):
        status = "OUT_OF_DOMAIN"
        risk = "HIGH"

        if sample_nn_distance > nn_out_threshold:
            reasons.append(
                "与最近训练样本的标准化距离超过 "
                "99% 训练近邻距离阈值"
            )

        if sample_centroid_distance > centroid_out_threshold:
            reasons.append(
                "到训练分布中心的标准化距离超过 "
                "99% 训练分布阈值"
            )

    elif (
        edge_features
        or sample_nn_distance > nn_in_threshold
        or sample_centroid_distance > centroid_in_threshold
    ):
        status = "BORDERLINE"
        risk = "MEDIUM"

        if edge_features:
            reasons.append(
                "部分特征处于训练范围边缘 5% 区域："
                + ", ".join(edge_features)
            )

        if sample_nn_distance > nn_in_threshold:
            reasons.append(
                "与最近训练样本距离高于训练近邻距离的 "
                "90% 分位阈值"
            )

        if sample_centroid_distance > centroid_in_threshold:
            reasons.append(
                "到训练分布中心距离高于训练样本的 "
                "95% 分位阈值"
            )

    else:
        status = "IN_DOMAIN"
        risk = "LOW"
        reasons.append(
            "所有特征均在训练范围内，且整体距离与最近邻距离"
            "均处于训练数据主要覆盖区域"
        )

    # ------------------------------------------------------------
    # Optional model prediction
    # ------------------------------------------------------------
    prediction = None

    if model is not None:
        prediction = float(
            model.predict(x.reshape(1, -1))[0]
        )

    # ------------------------------------------------------------
    # Persist report
    # ------------------------------------------------------------
    output_dir = (
        Path(".runtime")
        / "v013"
        / "applicability_domain"
        / f"project_{project_id}_{target_metric}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = sample_doc.get(
        "sample_name",
        sample_path.stem,
    )

    safe_sample_name = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(sample_name)
    )

    report_path = output_dir / f"{safe_sample_name}_ad_report.json"

    report = {
        "stage": "V0.1.3-F_applicability_domain",
        "project_id": project_id,
        "target_metric": target_metric,
        "sample_name": sample_name,
        "sample_json": str(sample_path),
        "training_dataset_csv": str(dataset_path),
        "gate_json": str(gate_path),
        "gate_decision": gate.get("decision"),
        "feature_columns": feature_cols,
        "extra_sample_features_ignored": extra_features,
        "calibration": {
            "usable_training_rows": int(len(X)),
            "dropped_training_rows": dropped_rows,
            "edge_fraction_threshold": EDGE_FRACTION,
            "nearest_neighbor": {
                "metric": "euclidean_on_standardized_features",
                "in_quantile": NN_IN_QUANTILE,
                "out_quantile": NN_OUT_QUANTILE,
                "in_threshold": nn_in_threshold,
                "out_threshold": nn_out_threshold,
            },
            "centroid_distance": {
                "metric": "euclidean_on_standardized_features",
                "in_quantile": CENTROID_IN_QUANTILE,
                "out_quantile": CENTROID_OUT_QUANTILE,
                "in_threshold": centroid_in_threshold,
                "out_threshold": centroid_out_threshold,
            },
        },
        "range_checks": range_checks,
        "distance_checks": {
            "sample_nearest_neighbor_distance": sample_nn_distance,
            "nearest_training_row_index": nearest_training_row_index,
            "sample_centroid_distance": sample_centroid_distance,
        },
        "applicability_domain": {
            "status": status,
            "risk": risk,
            "reasons": reasons,
        },
        "prediction": (
            {
                "model_name": model_name,
                "value": prediction,
                "target_metric": target_metric,
                "interpretation_rule": (
                    "Prediction is numerically available, but trust must be "
                    "conditioned on applicability_domain.status."
                ),
            }
            if prediction is not None
            else None
        ),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------
    print("TRAINING DOMAIN")
    print(f"usable_training_rows: {len(X)}")
    print(f"feature_count: {len(feature_cols)}")
    print()

    print("RANGE CHECK")
    print(f"outside_feature_count: {len(outside_features)}")
    print(f"edge_feature_count: {len(edge_features)}")

    for item in range_checks:
        state = "IN"
        if not item["in_range"]:
            state = "OUT"
        elif (
            item["edge_fraction"] is not None
            and item["edge_fraction"] < EDGE_FRACTION
        ):
            state = "EDGE"

        print(
            f"- {item['feature']}: "
            f"value={item['value']:.6f}, "
            f"train=[{item['train_min']:.6f}, "
            f"{item['train_max']:.6f}] -> {state}"
        )

    print()
    print("DISTANCE CHECK")
    print(
        "nearest_neighbor_distance: "
        f"{sample_nn_distance:.6f}"
    )
    print(
        "nearest_neighbor_thresholds: "
        f"IN<={nn_in_threshold:.6f}, "
        f"OUT>{nn_out_threshold:.6f}"
    )
    print(
        "centroid_distance: "
        f"{sample_centroid_distance:.6f}"
    )
    print(
        "centroid_thresholds: "
        f"IN<={centroid_in_threshold:.6f}, "
        f"OUT>{centroid_out_threshold:.6f}"
    )
    print()

    if prediction is not None:
        print("PREDICTION")
        print(f"model_name: {model_name}")
        print(
            f"predicted_{target_metric}: "
            f"{prediction:.6f}"
        )
        print()

    print("APPLICABILITY DOMAIN")
    print(f"status: {status}")
    print(f"risk: {risk}")

    print("reasons:")
    for reason in reasons:
        print(f"- {reason}")

    print()
    print("OUTPUT")
    print(f"ad_report_json: {report_path}")
    print()
    print("V0.1.3-F APPLICABILITY DOMAIN PASS")


if __name__ == "__main__":
    main()
