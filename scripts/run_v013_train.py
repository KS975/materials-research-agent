from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


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
        description="V0.1.3-C single-model training runner."
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(
            "ERROR: ML dependencies are missing.\n"
            "Install them with:\n"
            "  python -m pip install -r requirements-v013-ml.txt\n"
            f"Original import error: {exc}"
        )

    if not (0.05 <= args.test_size <= 0.50):
        raise SystemExit("ERROR: --test-size must be between 0.05 and 0.50.")

    project_id = args.project_id
    target_metric = args.target
    dataset_path = Path(args.dataset_csv)
    gate_path = Path(args.gate_json)

    print("V0.1.3-C MODEL TRAINING")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print()

    # ------------------------------------------------------------------
    # Hard dependency on Modeling Gate
    # ------------------------------------------------------------------
    gate = load_json(gate_path)

    if gate.get("stage") != "V0.1.3-B_modeling_gate":
        raise SystemExit(
            "ERROR: gate JSON is not a V0.1.3-B Modeling Gate report."
        )

    gate_project_id = gate.get("project_id")
    gate_target = gate.get("target_metric")

    if gate_project_id != project_id:
        raise SystemExit(
            "ERROR: project_id mismatch between command and gate report: "
            f"{project_id} != {gate_project_id}"
        )

    if gate_target != target_metric:
        raise SystemExit(
            "ERROR: target mismatch between command and gate report: "
            f"{target_metric!r} != {gate_target!r}"
        )

    if gate.get("training_allowed") is not True:
        decision = gate.get("decision")
        raise SystemExit(
            "TRAINING BLOCKED BY MODELING GATE\n"
            f"decision: {decision}\n"
            "training_allowed: false\n"
            "No model was trained."
        )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if not dataset_path.exists():
        raise SystemExit(f"ERROR: dataset not found: {dataset_path}")

    target_col = f"target::{target_metric}"

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise SystemExit("ERROR: dataset CSV has no header.")

        fieldnames = list(reader.fieldnames)

        if target_col not in fieldnames:
            raise SystemExit(
                f"ERROR: target column not found: {target_col}\n"
                f"Available columns: {fieldnames}"
            )

        feature_cols = [
            name
            for name in fieldnames
            if name.startswith("formula::")
            or name.startswith("process::")
        ]

        if not feature_cols:
            raise SystemExit(
                "ERROR: no feature columns found. "
                "Expected formula::* and/or process::* columns."
            )

        rows = list(reader)

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    dropped_rows = 0

    for row in rows:
        features: list[float] = []
        valid = True

        for col in feature_cols:
            value = parse_float(row.get(col))
            if value is None:
                valid = False
                break
            features.append(value)

        target_value = parse_float(row.get(target_col))

        if not valid or target_value is None:
            dropped_rows += 1
            continue

        X_rows.append(features)
        y_rows.append(target_value)

    usable_rows = len(X_rows)

    if usable_rows < 10:
        raise SystemExit(
            "ERROR: fewer than 10 fully numeric training rows after cleaning: "
            f"{usable_rows}"
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)

    # ------------------------------------------------------------------
    # Real train/test split and sklearn execution
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if len(y_test) < 2:
        raise SystemExit(
            "ERROR: test split has fewer than 2 rows; R² cannot be evaluated."
        )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)

    importances = sorted(
        (
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(
                feature_cols,
                model.feature_importances_,
            )
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Persist model + report
    # ------------------------------------------------------------------
    output_dir = (
        Path(".runtime")
        / "v013"
        / "models"
        / f"project_{project_id}_{target_metric}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "random_forest.joblib"
    report_path = output_dir / "random_forest_report.json"

    model_bundle = {
        "stage": "V0.1.3-C_model_training",
        "project_id": project_id,
        "target_metric": target_metric,
        "feature_columns": feature_cols,
        "model_name": "RandomForestRegressor",
        "model": model,
    }

    joblib.dump(model_bundle, model_path)

    report = {
        "stage": "V0.1.3-C_model_training",
        "project_id": project_id,
        "target_metric": target_metric,
        "gate_json": str(gate_path),
        "gate_decision": gate.get("decision"),
        "dataset_csv": str(dataset_path),
        "model_name": "RandomForestRegressor",
        "model_params": {
            "n_estimators": 300,
            "random_state": args.random_state,
            "n_jobs": -1,
        },
        "dataset": {
            "raw_rows": len(rows),
            "usable_rows": usable_rows,
            "dropped_rows": dropped_rows,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "test_size": args.test_size,
        },
        "metrics": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        },
        "feature_importance": importances,
        "scientific_status": {
            "fixture_or_real_data": (
                "fixture"
                if "fixtures" in dataset_path.parts
                else "unknown_or_real"
            ),
            "official_scientific_conclusion_allowed": (
                "fixtures" not in dataset_path.parts
                and gate.get("official_model_allowed") is True
            ),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print("GATE")
    print(f"decision: {gate.get('decision')}")
    print("training_allowed: true")
    print()

    print("DATASET")
    print(f"raw_rows: {len(rows)}")
    print(f"usable_rows: {usable_rows}")
    print(f"dropped_rows: {dropped_rows}")
    print(f"feature_count: {len(feature_cols)}")
    print(f"train_rows: {len(y_train)}")
    print(f"test_rows: {len(y_test)}")
    print()

    print("MODEL")
    print("name: RandomForestRegressor")
    print("n_estimators: 300")
    print(f"random_state: {args.random_state}")
    print()

    print("METRICS (REAL SKLEARN EXECUTION)")
    print(f"R2: {r2:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print()

    print("TOP FEATURES")
    for item in importances[:5]:
        print(
            f"- {item['feature']}: "
            f"{item['importance']:.6f}"
        )

    print()
    print("OUTPUT")
    print(f"model_file: {model_path}")
    print(f"report_json: {report_path}")
    print()

    if "fixtures" in dataset_path.parts:
        print(
            "NOTE: This model was trained on a synthetic T10 fixture. "
            "Metrics verify the ML pipeline only; "
            "they are not a materials-science conclusion."
        )
        print()

    print("V0.1.3-C MODEL TRAINING PASS")


if __name__ == "__main__":
    main()
