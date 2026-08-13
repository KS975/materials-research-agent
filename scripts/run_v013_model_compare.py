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
        description="V0.1.3-D multi-model comparison and selection."
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
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
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

    print("V0.1.3-D MULTI-MODEL COMPARISON")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print()

    # ------------------------------------------------------------
    # Modeling Gate must explicitly allow training.
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
            "MODEL COMPARISON BLOCKED BY MODELING GATE\n"
            f"decision: {gate.get('decision')}\n"
            "training_allowed: false\n"
            "No candidate model was trained."
        )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
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
                f"ERROR: target column not found: {target_col}"
            )

        feature_cols = [
            col
            for col in fieldnames
            if col.startswith("formula::")
            or col.startswith("process::")
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
            "ERROR: fewer than 10 fully numeric rows after cleaning: "
            f"{usable_rows}"
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)

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

    # Same split for every candidate so comparison is fair.
    candidates = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=args.random_state,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTreesRegressor",
            ExtraTreesRegressor(
                n_estimators=300,
                random_state=args.random_state,
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=args.random_state,
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    for model_name, model in candidates:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)

        importances: list[dict[str, Any]] = []

        if hasattr(model, "feature_importances_"):
            importances = sorted(
                [
                    {
                        "feature": feature,
                        "importance": float(importance),
                    }
                    for feature, importance in zip(
                        feature_cols,
                        model.feature_importances_,
                    )
                ],
                key=lambda item: item["importance"],
                reverse=True,
            )

        results.append(
            {
                "model_name": model_name,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "feature_importance": importances,
            }
        )
        trained_models[model_name] = model

    # Selection rule:
    # 1) maximize R²
    # 2) if tied, minimize RMSE
    # 3) if still tied, minimize MAE
    results.sort(
        key=lambda item: (
            -item["r2"],
            item["rmse"],
            item["mae"],
        )
    )

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    best = results[0]
    best_name = best["model_name"]
    best_model = trained_models[best_name]

    # ------------------------------------------------------------
    # Persist all candidate models + leaderboard + best model
    # ------------------------------------------------------------
    output_dir = (
        Path(".runtime")
        / "v013"
        / "model_comparison"
        / f"project_{project_id}_{target_metric}"
    )
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    model_paths: dict[str, str] = {}

    for model_name, model in trained_models.items():
        model_path = candidates_dir / f"{model_name}.joblib"

        joblib.dump(
            {
                "stage": "V0.1.3-D_model_comparison_candidate",
                "project_id": project_id,
                "target_metric": target_metric,
                "feature_columns": feature_cols,
                "model_name": model_name,
                "model": model,
            },
            model_path,
        )

        model_paths[model_name] = str(model_path)

    best_model_path = output_dir / "best_model.joblib"
    best_report_path = output_dir / "best_model_report.json"
    leaderboard_json_path = output_dir / "leaderboard.json"
    leaderboard_csv_path = output_dir / "leaderboard.csv"

    joblib.dump(
        {
            "stage": "V0.1.3-D_best_model",
            "project_id": project_id,
            "target_metric": target_metric,
            "feature_columns": feature_cols,
            "selection_rule": "max_r2_then_min_rmse_then_min_mae",
            "model_name": best_name,
            "model": best_model,
        },
        best_model_path,
    )

    is_fixture = "fixtures" in dataset_path.parts

    report = {
        "stage": "V0.1.3-D_model_comparison",
        "project_id": project_id,
        "target_metric": target_metric,
        "gate_json": str(gate_path),
        "gate_decision": gate.get("decision"),
        "dataset_csv": str(dataset_path),
        "dataset": {
            "raw_rows": len(rows),
            "usable_rows": usable_rows,
            "dropped_rows": dropped_rows,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
        "selection_rule": "max_r2_then_min_rmse_then_min_mae",
        "leaderboard": results,
        "candidate_model_files": model_paths,
        "best_model": {
            "model_name": best_name,
            "r2": best["r2"],
            "mae": best["mae"],
            "rmse": best["rmse"],
            "model_file": str(best_model_path),
        },
        "scientific_status": {
            "fixture_or_real_data": "fixture" if is_fixture else "unknown_or_real",
            "official_scientific_conclusion_allowed": (
                (not is_fixture)
                and gate.get("official_model_allowed") is True
            ),
        },
    }

    with leaderboard_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with best_report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": "V0.1.3-D_best_model_report",
                "project_id": project_id,
                "target_metric": target_metric,
                "selection_rule": report["selection_rule"],
                "best_model": report["best_model"],
                "feature_importance": best["feature_importance"],
                "scientific_status": report["scientific_status"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with leaderboard_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "model_name",
                "r2",
                "mae",
                "rmse",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "rank": item["rank"],
                    "model_name": item["model_name"],
                    "r2": item["r2"],
                    "mae": item["mae"],
                    "rmse": item["rmse"],
                }
            )

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
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

    print("LEADERBOARD")
    for item in results:
        print(
            f"{item['rank']}. {item['model_name']} | "
            f"R2={item['r2']:.6f} | "
            f"MAE={item['mae']:.6f} | "
            f"RMSE={item['rmse']:.6f}"
        )

    print()
    print("BEST MODEL")
    print(f"model_name: {best_name}")
    print(f"selection_rule: max_r2_then_min_rmse_then_min_mae")
    print(f"R2: {best['r2']:.6f}")
    print(f"MAE: {best['mae']:.6f}")
    print(f"RMSE: {best['rmse']:.6f}")
    print()

    print("OUTPUT")
    print(f"best_model: {best_model_path}")
    print(f"best_report: {best_report_path}")
    print(f"leaderboard_json: {leaderboard_json_path}")
    print(f"leaderboard_csv: {leaderboard_csv_path}")
    print(f"candidate_models: {candidates_dir}")
    print()

    if is_fixture:
        print(
            "NOTE: Comparison used a synthetic fixture. "
            "The ranking verifies the ML selection pipeline only; "
            "it is not a materials-science conclusion."
        )
        print()

    print("V0.1.3-D MULTI-MODEL COMPARISON PASS")


if __name__ == "__main__":
    main()
