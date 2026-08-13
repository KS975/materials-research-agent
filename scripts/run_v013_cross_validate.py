from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
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


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list.")

    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V0.1.3-E cross-validation stability evaluation."
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import numpy as np
        from sklearn.base import clone
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
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise SystemExit(
            "ERROR: ML dependencies are missing.\n"
            "Install them with:\n"
            "  python -m pip install -r requirements-v013-ml.txt\n"
            f"Original import error: {exc}"
        )

    if args.folds < 3:
        raise SystemExit("ERROR: --folds must be at least 3.")

    project_id = args.project_id
    target_metric = args.target
    dataset_path = Path(args.dataset_csv)
    gate_path = Path(args.gate_json)

    print("V0.1.3-E CROSS-VALIDATION STABILITY")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print(f"folds: {args.folds}")
    print()

    # ------------------------------------------------------------
    # Modeling Gate
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
            "CROSS-VALIDATION BLOCKED BY MODELING GATE\n"
            f"decision: {gate.get('decision')}\n"
            "training_allowed: false\n"
            "No cross-validation model was trained."
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

    if usable_rows < args.folds * 2:
        raise SystemExit(
            "ERROR: too few usable rows for requested folds: "
            f"usable_rows={usable_rows}, folds={args.folds}"
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)

    # ------------------------------------------------------------
    # Candidate models
    # Same model families/hyperparameters as T11.
    # ------------------------------------------------------------
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

    splitter = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state,
    )

    model_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # Manual CV loop so every fold metric is preserved.
    # ------------------------------------------------------------
    for model_name, prototype in candidates:
        fold_results: list[dict[str, Any]] = []

        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(X),
            start=1,
        ):
            model = clone(prototype)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)

            fold_results.append(
                {
                    "fold": fold_index,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

        r2_values = [item["r2"] for item in fold_results]
        mae_values = [item["mae"] for item in fold_results]
        rmse_values = [item["rmse"] for item in fold_results]

        model_results.append(
            {
                "model_name": model_name,
                "folds": fold_results,
                "summary": {
                    "r2": summarize(r2_values),
                    "mae": summarize(mae_values),
                    "rmse": summarize(rmse_values),
                },
            }
        )

    # ------------------------------------------------------------
    # CV ranking:
    # 1) highest mean R²
    # 2) lowest mean RMSE
    # 3) lowest mean MAE
    # 4) lower R² std as final stability tie-breaker
    # ------------------------------------------------------------
    model_results.sort(
        key=lambda item: (
            -item["summary"]["r2"]["mean"],
            item["summary"]["rmse"]["mean"],
            item["summary"]["mae"]["mean"],
            item["summary"]["r2"]["std"],
        )
    )

    for rank, item in enumerate(model_results, start=1):
        item["rank"] = rank

    best = model_results[0]

    # ------------------------------------------------------------
    # Compare against T11 leaderboard when available.
    # This is informational only; T12 does not require it.
    # ------------------------------------------------------------
    t11_path = (
        Path(".runtime")
        / "v013"
        / "model_comparison"
        / f"project_{project_id}_{target_metric}"
        / "leaderboard.json"
    )

    t11_best_model = None

    if t11_path.exists():
        try:
            t11_report = load_json(t11_path)
            t11_best = t11_report.get("best_model", {})
            if isinstance(t11_best, dict):
                t11_best_model = t11_best.get("model_name")
        except Exception:
            t11_best_model = None

    same_as_t11 = (
        t11_best_model == best["model_name"]
        if t11_best_model is not None
        else None
    )

    # ------------------------------------------------------------
    # Persist report
    # ------------------------------------------------------------
    output_dir = (
        Path(".runtime")
        / "v013"
        / "cross_validation"
        / f"project_{project_id}_{target_metric}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = output_dir / "cv_report.json"
    leaderboard_csv_path = output_dir / "cv_leaderboard.csv"
    fold_csv_path = output_dir / "cv_fold_metrics.csv"

    is_fixture = "fixtures" in dataset_path.parts

    report = {
        "stage": "V0.1.3-E_cross_validation_stability",
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
        },
        "cv": {
            "method": "KFold",
            "folds": args.folds,
            "shuffle": True,
            "random_state": args.random_state,
        },
        "selection_rule": (
            "max_mean_r2_then_min_mean_rmse_then_min_mean_mae_"
            "then_min_r2_std"
        ),
        "leaderboard": model_results,
        "best_cv_model": {
            "model_name": best["model_name"],
            "r2_mean": best["summary"]["r2"]["mean"],
            "r2_std": best["summary"]["r2"]["std"],
            "mae_mean": best["summary"]["mae"]["mean"],
            "mae_std": best["summary"]["mae"]["std"],
            "rmse_mean": best["summary"]["rmse"]["mean"],
            "rmse_std": best["summary"]["rmse"]["std"],
        },
        "comparison_with_t11": {
            "t11_leaderboard_found": t11_best_model is not None,
            "t11_best_model": t11_best_model,
            "t12_best_cv_model": best["model_name"],
            "same_winner": same_as_t11,
        },
        "scientific_status": {
            "fixture_or_real_data": (
                "fixture" if is_fixture else "unknown_or_real"
            ),
            "official_scientific_conclusion_allowed": (
                (not is_fixture)
                and gate.get("official_model_allowed") is True
            ),
            "note": (
                "Cross-validation reduces dependence on one train/test split, "
                "but does not by itself establish external validity."
            ),
        },
    }

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

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
                "r2_mean",
                "r2_std",
                "r2_min",
                "r2_max",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
            ],
        )
        writer.writeheader()

        for item in model_results:
            s = item["summary"]
            writer.writerow(
                {
                    "rank": item["rank"],
                    "model_name": item["model_name"],
                    "r2_mean": s["r2"]["mean"],
                    "r2_std": s["r2"]["std"],
                    "r2_min": s["r2"]["min"],
                    "r2_max": s["r2"]["max"],
                    "mae_mean": s["mae"]["mean"],
                    "mae_std": s["mae"]["std"],
                    "rmse_mean": s["rmse"]["mean"],
                    "rmse_std": s["rmse"]["std"],
                }
            )

    with fold_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "fold",
                "train_rows",
                "test_rows",
                "r2",
                "mae",
                "rmse",
            ],
        )
        writer.writeheader()

        for item in model_results:
            for fold in item["folds"]:
                writer.writerow(
                    {
                        "model_name": item["model_name"],
                        **fold,
                    }
                )

    # ------------------------------------------------------------
    # Console output
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
    print()

    print("CV LEADERBOARD")
    for item in model_results:
        s = item["summary"]
        print(
            f"{item['rank']}. {item['model_name']} | "
            f"R2={s['r2']['mean']:.6f} ± {s['r2']['std']:.6f} | "
            f"MAE={s['mae']['mean']:.6f} ± {s['mae']['std']:.6f} | "
            f"RMSE={s['rmse']['mean']:.6f} ± {s['rmse']['std']:.6f}"
        )

    print()
    print("BEST CV MODEL")
    print(f"model_name: {best['model_name']}")
    print(
        "selection_rule: "
        "max_mean_r2_then_min_mean_rmse_then_min_mean_mae_then_min_r2_std"
    )
    print(
        f"R2: {best['summary']['r2']['mean']:.6f} "
        f"± {best['summary']['r2']['std']:.6f}"
    )
    print(
        f"MAE: {best['summary']['mae']['mean']:.6f} "
        f"± {best['summary']['mae']['std']:.6f}"
    )
    print(
        f"RMSE: {best['summary']['rmse']['mean']:.6f} "
        f"± {best['summary']['rmse']['std']:.6f}"
    )
    print()

    if t11_best_model is not None:
        print("T11 vs T12")
        print(f"T11 best model: {t11_best_model}")
        print(f"T12 best CV model: {best['model_name']}")
        print(f"same_winner: {str(bool(same_as_t11)).lower()}")
        print()

    print("OUTPUT")
    print(f"cv_report_json: {report_json_path}")
    print(f"cv_leaderboard_csv: {leaderboard_csv_path}")
    print(f"cv_fold_metrics_csv: {fold_csv_path}")
    print()

    if is_fixture:
        print(
            "NOTE: Cross-validation used a synthetic fixture. "
            "Metrics verify evaluation stability only; "
            "they are not a materials-science conclusion."
        )
        print()

    print("V0.1.3-E CROSS-VALIDATION STABILITY PASS")


if __name__ == "__main__":
    main()
