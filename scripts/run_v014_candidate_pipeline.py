from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from optimization import (
    ApplicabilityDomainCalibrator,
    ApplicabilityDomainError,
    CandidateGenerationError,
    CandidateGenerator,
    SearchSpaceError,
    load_search_space,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: expected JSON object: {path}")

    return data


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    target_metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "candidate_id",
        "constraint_status",
        "soft_penalty",
        f"predicted::{target_metric}",
        "ad_status",
        "risk",
        "selection_bucket",
        "optimization_eligible",
    ] + feature_columns

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            payload = {
                "candidate_id": row["candidate_id"],
                "constraint_status": row["constraint_status"],
                "soft_penalty": row["soft_penalty"],
                f"predicted::{target_metric}": row["prediction"],
                "ad_status": row["applicability_domain"]["status"],
                "risk": row["applicability_domain"]["risk"],
                "selection_bucket": row["selection_bucket"],
                "optimization_eligible": row["optimization_eligible"],
            }
            payload.update(
                {
                    col: row["features"].get(col)
                    for col in feature_columns
                }
            )
            writer.writerow(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V0.1.4-T15 candidate generation + prediction + "
            "applicability-domain filtering."
        )
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--search-space-json", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--candidate-count", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "ERROR: ML dependencies are missing.\n"
            "Install requirements first.\n"
            f"Original error: {exc}"
        )

    project_id = args.project_id
    target_metric = args.target
    space_path = Path(args.search_space_json)
    dataset_path = Path(args.dataset_csv)
    gate_path = Path(args.gate_json)
    model_path = Path(args.best_model)

    print("V0.1.4-T15 CANDIDATE GENERATION PIPELINE")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print(f"candidate_count: {args.candidate_count}")
    print(f"random_state: {args.random_state}")
    print()

    # ------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------
    gate = load_json(gate_path)

    if gate.get("stage") != "V0.1.3-B_modeling_gate":
        raise SystemExit(
            "ERROR: gate JSON is not a V0.1.3-B Modeling Gate report."
        )

    if gate.get("project_id") != project_id:
        raise SystemExit("ERROR: project_id mismatch in gate report.")

    if gate.get("target_metric") != target_metric:
        raise SystemExit("ERROR: target mismatch in gate report.")

    if gate.get("training_allowed") is not True:
        raise SystemExit(
            "T15 BLOCKED BY MODELING GATE\n"
            f"decision: {gate.get('decision')}\n"
            "training_allowed: false\n"
            "No optimization candidate pool was generated."
        )

    # ------------------------------------------------------------
    # Search space
    # ------------------------------------------------------------
    try:
        search_space = load_search_space(load_json(space_path))
    except SearchSpaceError as exc:
        raise SystemExit(f"SEARCH SPACE INVALID\n- {exc}") from exc

    if (
        search_space.project_id is not None
        and search_space.project_id != project_id
    ):
        raise SystemExit(
            "ERROR: project_id mismatch between search space and command."
        )

    # ------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------
    if not model_path.exists():
        raise SystemExit(f"ERROR: best model not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise SystemExit("ERROR: invalid best-model bundle.")

    if bundle.get("project_id") != project_id:
        raise SystemExit(
            "ERROR: project_id mismatch between model and command."
        )

    if bundle.get("target_metric") != target_metric:
        raise SystemExit(
            "ERROR: target mismatch between model and command."
        )

    model = bundle.get("model")
    model_name = bundle.get("model_name")
    feature_columns = bundle.get("feature_columns")

    if model is None:
        raise SystemExit("ERROR: best-model bundle contains no model.")

    if not isinstance(feature_columns, list) or not feature_columns:
        raise SystemExit(
            "ERROR: best-model bundle contains no feature_columns."
        )

    missing_model_features = [
        col for col in feature_columns
        if col not in search_space.variable_map
    ]
    if missing_model_features:
        raise SystemExit(
            "ERROR: search space is missing model features: "
            f"{missing_model_features}"
        )

    non_numeric_model_features = [
        col for col in feature_columns
        if search_space.variable_map[col].kind == "categorical"
    ]
    if non_numeric_model_features:
        raise SystemExit(
            "ERROR: T15 当前版本只支持数值型模型特征；"
            f"发现 categorical model features: {non_numeric_model_features}"
        )

    # ------------------------------------------------------------
    # Applicability Domain calibration
    # ------------------------------------------------------------
    try:
        ad = ApplicabilityDomainCalibrator.from_csv(
            dataset_path,
            feature_columns=feature_columns,
        )
    except ApplicabilityDomainError as exc:
        raise SystemExit(
            f"APPLICABILITY DOMAIN CALIBRATION FAILED\n- {exc}"
        ) from exc

    # ------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------
    generator = CandidateGenerator(
        search_space,
        random_state=args.random_state,
    )

    try:
        generation = generator.generate(
            candidate_count=args.candidate_count,
            max_attempts=args.max_attempts,
        )
    except CandidateGenerationError as exc:
        raise SystemExit(f"CANDIDATE GENERATION FAILED\n- {exc}") from exc

    if not generation["generation_complete"]:
        raise SystemExit(
            "CANDIDATE GENERATION INCOMPLETE\n"
            f"requested={generation['requested_count']}\n"
            f"generated={generation['generated_count']}\n"
            f"attempts={generation['attempts']}\n"
            "Increase --max-attempts or relax HARD constraints."
        )

    candidates = generation["candidates"]

    # ------------------------------------------------------------
    # Real model prediction + AD classification
    # ------------------------------------------------------------
    X = np.asarray(
        [
            [
                float(candidate["features"][col])
                for col in feature_columns
            ]
            for candidate in candidates
        ],
        dtype=float,
    )

    predictions = model.predict(X)

    evaluated = []

    for candidate, prediction in zip(candidates, predictions):
        ad_report = ad.evaluate(candidate["features"])

        if ad_report["status"] == "IN_DOMAIN":
            bucket = "TRUSTED_IN_DOMAIN"
            optimization_eligible = True
        elif ad_report["status"] == "BORDERLINE":
            bucket = "REVIEW_BORDERLINE"
            optimization_eligible = False
        else:
            bucket = "EXCLUDED_OUT_OF_DOMAIN"
            optimization_eligible = False

        evaluated.append(
            {
                **candidate,
                "model_name": model_name,
                "prediction": float(prediction),
                "target_metric": target_metric,
                "applicability_domain": ad_report,
                "selection_bucket": bucket,
                "optimization_eligible": optimization_eligible,
            }
        )

    trusted = [
        row for row in evaluated
        if row["selection_bucket"] == "TRUSTED_IN_DOMAIN"
    ]
    borderline = [
        row for row in evaluated
        if row["selection_bucket"] == "REVIEW_BORDERLINE"
    ]
    excluded = [
        row for row in evaluated
        if row["selection_bucket"] == "EXCLUDED_OUT_OF_DOMAIN"
    ]

    # T15 is not a recommendation stage. This preview is diagnostic only.
    trusted_preview = sorted(
        trusted,
        key=lambda row: row["prediction"],
        reverse=True,
    )[:5]

    # ------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------
    out = (
        Path(".runtime")
        / "v014"
        / "candidate_generation"
        / f"project_{project_id}_{target_metric}"
    )
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "candidate_pipeline_report.json"
    all_csv = out / "candidate_pool.csv"
    trusted_csv = out / "trusted_in_domain.csv"
    borderline_csv = out / "review_borderline.csv"
    excluded_csv = out / "excluded_out_of_domain.csv"

    is_fixture = (
        "fixtures" in dataset_path.parts
        or "fixtures" in space_path.parts
    )

    report = {
        "stage": "V0.1.4-T15_candidate_generation_pipeline",
        "project_id": project_id,
        "target_metric": target_metric,
        "search_space_json": str(space_path),
        "dataset_csv": str(dataset_path),
        "gate_json": str(gate_path),
        "best_model": str(model_path),
        "model_name": model_name,
        "feature_columns": feature_columns,
        "generation": {
            key: value
            for key, value in generation.items()
            if key != "candidates"
        },
        "applicability_domain_calibration": ad.summary(),
        "counts": {
            "generated_hard_valid": len(evaluated),
            "trusted_in_domain": len(trusted),
            "review_borderline": len(borderline),
            "excluded_out_of_domain": len(excluded),
            "soft_penalty_candidates": sum(
                row["soft_penalty"] > 0
                for row in evaluated
            ),
        },
        "prediction_summary": {
            "all_min": min(row["prediction"] for row in evaluated),
            "all_max": max(row["prediction"] for row in evaluated),
            "all_mean": mean(row["prediction"] for row in evaluated),
            "trusted_min": (
                min(row["prediction"] for row in trusted)
                if trusted else None
            ),
            "trusted_max": (
                max(row["prediction"] for row in trusted)
                if trusted else None
            ),
        },
        "trusted_prediction_preview_not_final_recommendation": [
            {
                "candidate_id": row["candidate_id"],
                "prediction": row["prediction"],
                "soft_penalty": row["soft_penalty"],
                "features": row["features"],
            }
            for row in trusted_preview
        ],
        "candidates": evaluated,
        "scientific_status": {
            "fixture_or_real_data": (
                "fixture" if is_fixture else "unknown_or_real"
            ),
            "official_recommendation_allowed": False,
            "note": (
                "T15 only builds and filters a candidate pool. "
                "It does not perform Pareto or Bayesian optimization."
            ),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_csv(
        all_csv,
        evaluated,
        feature_columns=list(search_space.variable_map),
        target_metric=target_metric,
    )
    write_csv(
        trusted_csv,
        trusted,
        feature_columns=list(search_space.variable_map),
        target_metric=target_metric,
    )
    write_csv(
        borderline_csv,
        borderline,
        feature_columns=list(search_space.variable_map),
        target_metric=target_metric,
    )
    write_csv(
        excluded_csv,
        excluded,
        feature_columns=list(search_space.variable_map),
        target_metric=target_metric,
    )

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
    print("GATE")
    print(f"decision: {gate.get('decision')}")
    print("training_allowed: true")
    print()

    print("SEARCH SPACE")
    summary = search_space.summary()
    print(f"name: {summary['name']}")
    print(f"variable_count: {summary['variable_count']}")
    print(f"hard_constraint_count: {summary['hard_constraint_count']}")
    print(f"soft_constraint_count: {summary['soft_constraint_count']}")
    print()

    print("GENERATION")
    print(f"requested: {generation['requested_count']}")
    print(f"generated_hard_valid: {generation['generated_count']}")
    print(f"attempts: {generation['attempts']}")
    print(f"acceptance_rate: {generation['acceptance_rate']:.2%}")
    print(f"duplicates_skipped: {generation['duplicate_count']}")
    print()

    print("MODEL")
    print(f"model_name: {model_name}")
    print(f"feature_count: {len(feature_columns)}")
    print()

    print("APPLICABILITY DOMAIN")
    print(f"IN_DOMAIN trusted: {len(trusted)}")
    print(f"BORDERLINE review: {len(borderline)}")
    print(f"OUT_OF_DOMAIN excluded: {len(excluded)}")
    print()

    print("SOFT CONSTRAINTS")
    print(
        "candidates_with_soft_penalty: "
        f"{sum(row['soft_penalty'] > 0 for row in evaluated)}"
    )
    print()

    if trusted_preview:
        print("TRUSTED PREDICTION PREVIEW (NOT FINAL RECOMMENDATION)")
        for row in trusted_preview:
            print(
                f"- {row['candidate_id']} | "
                f"predicted_{target_metric}={row['prediction']:.6f} | "
                f"soft_penalty={row['soft_penalty']:.6f}"
            )
        print()

    print("OUTPUT")
    print(f"report_json: {report_path}")
    print(f"candidate_pool_csv: {all_csv}")
    print(f"trusted_in_domain_csv: {trusted_csv}")
    print(f"review_borderline_csv: {borderline_csv}")
    print(f"excluded_out_of_domain_csv: {excluded_csv}")
    print()

    if is_fixture:
        print(
            "NOTE: T15 used synthetic fixture data. "
            "Generated candidates verify the engineering pipeline only; "
            "they are not materials-science recommendations."
        )
        print()

    print("V0.1.4-T15 CANDIDATE GENERATION PIPELINE PASS")


if __name__ == "__main__":
    main()
