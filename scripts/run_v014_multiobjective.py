from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from optimization import (
    ApplicabilityDomainCalibrator,
    CandidateGenerator,
    MultiObjectiveError,
    diverse_select,
    load_search_space,
    non_dominated_sort,
    normalized_utilities,
    parse_objectives,
    threshold_pass,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: expected JSON object: {path}")

    return data


def standard_gate_path(project_id: int, metric: str) -> Path:
    return (
        Path(".runtime")
        / "v013"
        / "gates"
        / f"project_{project_id}_{metric}_modeling_gate.json"
    )


def standard_model_path(project_id: int, metric: str) -> Path:
    return (
        Path(".runtime")
        / "v013"
        / "model_comparison"
        / f"project_{project_id}_{metric}"
        / "best_model.joblib"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V0.1.4-T16 multi-objective Pareto optimization."
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--search-space-json", required=True)
    parser.add_argument("--objective-spec-json", required=True)
    parser.add_argument("--candidate-count", type=int, default=400)
    parser.add_argument("--max-attempts", type=int, default=80000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: ML dependencies are missing: {exc}"
        )

    project_id = args.project_id
    dataset_path = Path(args.dataset_csv)
    search_space_path = Path(args.search_space_json)
    objective_path = Path(args.objective_spec_json)

    print("V0.1.4-T16 MULTI-OBJECTIVE PARETO")
    print(f"project_id: {project_id}")
    print(f"candidate_count: {args.candidate_count}")
    print(f"random_state: {args.random_state}")
    print()

    search_space = load_search_space(
        load_json(search_space_path)
    )

    if (
        search_space.project_id is not None
        and search_space.project_id != project_id
    ):
        raise SystemExit(
            "ERROR: search-space project_id mismatch."
        )

    objective_doc = load_json(objective_path)

    if (
        objective_doc.get("project_id") is not None
        and objective_doc.get("project_id") != project_id
    ):
        raise SystemExit(
            "ERROR: objective-spec project_id mismatch."
        )

    try:
        objectives = parse_objectives(objective_doc)
    except MultiObjectiveError as exc:
        raise SystemExit(
            f"OBJECTIVE SPEC INVALID\n- {exc}"
        ) from exc

    # ------------------------------------------------------------
    # Require a PASS/allowed V0.1.3 gate + actual best model for
    # every objective.
    # ------------------------------------------------------------
    models = {}
    model_names = {}
    feature_columns = None

    for objective in objectives:
        gate_path = standard_gate_path(
            project_id,
            objective.metric,
        )
        gate = load_json(gate_path)

        if gate.get("stage") != "V0.1.3-B_modeling_gate":
            raise SystemExit(
                f"ERROR: invalid gate for {objective.metric}"
            )

        if gate.get("training_allowed") is not True:
            raise SystemExit(
                "T16 BLOCKED BY MODELING GATE\n"
                f"metric: {objective.metric}\n"
                f"decision: {gate.get('decision')}"
            )

        model_path = standard_model_path(
            project_id,
            objective.metric,
        )
        if not model_path.exists():
            raise SystemExit(
                f"ERROR: best model missing for {objective.metric}: "
                f"{model_path}"
            )

        bundle = joblib.load(model_path)

        if not isinstance(bundle, dict):
            raise SystemExit(
                f"ERROR: invalid model bundle for {objective.metric}"
            )

        if bundle.get("project_id") != project_id:
            raise SystemExit(
                f"ERROR: model project_id mismatch for {objective.metric}"
            )

        if bundle.get("target_metric") != objective.metric:
            raise SystemExit(
                f"ERROR: model metric mismatch for {objective.metric}"
            )

        cols = bundle.get("feature_columns")
        if not isinstance(cols, list) or not cols:
            raise SystemExit(
                f"ERROR: model feature_columns missing for {objective.metric}"
            )

        if feature_columns is None:
            feature_columns = list(cols)
        elif list(cols) != feature_columns:
            raise SystemExit(
                "ERROR: T16 fixture requires all objective models to use "
                "the same ordered feature_columns."
            )

        models[objective.metric] = bundle["model"]
        model_names[objective.metric] = bundle.get("model_name")

    missing_features = [
        col for col in feature_columns
        if col not in search_space.variable_map
    ]
    if missing_features:
        raise SystemExit(
            f"ERROR: search space missing model features: {missing_features}"
        )

    # ------------------------------------------------------------
    # One AD calibration because both objective models use the same
    # design features and same training dataset.
    # ------------------------------------------------------------
    ad = ApplicabilityDomainCalibrator.from_csv(
        dataset_path,
        feature_columns=feature_columns,
    )

    # ------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------
    generation = CandidateGenerator(
        search_space,
        random_state=args.random_state,
    ).generate(
        candidate_count=args.candidate_count,
        max_attempts=args.max_attempts,
    )

    if not generation["generation_complete"]:
        raise SystemExit(
            "CANDIDATE GENERATION INCOMPLETE\n"
            f"requested={generation['requested_count']}\n"
            f"generated={generation['generated_count']}"
        )

    candidates = generation["candidates"]

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

    predictions_by_metric = {
        objective.metric: models[objective.metric].predict(X)
        for objective in objectives
    }

    evaluated = []

    for row_index, candidate in enumerate(candidates):
        predictions = {
            objective.metric: float(
                predictions_by_metric[objective.metric][row_index]
            )
            for objective in objectives
        }

        ad_report = ad.evaluate(candidate["features"])

        threshold_results = {
            objective.metric: threshold_pass(
                predictions[objective.metric],
                objective,
            )
            for objective in objectives
        }

        evaluated.append(
            {
                **candidate,
                "predictions": predictions,
                "applicability_domain": ad_report,
                "threshold_results": threshold_results,
                "all_target_thresholds_pass": all(
                    threshold_results.values()
                ),
                "trusted_for_optimization": (
                    ad_report["status"] == "IN_DOMAIN"
                ),
            }
        )

    trusted = [
        row for row in evaluated
        if row["trusted_for_optimization"]
    ]

    qualified = [
        row for row in trusted
        if row["all_target_thresholds_pass"]
    ]

    if len(qualified) < 2:
        raise SystemExit(
            "T16 INSUFFICIENT QUALIFIED CANDIDATES\n"
            f"trusted={len(trusted)}\n"
            f"qualified={len(qualified)}\n"
            "Relax objective thresholds or enlarge candidate count."
        )

    prediction_rows = [
        row["predictions"]
        for row in qualified
    ]

    pareto_ranks = non_dominated_sort(
        prediction_rows,
        objectives,
    )

    utilities = normalized_utilities(
        prediction_rows,
        objectives,
    )

    soft_penalty_weight = float(
        objective_doc.get("soft_penalty_weight", 0.20)
    )

    for row, pareto_rank, utility in zip(
        qualified,
        pareto_ranks,
        utilities,
    ):
        row["pareto_rank"] = int(pareto_rank)
        row["base_utility"] = float(utility)
        row["adjusted_utility"] = float(
            utility
            - soft_penalty_weight * row["soft_penalty"]
        )

    pareto_front = [
        row for row in qualified
        if row["pareto_rank"] == 1
    ]

    recommendation_count = int(
        objective_doc.get("recommendation_count", 5)
    )
    diversity_weight = float(
        objective_doc.get("diversity_weight", 0.35)
    )

    source_pool = (
        pareto_front
        if len(pareto_front) >= recommendation_count
        else sorted(
            qualified,
            key=lambda row: (
                row["pareto_rank"],
                -row["adjusted_utility"],
            ),
        )
    )

    recommendations = diverse_select(
        source_pool,
        count=min(recommendation_count, len(source_pool)),
        feature_columns=feature_columns,
        utility_key="adjusted_utility",
        diversity_weight=diversity_weight,
    )

    recommended_ids = {
        row["candidate_id"]
        for row in recommendations
    }

    for row in qualified:
        row["recommended"] = (
            row["candidate_id"] in recommended_ids
        )

    # ------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------
    out = (
        Path(".runtime")
        / "v014"
        / "multiobjective"
        / f"project_{project_id}"
    )
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "t16_multiobjective_report.json"
    qualified_csv = out / "qualified_candidates.csv"
    pareto_csv = out / "pareto_front.csv"
    recommendation_csv = out / "recommended_designs.csv"

    report = {
        "stage": "V0.1.4-T16_multiobjective_pareto",
        "project_id": project_id,
        "dataset_csv": str(dataset_path),
        "search_space_json": str(search_space_path),
        "objective_spec_json": str(objective_path),
        "objectives": [
            {
                "metric": objective.metric,
                "direction": objective.direction,
                "threshold_operator": objective.threshold_operator,
                "threshold_value": objective.threshold_value,
                "weight": objective.weight,
                "model_name": model_names[objective.metric],
                "model_path": str(
                    standard_model_path(
                        project_id,
                        objective.metric,
                    )
                ),
            }
            for objective in objectives
        ],
        "generation": {
            key: value
            for key, value in generation.items()
            if key != "candidates"
        },
        "counts": {
            "generated_hard_valid": len(evaluated),
            "trusted_in_domain": len(trusted),
            "qualified_all_target_thresholds": len(qualified),
            "pareto_front": len(pareto_front),
            "recommended": len(recommendations),
        },
        "ranking_policy": {
            "pareto_definition": (
                "standard_non_dominated_sort_on_raw_predictions"
            ),
            "soft_penalty_applied_to_pareto": False,
            "soft_penalty_weight_for_utility": soft_penalty_weight,
            "diversity_weight": diversity_weight,
            "recommendation_count": recommendation_count,
            "note": (
                "Soft constraints affect utility/recommendation order but "
                "do not alter mathematical Pareto dominance."
            ),
        },
        "applicability_domain_calibration": ad.summary(),
        "qualified_candidates": qualified,
        "pareto_front": pareto_front,
        "recommendations": recommendations,
        "scientific_status": {
            "fixture_or_real_data": (
                "fixture"
                if "fixtures" in dataset_path.parts
                else "unknown_or_real"
            ),
            "official_scientific_recommendation_allowed": False,
            "note": (
                "T16 synthetic fixture validates multi-objective optimization "
                "plumbing only."
            ),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = (
            ["candidate_id", "pareto_rank", "base_utility",
             "adjusted_utility", "soft_penalty", "ad_status"]
            + [f"predicted::{obj.metric}" for obj in objectives]
            + feature_columns
        )

        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for row in rows:
                payload = {
                    "candidate_id": row["candidate_id"],
                    "pareto_rank": row.get("pareto_rank"),
                    "base_utility": row.get("base_utility"),
                    "adjusted_utility": row.get("adjusted_utility"),
                    "soft_penalty": row["soft_penalty"],
                    "ad_status": row["applicability_domain"]["status"],
                }

                for obj in objectives:
                    payload[f"predicted::{obj.metric}"] = (
                        row["predictions"][obj.metric]
                    )

                for col in feature_columns:
                    payload[col] = row["features"][col]

                writer.writerow(payload)

    write_rows(qualified_csv, qualified)
    write_rows(pareto_csv, pareto_front)
    write_rows(recommendation_csv, recommendations)

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
    print("OBJECTIVES")
    for objective in objectives:
        threshold_text = (
            f"{objective.threshold_operator} "
            f"{objective.threshold_value}"
            if objective.threshold_operator
            else "none"
        )
        print(
            f"- {objective.metric}: {objective.direction}, "
            f"threshold={threshold_text}, "
            f"model={model_names[objective.metric]}"
        )
    print()

    print("CANDIDATE FLOW")
    print(f"generated_hard_valid: {len(evaluated)}")
    print(f"trusted_in_domain: {len(trusted)}")
    print(
        "qualified_all_target_thresholds: "
        f"{len(qualified)}"
    )
    print(f"pareto_front: {len(pareto_front)}")
    print(f"recommended: {len(recommendations)}")
    print()

    print("PARETO FRONT PREVIEW")
    for row in sorted(
        pareto_front,
        key=lambda x: -x["adjusted_utility"],
    )[:10]:
        prediction_text = " | ".join(
            f"{obj.metric}={row['predictions'][obj.metric]:.6f}"
            for obj in objectives
        )
        print(
            f"- {row['candidate_id']} | "
            f"{prediction_text} | "
            f"soft_penalty={row['soft_penalty']:.6f}"
        )
    print()

    print("DIVERSE RECOMMENDATIONS")
    for index, row in enumerate(recommendations, start=1):
        prediction_text = " | ".join(
            f"{obj.metric}={row['predictions'][obj.metric]:.6f}"
            for obj in objectives
        )
        print(
            f"{index}. {row['candidate_id']} | "
            f"{prediction_text} | "
            f"pareto_rank={row['pareto_rank']} | "
            f"soft_penalty={row['soft_penalty']:.6f}"
        )
    print()

    print("OUTPUT")
    print(f"report_json: {report_path}")
    print(f"qualified_csv: {qualified_csv}")
    print(f"pareto_front_csv: {pareto_csv}")
    print(f"recommended_designs_csv: {recommendation_csv}")
    print()
    print(
        "NOTE: T16 uses synthetic data and is not a materials-science "
        "recommendation."
    )
    print()
    print("V0.1.4-T16 MULTI-OBJECTIVE PARETO PASS")


if __name__ == "__main__":
    main()
