from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from optimization import (
    ApplicabilityDomainCalibrator,
    BOConfig,
    BayesianOptimizationError,
    CandidateGenerator,
    GaussianProcessBayesianOptimizer,
    filter_already_observed_candidate_indices,
    load_search_space,
)


REQUEST_STAGE = "V0.1.4-T18_bayesian_optimization_request"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit(
            f"ERROR: expected JSON object: {path}"
        )

    return data


def parse_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"ERROR: {name} 不是数值: {value!r}"
        ) from exc

    if not np.isfinite(result):
        raise SystemExit(
            f"ERROR: {name} 不是有限数值"
        )

    return result


def load_observations(
    path: Path,
    *,
    feature_columns: list[str],
    target_metric: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: observations CSV not found: {path}"
        )

    target_column = f"target::{target_metric}"

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise SystemExit(
                "ERROR: observations CSV has no header"
            )

        required = feature_columns + [target_column]
        missing = [
            col for col in required
            if col not in reader.fieldnames
        ]
        if missing:
            raise SystemExit(
                f"ERROR: observations CSV missing columns: {missing}"
            )

        rows = list(reader)

    X = []
    y = []
    clean_rows = []

    for row in rows:
        vector = [
            parse_float(row.get(col), col)
            for col in feature_columns
        ]
        target = parse_float(
            row.get(target_column),
            target_column,
        )

        X.append(vector)
        y.append(target)
        clean_rows.append(row)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        clean_rows,
    )




def main() -> None:
    parser = argparse.ArgumentParser(
        description="V0.1.4-T18 Gaussian Process Bayesian Optimization."
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--observations-csv", required=True)
    parser.add_argument("--search-space-json", required=True)
    parser.add_argument(
        "--runtime-root",
        default=".runtime",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    request_path = Path(args.request_json)
    observations_path = Path(args.observations_csv)
    search_space_path = Path(args.search_space_json)
    runtime_root = Path(args.runtime_root)

    request = load_json(request_path)

    if request.get("stage") != REQUEST_STAGE:
        raise SystemExit(
            f"ERROR: request stage must be {REQUEST_STAGE}"
        )

    project_id = request.get("project_id")
    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise SystemExit("ERROR: project_id must be integer")

    target_metric = str(
        request.get("target_metric") or ""
    ).strip()
    if not target_metric:
        raise SystemExit("ERROR: target_metric is empty")

    direction = str(
        request.get("direction") or "maximize"
    ).strip().lower()

    acquisition = str(
        request.get("acquisition") or "EI"
    ).strip().upper()

    batch_size = int(
        request.get("batch_size", 5)
    )

    candidate_count = (
        args.candidate_count
        if args.candidate_count is not None
        else int(request.get("candidate_count", 900))
    )

    random_state = (
        args.random_state
        if args.random_state is not None
        else int(request.get("random_state", 42))
    )

    max_attempts = int(
        request.get(
            "max_attempts",
            max(10000, candidate_count * 100),
        )
    )

    xi = float(request.get("xi", 0.01))
    kappa = float(request.get("kappa", 2.0))
    min_batch_distance = float(
        request.get("min_batch_distance", 0.20)
    )
    allow_borderline = bool(
        request.get(
            "allow_borderline_for_exploration",
            True,
        )
    )
    soft_penalty_weight = float(
        request.get("soft_penalty_weight", 0.10)
    )

    print("V0.1.4-T18 BAYESIAN OPTIMIZATION")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print(f"direction: {direction}")
    print(f"acquisition: {acquisition}")
    print(f"batch_size: {batch_size}")
    print(f"candidate_count: {candidate_count}")
    print(f"random_state: {random_state}")
    print()

    # ------------------------------------------------------------
    # Modeling Gate
    # ------------------------------------------------------------
    gate_path = (
        runtime_root
        / "v013"
        / "gates"
        / f"project_{project_id}_{target_metric}_modeling_gate.json"
    )
    gate = load_json(gate_path)

    if gate.get("stage") != "V0.1.3-B_modeling_gate":
        raise SystemExit("ERROR: invalid Modeling Gate report")
    if gate.get("project_id") != project_id:
        raise SystemExit("ERROR: gate project_id mismatch")
    if gate.get("target_metric") != target_metric:
        raise SystemExit("ERROR: gate target_metric mismatch")
    if gate.get("training_allowed") is not True:
        raise SystemExit(
            "T18 BLOCKED BY MODELING GATE\n"
            f"decision: {gate.get('decision')}\n"
            "training_allowed: false"
        )
    if gate.get("official_model_allowed") is not True:
        raise SystemExit(
            "T18 BLOCKED BY MODELING GATE\n"
            "official_model_allowed: false"
        )

    # ------------------------------------------------------------
    # Search space + observations
    # ------------------------------------------------------------
    search_space = load_search_space(
        load_json(search_space_path)
    )

    if (
        search_space.project_id is not None
        and search_space.project_id != project_id
    ):
        raise SystemExit(
            "ERROR: search space project_id mismatch"
        )

    feature_columns = [
        spec.name
        for spec in search_space.variables
        if spec.kind in {"continuous", "integer"}
    ]

    if not feature_columns:
        raise SystemExit(
            "ERROR: T18 requires numeric search-space features"
        )

    X_obs, y_obs, observation_rows = load_observations(
        observations_path,
        feature_columns=feature_columns,
        target_metric=target_metric,
    )

    if len(X_obs) < 10:
        raise SystemExit(
            "ERROR: T18 requires at least 10 observations"
        )

    # ------------------------------------------------------------
    # AD calibration from observations only.
    # No hidden fixture truth is used.
    # ------------------------------------------------------------
    ad = ApplicabilityDomainCalibrator(
        feature_columns=feature_columns,
        X=X_obs,
        dropped_rows=0,
    )

    # ------------------------------------------------------------
    # Generate constrained candidate pool.
    # ------------------------------------------------------------
    generation = CandidateGenerator(
        search_space,
        random_state=random_state,
        id_prefix="V014_T18",
    ).generate(
        candidate_count=candidate_count,
        max_attempts=max_attempts,
    )

    if not generation["generation_complete"]:
        raise SystemExit(
            "CANDIDATE GENERATION INCOMPLETE"
        )

    generated_candidates = generation["candidates"]
    X_generated = np.asarray(
        [
            [
                float(candidate["features"][col])
                for col in feature_columns
            ]
            for candidate in generated_candidates
        ],
        dtype=float,
    )

    keep_indices, duplicate_indices = (
        filter_already_observed_candidate_indices(
            X_generated,
            X_obs,
        )
    )

    already_observed = len(duplicate_indices)
    candidate_records = []
    ad_out_excluded = 0
    borderline_kept = 0

    for candidate_index in keep_indices:
        candidate = generated_candidates[candidate_index]

        ad_report = ad.evaluate(
            candidate["features"]
        )

        if ad_report["status"] == "OUT_OF_DOMAIN":
            ad_out_excluded += 1
            continue

        if (
            ad_report["status"] == "BORDERLINE"
            and not allow_borderline
        ):
            ad_out_excluded += 1
            continue

        if ad_report["status"] == "BORDERLINE":
            borderline_kept += 1

        candidate_records.append(
            {
                **candidate,
                "applicability_domain": ad_report,
            }
        )

    if len(candidate_records) < batch_size:
        raise SystemExit(
            "T18 INSUFFICIENT SAFE CANDIDATES\n"
            f"eligible={len(candidate_records)}\n"
            f"batch_size={batch_size}"
        )

    X_candidates = np.asarray(
        [
            [
                float(row["features"][col])
                for col in feature_columns
            ]
            for row in candidate_records
        ],
        dtype=float,
    )

    candidate_ids = [
        row["candidate_id"]
        for row in candidate_records
    ]

    # ------------------------------------------------------------
    # Real GP Bayesian Optimization.
    # ------------------------------------------------------------
    try:
        optimizer = GaussianProcessBayesianOptimizer(
            X_obs,
            y_obs,
            config=BOConfig(
                acquisition=acquisition,
                direction=direction,
                xi=xi,
                kappa=kappa,
                batch_size=batch_size,
                min_batch_distance=min_batch_distance,
                random_state=random_state,
            ),
        )
        fit_summary = optimizer.fit_summary()
        candidate_penalties = np.asarray(
            [
                float(row["soft_penalty"])
                for row in candidate_records
            ],
            dtype=float,
        )

        bo_result = optimizer.propose_batch(
            X_candidates,
            candidate_ids,
            candidate_penalties=candidate_penalties,
            penalty_weight=soft_penalty_weight,
        )
    except BayesianOptimizationError as exc:
        raise SystemExit(
            f"T18 BAYESIAN OPTIMIZATION FAILED\n- {exc}"
        ) from exc

    selected_records = []

    for round_info in bo_result["rounds"]:
        idx = int(round_info["candidate_index"])
        candidate = candidate_records[idx]

        selected_records.append(
            {
                "round": round_info["round"],
                "candidate_id": candidate["candidate_id"],
                "features": candidate["features"],
                "constraint_status": candidate["constraint_status"],
                "soft_penalty": candidate["soft_penalty"],
                "soft_violations": candidate["soft_violations"],
                "applicability_domain": candidate["applicability_domain"],
                "posterior_mean": round_info["posterior_mean"],
                "posterior_std": round_info["posterior_std"],
                "acquisition_value": round_info["acquisition_value"],
                "adjusted_acquisition": round_info["adjusted_acquisition"],
                "min_standardized_distance_to_selected": (
                    round_info[
                        "min_standardized_distance_to_selected"
                    ]
                ),
                "diversity_threshold_relaxed": (
                    round_info["diversity_threshold_relaxed"]
                ),
                "fitted_kernel_for_round": round_info["fitted_kernel"],
                "selection_reason": (
                    f"adjusted {acquisition} = raw acquisition - "
                    "soft_penalty_weight × soft_penalty; "
                    "selected by Kriging Believer batch update"
                ),
            }
        )

    # ------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------
    request_name = str(
        request.get("request_name")
        or f"{target_metric}_next_experiments"
    )

    out = (
        runtime_root
        / "v014"
        / "bayesian_optimization"
        / f"project_{project_id}"
        / request_name
    )
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "bo_report.json"
    next_csv = out / "next_experiments.csv"

    report = {
        "stage": "V0.1.4-T18_bayesian_optimization",
        "status": "SUCCESS",
        "project_id": project_id,
        "target_metric": target_metric,
        "request": request,
        "gate": {
            "path": str(gate_path),
            "decision": gate.get("decision"),
            "training_allowed": gate.get("training_allowed"),
            "official_model_allowed": gate.get("official_model_allowed"),
        },
        "observations": {
            "path": str(observations_path),
            "rows": int(len(X_obs)),
            "best_observed": fit_summary["best_observed"],
            "feature_columns": feature_columns,
        },
        "gaussian_process": fit_summary,
        "candidate_generation": {
            key: value
            for key, value in generation.items()
            if key != "candidates"
        },
        "candidate_filtering": {
            "hard_valid_generated": generation["generated_count"],
            "already_observed_filtered": already_observed,
            "out_of_domain_excluded": ad_out_excluded,
            "borderline_kept_for_exploration": borderline_kept,
            "eligible_for_bo": len(candidate_records),
            "allow_borderline_for_exploration": allow_borderline,
        },
        "bayesian_optimization": {
            "batch_strategy": bo_result["batch_strategy"],
            "acquisition": bo_result["acquisition"],
            "direction": bo_result["direction"],
            "xi": bo_result["xi"],
            "kappa": bo_result["kappa"],
            "batch_size": bo_result["batch_size"],
            "min_batch_distance": bo_result[
                "min_batch_distance"
            ],
            "soft_penalty_weight": soft_penalty_weight,
            "selection_score": bo_result["selection_score"],
        },
        "next_experiments": selected_records,
        "safety": {
            "gate_required": True,
            "official_model_allowed_required": True,
            "hard_constraints_required": True,
            "already_observed_candidates_removed": True,
            "out_of_domain_candidates_excluded": True,
            "borderline_candidates_allowed_for_controlled_exploration": (
                allow_borderline
            ),
            "objective_values_are_gp_posterior_predictions": True,
            "future_experimental_results_are_not_fabricated": True,
        },
        "scientific_status": {
            "fixture_or_real_data": (
                "fixture"
                if "fixtures" in observations_path.parts
                else "unknown_or_real"
            ),
            "note": (
                "T18 only proposes which experiments to run next. "
                "Posterior means are model estimates, not measured results."
            ),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    fields = [
        "round",
        "candidate_id",
        "posterior_mean",
        "posterior_std",
        "acquisition_value",
        "adjusted_acquisition",
        "ad_status",
        "ad_risk",
        "soft_penalty",
        "min_standardized_distance_to_selected",
        *feature_columns,
    ]

    with next_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in selected_records:
            payload = {
                "round": row["round"],
                "candidate_id": row["candidate_id"],
                "posterior_mean": row["posterior_mean"],
                "posterior_std": row["posterior_std"],
                "acquisition_value": row["acquisition_value"],
                "adjusted_acquisition": row["adjusted_acquisition"],
                "ad_status": row[
                    "applicability_domain"
                ]["status"],
                "ad_risk": row[
                    "applicability_domain"
                ]["risk"],
                "soft_penalty": row["soft_penalty"],
                "min_standardized_distance_to_selected": row[
                    "min_standardized_distance_to_selected"
                ],
            }
            payload.update(row["features"])
            writer.writerow(payload)

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------
    print("GATE")
    print(f"decision: {gate.get('decision')}")
    print("training_allowed: true")
    print("official_model_allowed: true")
    print()

    print("OBSERVATIONS")
    print(f"rows: {len(X_obs)}")
    print(
        f"best_observed_{target_metric}: "
        f"{fit_summary['best_observed']:.6f}"
    )
    print(f"feature_count: {len(feature_columns)}")
    print()

    print("GAUSSIAN PROCESS")
    print(f"kernel: {fit_summary['fitted_kernel']}")
    print(
        "log_marginal_likelihood: "
        f"{fit_summary['log_marginal_likelihood']:.6f}"
    )
    print()

    print("CANDIDATE FLOW")
    print(
        f"generated_hard_valid: {generation['generated_count']}"
    )
    print(
        f"already_observed_filtered: {already_observed}"
    )
    print(
        f"out_of_domain_excluded: {ad_out_excluded}"
    )
    print(
        f"borderline_kept_for_exploration: {borderline_kept}"
    )
    print(
        f"eligible_for_bo: {len(candidate_records)}"
    )
    print()

    print("ACQUISITION")
    print(f"type: {acquisition}")
    print(f"batch_strategy: {bo_result['batch_strategy']}")
    print(
        "selection_score: adjusted acquisition "
        "= raw acquisition - soft penalty"
    )
    if acquisition in {"EI", "PI"}:
        print(f"xi: {xi}")
    if acquisition == "UCB":
        print(f"kappa: {kappa}")
    print()

    print("NEXT EXPERIMENTS")
    for row in selected_records:
        print(
            f"{row['round']}. {row['candidate_id']} | "
            f"posterior_mean={row['posterior_mean']:.6f} | "
            f"posterior_std={row['posterior_std']:.6f} | "
            f"{acquisition}={row['acquisition_value']:.6f} | "
            f"adjusted_{acquisition}={row['adjusted_acquisition']:.6f} | "
            f"AD={row['applicability_domain']['status']} | "
            f"soft_penalty={row['soft_penalty']:.6f}"
        )
    print()

    print("OUTPUT")
    print(f"report_json: {report_path}")
    print(f"next_experiments_csv: {next_csv}")
    print()

    print(
        "NOTE: posterior_mean is a GP model estimate, not an observed "
        "experimental result. T18 does not fabricate future measurements."
    )
    print()

    print("V0.1.4-T18 BAYESIAN OPTIMIZATION PASS")


if __name__ == "__main__":
    main()
