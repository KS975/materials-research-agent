from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from optimization import (
    InverseDesignError,
    load_search_space,
    parse_inverse_design_request,
    parse_inverse_design_text,
    run_inverse_design,
)


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


def safe_name(value: str) -> str:
    cleaned = re.sub(
        r"[^\w\-\u4e00-\u9fff]+",
        "_",
        str(value),
    ).strip("_")
    return cleaned or "inverse_design"


def write_design_csv(
    path: Path,
    cards: list[dict[str, Any]],
    *,
    objective_metrics: list[str],
    feature_columns: list[str],
) -> None:
    fields = (
        [
            "recommendation_rank",
            "candidate_id",
            "pareto_rank",
            "soft_penalty",
            "ad_status",
            "ad_risk",
        ]
        + [f"predicted::{metric}" for metric in objective_metrics]
        + [f"margin::{metric}" for metric in objective_metrics]
        + feature_columns
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for card in cards:
            row = {
                "recommendation_rank": card["recommendation_rank"],
                "candidate_id": card["candidate_id"],
                "pareto_rank": card.get("pareto_rank"),
                "soft_penalty": card["soft_penalty"],
                "ad_status": card["applicability_domain"]["status"],
                "ad_risk": card["applicability_domain"]["risk"],
            }

            for metric in objective_metrics:
                row[f"predicted::{metric}"] = (
                    card["predictions"][metric]
                )
                row[f"margin::{metric}"] = (
                    card["target_margins"][metric]
                )

            for feature in feature_columns:
                row[feature] = card["features"].get(feature)

            writer.writerow(row)


def write_near_miss_csv(
    path: Path,
    cards: list[dict[str, Any]],
    *,
    objective_metrics: list[str],
    feature_columns: list[str],
) -> None:
    fields = (
        [
            "candidate_id",
            "total_normalized_threshold_shortfall",
            "soft_penalty",
            "ad_status",
            "ad_risk",
        ]
        + [f"predicted::{metric}" for metric in objective_metrics]
        + [f"margin::{metric}" for metric in objective_metrics]
        + feature_columns
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for card in cards:
            row = {
                "candidate_id": card["candidate_id"],
                "total_normalized_threshold_shortfall": (
                    card["total_normalized_threshold_shortfall"]
                ),
                "soft_penalty": card["soft_penalty"],
                "ad_status": card["applicability_domain"]["status"],
                "ad_risk": card["applicability_domain"]["risk"],
            }

            for metric in objective_metrics:
                row[f"predicted::{metric}"] = (
                    card["predictions"][metric]
                )
                row[f"margin::{metric}"] = (
                    card["target_margins"][metric]
                )

            for feature in feature_columns:
                row[feature] = card["features"].get(feature)

            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V0.1.4-T17 formal inverse-design acceptance."
    )

    request_group = parser.add_mutually_exclusive_group(
        required=True
    )
    request_group.add_argument("--request-json")
    request_group.add_argument("--request-text")

    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Required when --request-text is used.",
    )
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--search-space-json", required=True)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="Optional override.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional override.",
    )
    parser.add_argument(
        "--runtime-root",
        default=".runtime",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()

    try:
        if args.request_json:
            request_doc = load_json(
                Path(args.request_json)
            )
            request = parse_inverse_design_request(
                request_doc,
                source="json",
            )
        else:
            if args.project_id is None:
                raise InverseDesignError(
                    "--request-text 模式必须提供 --project-id"
                )
            request = parse_inverse_design_text(
                args.request_text,
                project_id=args.project_id,
                candidate_count=(
                    args.candidate_count
                    if args.candidate_count is not None
                    else 600
                ),
                random_state=(
                    args.random_state
                    if args.random_state is not None
                    else 42
                ),
            )

        search_space = load_search_space(
            load_json(
                Path(args.search_space_json)
            )
        )

        report = run_inverse_design(
            request=request,
            search_space=search_space,
            dataset_csv=Path(args.dataset_csv),
            runtime_root=Path(args.runtime_root),
            candidate_count_override=args.candidate_count,
            random_state_override=args.random_state,
        )

    except InverseDesignError as exc:
        raise SystemExit(
            f"V0.1.4-T17 INVERSE DESIGN BLOCKED\n- {exc}"
        ) from exc

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = (
            Path(args.runtime_root)
            / "v014"
            / "inverse_design"
            / f"project_{request.project_id}"
            / safe_name(request.request_name)
        )

    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "inverse_design_report.json"
    design_csv = out / "recommended_designs.csv"
    near_miss_csv = out / "near_miss_candidates.csv"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    objective_metrics = [
        item["metric"]
        for item in report["request"]["objectives"]
    ]
    feature_columns = report["evidence"]["feature_columns"]

    write_design_csv(
        design_csv,
        report["design_cards"],
        objective_metrics=objective_metrics,
        feature_columns=feature_columns,
    )
    write_near_miss_csv(
        near_miss_csv,
        report["near_miss_candidates"],
        objective_metrics=objective_metrics,
        feature_columns=feature_columns,
    )

    print("V0.1.4-T17 INVERSE DESIGN")
    print(f"project_id: {report['project_id']}")
    print(f"request_source: {report['request']['source']}")
    if report["request"]["raw_request_text"]:
        print(
            "request_text: "
            f"{report['request']['raw_request_text']}"
        )
    print()

    print("TARGETS")
    for objective in report["request"]["objectives"]:
        print(
            f"- {objective['metric']} "
            f"{objective['threshold_operator']} "
            f"{objective['threshold_value']} "
            f"({objective['direction']})"
        )
    print()

    print("SAFETY")
    print(
        "all_objectives_gate_training_allowed: "
        f"{str(report['safety']['all_objectives_gate_training_allowed']).lower()}"
    )
    print(
        "all_objectives_official_model_allowed: "
        f"{str(report['safety']['all_objectives_official_model_allowed']).lower()}"
    )
    print("formal_design_requires_in_domain: true")
    print("fabricate_missing_recommendations: false")
    print()

    print("CANDIDATE FLOW")
    counts = report["counts"]
    print(
        f"generated_hard_valid: {counts['generated_hard_valid']}"
    )
    print(
        f"trusted_in_domain: {counts['trusted_in_domain']}"
    )
    print(
        f"qualified_all_targets: {counts['qualified_all_targets']}"
    )
    print(
        f"pareto_front: {counts['pareto_front']}"
    )
    print(
        f"recommended: {counts['recommended']}"
    )
    print()

    print("DECISION")
    print(report["status"])
    print(report["answer"])
    print()

    if report["design_cards"]:
        print("RECOMMENDED DESIGNS")
        for card in report["design_cards"]:
            prediction_text = " | ".join(
                f"{metric}={card['predictions'][metric]:.6f}"
                for metric in objective_metrics
            )
            margin_text = " | ".join(
                f"margin_{metric}={card['target_margins'][metric]:+.6f}"
                for metric in objective_metrics
            )
            print(
                f"{card['recommendation_rank']}. "
                f"{card['candidate_id']} | "
                f"{prediction_text} | "
                f"{margin_text} | "
                f"pareto_rank={card['pareto_rank']} | "
                f"soft_penalty={card['soft_penalty']:.6f}"
            )
        print()

    if report["status"] == "NO_FEASIBLE_DESIGN":
        print("NEAR MISSES (DIAGNOSTIC ONLY)")
        for card in report["near_miss_candidates"][:5]:
            prediction_text = " | ".join(
                f"{metric}={card['predictions'][metric]:.6f}"
                for metric in objective_metrics
            )
            print(
                f"- {card['candidate_id']} | "
                f"{prediction_text} | "
                f"shortfall={card['total_normalized_threshold_shortfall']:.6f}"
            )
        print()

    print("OUTPUT")
    print(f"report_json: {report_path}")
    print(f"recommended_designs_csv: {design_csv}")
    print(f"near_miss_candidates_csv: {near_miss_csv}")
    print()

    if (
        report["scientific_status"]["fixture_or_real_data"]
        == "fixture"
    ):
        print(
            "NOTE: T17 used synthetic fixture data. "
            "The designs validate the engineering pipeline only; "
            "they are not materials-science recommendations."
        )
        print()

    print("V0.1.4-T17 INVERSE DESIGN PASS")


if __name__ == "__main__":
    main()
