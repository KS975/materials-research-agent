from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


FEATURES = [
    "formula::ABS",
    "formula::PC",
    "formula::增韧剂",
    "process::加工温度",
    "process::螺杆转速",
]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def synthetic_impact(
    abs_pct: float,
    toughener: float,
    temperature: float,
    speed: float,
    *,
    noise: float,
) -> float:
    # Nonlinear synthetic objective with a broad high-performance region.
    # Engineering fixture only; not a material-science law.
    local_peak = 11.0 * math.exp(
        -(
            ((abs_pct - 33.0) / 5.0) ** 2
            + ((toughener - 13.5) / 3.0) ** 2
            + ((temperature - 246.0) / 9.0) ** 2
            + ((speed - 280.0) / 50.0) ** 2
        )
    )

    value = (
        18.0
        + 0.48 * abs_pct
        + 0.72 * toughener
        + local_peak
        - 0.010 * (temperature - 246.0) ** 2
        + 0.010 * (speed - 250.0)
        + noise
    )

    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V0.1.4-T18 Bayesian Optimization fixture."
    )
    parser.add_argument(
        "--output-dir",
        default=".runtime/v014/fixtures/t18",
    )
    parser.add_argument(
        "--observed-rows",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if args.observed_rows < 30:
        raise SystemExit(
            "ERROR: T18 fixture requires at least 30 observed rows "
            "so the Modeling Gate can PASS."
        )

    rng = random.Random(args.random_state)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    project_id = 9018
    target_metric = "冲击强度"

    rows = []
    seen = set()

    while len(rows) < args.observed_rows:
        abs_pct = rng.randrange(42, 77) / 2.0       # 21.0 ... 38.0
        toughener = rng.randrange(12, 35) / 2.0    # 6.0 ... 17.0
        pc_pct = 100.0 - abs_pct - toughener

        if not 45.0 <= pc_pct <= 73.0:
            continue

        temperature = float(rng.randrange(112, 129) * 2)  # 224 ... 256
        speed = float(rng.randrange(10, 17) * 20)         # 200 ... 320

        key = (
            abs_pct,
            pc_pct,
            toughener,
            temperature,
            speed,
        )
        if key in seen:
            continue
        seen.add(key)

        impact = synthetic_impact(
            abs_pct,
            toughener,
            temperature,
            speed,
            noise=rng.gauss(0.0, 0.75),
        )

        rows.append(
            {
                "sample_name": f"T18_OBS_{len(rows)+1:03d}",
                "formula::ABS": abs_pct,
                "formula::PC": pc_pct,
                "formula::增韧剂": toughener,
                "process::加工温度": temperature,
                "process::螺杆转速": speed,
                "condition_signature": "T18_STANDARD_23C",
                "target::冲击强度": round(impact, 6),
            }
        )

    observations_path = out / "initial_observations.csv"

    with observations_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_name",
                *FEATURES,
                "condition_signature",
                "target::冲击强度",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    reality = {
        "stage": "V0.1.3-A_dataset_reality_check",
        "project_id": project_id,
        "company_id": "fixture-company",
        "target_metric": target_metric,
        "summary": {
            "total_samples": len(rows),
            "formula_present": len(rows),
            "process_present": len(rows),
            "target_present": len(rows),
            "conditions_present": len(rows),
            "core_closed_formula_process_target": len(rows),
            "strict_closed_with_conditions": len(rows),
        },
        "target": {
            "metric": target_metric,
            "resolved_field_ids": [],
            "present_count": len(rows),
            "numeric_count": len(rows),
            "non_numeric_count": 0,
            "conflicting_sample_ids": [],
        },
        "test_conditions": {
            "present_count": len(rows),
            "missing_count": 0,
            "unique_nonempty_signatures": 1,
            "signature_counts": {
                "T18_STANDARD_23C": len(rows)
            },
        },
        "duplicate_sample_name_groups": [],
        "duplicate_formula_process_target_groups": [],
        "unresolved_dynamic_fields": {},
        "warnings": [
            "Synthetic Bayesian Optimization fixture only."
        ],
    }

    search_space = {
        "stage": "V0.1.4-T14_search_space",
        "project_id": project_id,
        "name": "t18_bo_design_space",
        "metadata": {
            "purpose": "T18 Bayesian Optimization engineering fixture",
            "target_metric": target_metric,
        },
        "variables": [
            {
                "name": "formula::ABS",
                "kind": "continuous",
                "min": 20.0,
                "max": 40.0,
                "step": 0.5,
                "unit": "%",
            },
            {
                "name": "formula::PC",
                "kind": "continuous",
                "min": 45.0,
                "max": 75.0,
                "step": 0.5,
                "unit": "%",
            },
            {
                "name": "formula::增韧剂",
                "kind": "continuous",
                "min": 5.0,
                "max": 20.0,
                "step": 0.5,
                "unit": "%",
            },
            {
                "name": "process::加工温度",
                "kind": "integer",
                "min": 220,
                "max": 260,
                "step": 2,
                "unit": "℃",
            },
            {
                "name": "process::螺杆转速",
                "kind": "integer",
                "min": 180,
                "max": 320,
                "step": 20,
                "unit": "rpm",
            },
        ],
        "constraints": [
            {
                "id": "formula_sum_100",
                "type": "weighted_sum",
                "severity": "HARD",
                "terms": [
                    {"variable": "formula::ABS"},
                    {"variable": "formula::PC"},
                    {"variable": "formula::增韧剂"},
                ],
                "operator": "==",
                "value": 100.0,
                "tolerance": 0.25,
                "message": "主配方总和必须等于约 100%",
            },
            {
                "id": "toughener_preferred_max",
                "type": "scalar",
                "severity": "SOFT",
                "variable": "formula::增韧剂",
                "operator": "<=",
                "value": 15.0,
                "weight": 1.5,
                "message": "增韧剂超过 15% 时增加下一轮实验成本偏好惩罚",
            },
        ],
    }

    request = {
        "stage": "V0.1.4-T18_bayesian_optimization_request",
        "project_id": project_id,
        "request_name": "next_5_impact_experiments",
        "target_metric": target_metric,
        "direction": "maximize",
        "batch_size": 5,
        "candidate_count": 900,
        "max_attempts": 100000,
        "random_state": 42,
        "acquisition": "EI",
        "xi": 0.01,
        "kappa": 2.0,
        "min_batch_distance": 0.20,
        "allow_borderline_for_exploration": True,
        "soft_penalty_weight": 0.10,
    }

    reality_path = out / "reality_冲击强度.json"
    search_space_path = out / "search_space.json"
    request_path = out / "bo_request.json"

    write_json(reality_path, reality)
    write_json(search_space_path, search_space)
    write_json(request_path, request)

    best_observed = max(
        row["target::冲击强度"]
        for row in rows
    )

    print("V0.1.4-T18 FIXTURE BUILDER")
    print(f"project_id: {project_id}")
    print(f"observed_rows: {len(rows)}")
    print(f"best_observed_冲击强度: {best_observed:.6f}")
    print(f"observations_csv: {observations_path}")
    print(f"reality_json: {reality_path}")
    print(f"search_space_json: {search_space_path}")
    print(f"bo_request_json: {request_path}")
    print()
    print(
        "NOTE: 这是带非线性峰值和噪声的 synthetic BO fixture，"
        "仅用于验证下一轮实验推荐链路。"
    )
    print()
    print("V0.1.4-T18 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
