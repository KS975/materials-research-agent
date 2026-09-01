from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reality_report(
    *,
    project_id: int,
    target_metric: str,
    numeric_count: int,
) -> dict:
    return {
        "stage": "V0.1.3-A_dataset_reality_check",
        "project_id": project_id,
        "company_id": "fixture-company",
        "target_metric": target_metric,
        "summary": {
            "total_samples": numeric_count,
            "formula_present": numeric_count,
            "process_present": numeric_count,
            "target_present": numeric_count,
            "conditions_present": numeric_count,
            "core_closed_formula_process_target": numeric_count,
            "strict_closed_with_conditions": numeric_count,
        },
        "target": {
            "metric": target_metric,
            "resolved_field_ids": [],
            "present_count": numeric_count,
            "numeric_count": numeric_count,
            "non_numeric_count": 0,
            "conflicting_sample_ids": [],
        },
        "test_conditions": {
            "present_count": numeric_count,
            "missing_count": 0,
            "unique_nonempty_signatures": 1,
            "signature_counts": {
                "T16_STANDARD_23C": numeric_count
            },
        },
        "duplicate_sample_name_groups": [],
        "duplicate_formula_process_target_groups": [],
        "unresolved_dynamic_fields": {},
        "warnings": [
            "Synthetic fixture only; not a materials-science dataset."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V0.1.4-T16 dual-objective synthetic fixture."
    )
    parser.add_argument(
        "--output-dir",
        default=".runtime/v014/fixtures/t16",
    )
    parser.add_argument("--rows", type=int, default=140)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.rows < 60:
        raise SystemExit("ERROR: --rows must be >= 60")

    rng = random.Random(args.random_state)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_path = out / "multiobjective_dataset.csv"

    fieldnames = [
        "sample_name",
        "formula::ABS",
        "formula::PC",
        "formula::增韧剂",
        "process::加工温度",
        "process::螺杆转速",
        "condition_signature",
        "target::冲击强度",
        "target::MFR",
    ]

    rows = []

    for i in range(args.rows):
        abs_pct = rng.uniform(21.0, 38.0)
        toughener = rng.uniform(6.0, 17.0)
        pc_pct = 100.0 - abs_pct - toughener

        temperature = rng.uniform(224.0, 256.0)
        speed = rng.uniform(190.0, 310.0)

        # Synthetic trade-off:
        # more ABS/toughener tends to increase impact but reduce MFR.
        impact = (
            8.0
            + 0.78 * abs_pct
            + 1.42 * toughener
            - 0.055 * (temperature - 240.0)
            + 0.010 * (speed - 250.0)
            + rng.gauss(0.0, 1.6)
        )

        mfr = (
            35.0
            - 0.43 * abs_pct
            - 0.82 * toughener
            + 0.095 * (temperature - 240.0)
            + 0.018 * (speed - 250.0)
            + rng.gauss(0.0, 0.9)
        )

        rows.append(
            {
                "sample_name": f"T16_{i + 1:04d}",
                "formula::ABS": round(abs_pct, 6),
                "formula::PC": round(pc_pct, 6),
                "formula::增韧剂": round(toughener, 6),
                "process::加工温度": round(temperature, 6),
                "process::螺杆转速": round(speed, 6),
                "condition_signature": "T16_STANDARD_23C",
                "target::冲击强度": round(impact, 6),
                "target::MFR": round(mfr, 6),
            }
        )

    with dataset_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    project_id = 9016

    impact_reality = out / "reality_冲击强度.json"
    mfr_reality = out / "reality_MFR.json"

    write_json(
        impact_reality,
        reality_report(
            project_id=project_id,
            target_metric="冲击强度",
            numeric_count=args.rows,
        ),
    )
    write_json(
        mfr_reality,
        reality_report(
            project_id=project_id,
            target_metric="MFR",
            numeric_count=args.rows,
        ),
    )

    search_space = {
        "stage": "V0.1.4-T14_search_space",
        "project_id": project_id,
        "name": "t16_multiobjective_design_space",
        "metadata": {
            "purpose": "T16 双目标工程验收 fixture",
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
                "max": 74.0,
                "step": 0.5,
                "unit": "%",
            },
            {
                "name": "formula::增韧剂",
                "kind": "continuous",
                "min": 5.0,
                "max": 19.0,
                "step": 0.5,
                "unit": "%",
            },
            {
                "name": "process::加工温度",
                "kind": "continuous",
                "min": 220.0,
                "max": 260.0,
                "step": 1.0,
                "unit": "℃",
            },
            {
                "name": "process::螺杆转速",
                "kind": "integer",
                "min": 180,
                "max": 320,
                "step": 10,
                "unit": "rpm",
            },
            {
                "name": "process::催化剂",
                "kind": "categorical",
                "choices": ["NONE", "A", "B"],
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
                "tolerance": 0.5,
                "message": "主配方总和必须约等于 100%",
            },
            {
                "id": "toughener_recommended_max",
                "type": "scalar",
                "severity": "SOFT",
                "variable": "formula::增韧剂",
                "operator": "<=",
                "value": 15.0,
                "weight": 2.0,
                "message": "增韧剂超过 15% 时增加工程软惩罚",
            },
            {
                "id": "forbid_hot_catalyst_b",
                "type": "forbidden_combination",
                "severity": "HARD",
                "clauses": [
                    {
                        "variable": "process::加工温度",
                        "operator": ">",
                        "value": 252.0,
                    },
                    {
                        "variable": "process::催化剂",
                        "operator": "==",
                        "value": "B",
                    },
                ],
                "message": "高温区禁止催化剂 B",
            },
        ],
    }

    objectives = {
        "stage": "V0.1.4-T16_multiobjective_spec",
        "project_id": project_id,
        "name": "impact_and_mfr",
        "objectives": [
            {
                "metric": "冲击强度",
                "direction": "maximize",
                "weight": 1.0,
                "threshold": {
                    "operator": ">=",
                    "value": 43.0,
                },
            },
            {
                "metric": "MFR",
                "direction": "maximize",
                "weight": 1.0,
                "threshold": {
                    "operator": ">=",
                    "value": 8.5,
                },
            },
        ],
        "recommendation_count": 5,
        "soft_penalty_weight": 0.20,
        "diversity_weight": 0.35,
    }

    search_space_path = out / "search_space.json"
    objective_path = out / "objectives.json"

    write_json(search_space_path, search_space)
    write_json(objective_path, objectives)

    print("V0.1.4-T16 FIXTURE BUILDER")
    print(f"project_id: {project_id}")
    print(f"rows: {args.rows}")
    print(f"dataset_csv: {dataset_path}")
    print(f"impact_reality: {impact_reality}")
    print(f"mfr_reality: {mfr_reality}")
    print(f"search_space_json: {search_space_path}")
    print(f"objective_spec_json: {objective_path}")
    print()
    print(
        "NOTE: 冲击强度与 MFR 被故意设计为存在权衡关系，"
        "用于验证真实 Pareto Front。"
    )
    print()
    print("V0.1.4-T16 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
