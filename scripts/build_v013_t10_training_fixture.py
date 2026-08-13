from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a qualified V0.1.3 T10 training fixture."
    )
    parser.add_argument("--project-id", type=int, default=9010)
    parser.add_argument("--target", default="冲击强度")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.samples < 30:
        raise SystemExit("ERROR: T10 fixture requires at least 30 samples.")

    rng = random.Random(args.seed)

    output_dir = Path(".runtime") / "v013" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "t10_training_dataset.csv"
    reality_path = output_dir / "t10_training_reality.json"

    target_col = f"target::{args.target}"

    fieldnames = [
        "sample_id",
        "sample_name",
        "formula::ABS",
        "formula::PC",
        "formula::增韧剂",
        "process::加工温度",
        "process::螺杆转速",
        "condition_signature",
        target_col,
    ]

    rows = []

    for i in range(args.samples):
        abs_pct = rng.uniform(18.0, 38.0)
        toughener_pct = rng.uniform(5.0, 16.0)

        # Keep a simple closed formulation for the fixture.
        pc_pct = 100.0 - abs_pct - toughener_pct

        temperature = rng.uniform(220.0, 260.0)
        speed = rng.uniform(180.0, 320.0)

        # Deterministic synthetic relationship + bounded noise.
        # This is only a test fixture; metrics must still come from real sklearn execution.
        impact = (
            5.0
            + 0.70 * abs_pct
            + 1.25 * toughener_pct
            - 0.08 * (temperature - 240.0)
            + 0.015 * (speed - 250.0)
            + rng.gauss(0.0, 2.2)
        )

        rows.append(
            {
                "sample_id": f"T10-{i + 1:03d}",
                "sample_name": f"T10_sample_{i + 1:03d}",
                "formula::ABS": round(abs_pct, 6),
                "formula::PC": round(pc_pct, 6),
                "formula::增韧剂": round(toughener_pct, 6),
                "process::加工温度": round(temperature, 6),
                "process::螺杆转速": round(speed, 6),
                "condition_signature": "ISO_179_23C_notched",
                target_col: round(impact, 6),
            }
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    values = [float(row[target_col]) for row in rows]

    reality = {
        "stage": "V0.1.3-A_dataset_reality_check",
        "project_id": args.project_id,
        "company_id": "V013-T10-TEST",
        "target_metric": args.target,
        "summary": {
            "total_samples": args.samples,
            "formula_present": args.samples,
            "process_present": args.samples,
            "target_present": args.samples,
            "conditions_present": args.samples,
            "core_closed_formula_process_target": args.samples,
            "strict_closed_with_conditions": args.samples,
        },
        "target": {
            "metric": args.target,
            "resolved_field_ids": [15774],
            "present_count": args.samples,
            "numeric_count": args.samples,
            "non_numeric_count": 0,
            "conflicting_sample_ids": [],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        },
        "test_conditions": {
            "present_count": args.samples,
            "missing_count": 0,
            "unique_nonempty_signatures": 1,
            "signature_counts": {
                "ISO_179_23C_notched": args.samples,
            },
        },
        "field_coverage": {
            "formula": {
                "formula::ABS": args.samples,
                "formula::PC": args.samples,
                "formula::增韧剂": args.samples,
            },
            "process": {
                "process::加工温度": args.samples,
                "process::螺杆转速": args.samples,
            },
        },
        "unresolved_dynamic_fields": {},
        "duplicate_sample_name_groups": 0,
        "duplicate_formula_process_target_groups": 0,
        "fixture": {
            "name": "V0.1.3-T10-qualified-training-fixture",
            "seed": args.seed,
            "note": "Synthetic fixture for pipeline verification only; not scientific evidence.",
        },
    }

    with reality_path.open("w", encoding="utf-8") as f:
        json.dump(reality, f, ensure_ascii=False, indent=2)

    print("V0.1.3 T10 TRAINING FIXTURE")
    print(f"project_id: {args.project_id}")
    print(f"target_metric: {args.target}")
    print(f"samples: {args.samples}")
    print("core_closed_samples:", args.samples)
    print("strict_closed_samples:", args.samples)
    print("condition_signatures: 1")
    print("unresolved_dynamic_field_instances: 0")
    print()
    print("OUTPUT")
    print(f"reality_json: {reality_path}")
    print(f"dataset_csv: {csv_path}")
    print()
    print("V0.1.3 T10 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
