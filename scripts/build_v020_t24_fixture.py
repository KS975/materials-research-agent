from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def response(abs_v, tough, temp, speed):
    return (
        24.0
        + 0.28 * abs_v
        + 0.92 * tough
        - 0.010 * (temp - 248.0) ** 2
        + 0.010 * speed
        + 2.5 * np.sin(abs_v / 7.0)
        + 1.2 * np.cos(tough / 4.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".runtime/v020/fixtures/t24",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(24)
    project_id = 9024
    campaign_id = "V020_T24_DEMO"
    columns = [
        "candidate_id",
        "project_id",
        "test_condition_signature",
        "source_campaign",
        "source_round",
        "formula::ABS",
        "formula::PC",
        "formula::增韧剂",
        "process::加工温度",
        "process::螺杆转速",
        "冲击强度",
    ]

    base_rows = []
    for i in range(35):
        abs_v = float(rng.uniform(22, 42))
        tough = float(rng.uniform(10, 20))
        pc = 100.0 - abs_v - tough
        temp = float(rng.uniform(225, 275))
        speed = float(rng.uniform(190, 350))
        y = float(response(abs_v, tough, temp, speed) + rng.normal(0, 0.7))
        base_rows.append({
            "candidate_id": f"BASE_{i+1:03d}",
            "project_id": str(project_id),
            "test_condition_signature": "T24_STANDARD_23C",
            "source_campaign": "BASE_IMPORT",
            "source_round": "BASE",
            "formula::ABS": f"{abs_v:.10f}",
            "formula::PC": f"{pc:.10f}",
            "formula::增韧剂": f"{tough:.10f}",
            "process::加工温度": f"{temp:.10f}",
            "process::螺杆转速": f"{speed:.10f}",
            "冲击强度": f"{y:.10f}",
        })

    base_best = max(float(r["冲击强度"]) for r in base_rows)

    base_csv = out / "dataset_v001.csv"
    with base_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(base_rows)

    round1_features = [
        (33.0, 18.0, 247.0, 310.0),
        (35.0, 19.0, 250.0, 325.0),
        (30.0, 17.0, 245.0, 295.0),
        (38.0, 18.5, 252.0, 315.0),
        (34.0, 19.5, 249.0, 335.0),
    ]
    measured = []
    for i, (abs_v, tough, temp, speed) in enumerate(round1_features):
        value = float(response(abs_v, tough, temp, speed))
        if i == 4:
            value = max(value, base_best + 3.0)
        measured.append(value)

    campaign = {
        "campaign_id": campaign_id,
        "project_id": project_id,
        "name": "冲击强度多轮闭环 BO 演示",
        "target_metrics": ["冲击强度"],
        "metadata": {"fixture": True, "purpose": "V0.2-T24 acceptance"},
    }
    round1_plan = {
        "planned_experiment_count": 5,
        "dataset_version": "dataset_v001",
        "model_versions": {"冲击强度": "model_v001"},
        "search_space_snapshot": {
            "version": "search_space_v001",
            "variables": [
                "formula::ABS", "formula::PC", "formula::增韧剂",
                "process::加工温度", "process::螺杆转速",
            ],
        },
        "constraints_snapshot": {
            "version": "constraints_v001",
            "hard": ["formula sum = 100"],
        },
        "optimizer_config": {
            "engine": "GaussianProcess",
            "acquisition": "EI",
            "batch_size": 5,
        },
        "source": "V0.1.4-T18",
    }

    planned = []
    results = []
    for i, ((abs_v, tough, temp, speed), actual) in enumerate(
        zip(round1_features, measured), start=1
    ):
        pc = 100.0 - abs_v - tough
        cid = f"V020_T24_R1_{i:02d}"
        features = {
            "formula::ABS": abs_v,
            "formula::PC": pc,
            "formula::增韧剂": tough,
            "process::加工温度": temp,
            "process::螺杆转速": speed,
        }
        planned.append({
            "candidate_id": cid,
            "required_metrics": ["冲击强度"],
            "expected_test_condition_signature": "T24_STANDARD_23C",
            "units": {"冲击强度": "kJ/m²"},
            "features": features,
            "prediction_snapshot": {
                "冲击强度": {
                    "value": actual - 1.0,
                    "posterior_std": 1.2,
                    "source": "previous_round_GP",
                }
            },
        })
        results.append({
            "candidate_id": cid,
            "status": "COMPLETED",
            "test_condition_signature": "T24_STANDARD_23C",
            "measurements": {"冲击强度": actual},
            "units": {"冲击强度": "kJ/m²"},
        })

    # Candidate pool: explicit observed duplicates + fresh in/near/out-domain points.
    pool_rows = []
    # Five exact observed feature duplicates; IDs are deliberately new.
    duplicate_source = base_rows[:3]
    for i, row in enumerate(duplicate_source, start=1):
        pool_rows.append({
            "candidate_id": f"POOL_OBS_DUP_{i:02d}",
            "hard_valid": "true",
            "soft_penalty": "0",
            **{c: row[c] for c in columns if c.startswith("formula::") or c.startswith("process::")},
        })
    for i, features in enumerate(round1_features[:2], start=4):
        abs_v, tough, temp, speed = features
        pool_rows.append({
            "candidate_id": f"POOL_OBS_DUP_{i:02d}",
            "hard_valid": "true",
            "soft_penalty": "0",
            "formula::ABS": f"{abs_v:.10f}",
            "formula::PC": f"{100-abs_v-tough:.10f}",
            "formula::增韧剂": f"{tough:.10f}",
            "process::加工温度": f"{temp:.10f}",
            "process::螺杆转速": f"{speed:.10f}",
        })

    for i in range(595):
        if i < 70:
            # Search-space-valid but intentionally outside observed AD.
            abs_v = float(rng.uniform(15, 50))
            tough = float(rng.uniform(6, 25))
            temp = float(rng.uniform(285, 315))
            speed = float(rng.uniform(150, 390))
        else:
            abs_v = float(rng.uniform(23, 41))
            tough = float(rng.uniform(10.5, 19.5))
            temp = float(rng.uniform(228, 272))
            speed = float(rng.uniform(195, 345))
        pc = 100.0 - abs_v - tough
        pool_rows.append({
            "candidate_id": f"V020_T24_POOL_{i+1:04d}",
            "hard_valid": "false" if i in {100, 200, 300} else "true",
            "soft_penalty": "0.12" if tough > 19.0 else "0",
            "formula::ABS": f"{abs_v:.10f}",
            "formula::PC": f"{pc:.10f}",
            "formula::增韧剂": f"{tough:.10f}",
            "process::加工温度": f"{temp:.10f}",
            "process::螺杆转速": f"{speed:.10f}",
        })

    pool_columns = [
        "candidate_id", "hard_valid", "soft_penalty",
        "formula::ABS", "formula::PC", "formula::增韧剂",
        "process::加工温度", "process::螺杆转速",
    ]
    pool_csv = out / "candidate_pool.csv"
    with pool_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pool_columns)
        writer.writeheader()
        writer.writerows(pool_rows)

    gate = {
        "decision": "PASS",
        "training_allowed": True,
        "official_model_allowed": True,
    }

    write_json(out / "campaign_create.json", campaign)
    write_json(out / "round1_plan.json", round1_plan)
    write_json(out / "round1_planned_experiments.json", planned)
    write_json(out / "round1_results.json", results)
    write_json(out / "gate_pass.json", gate)

    print("V0.2-T24 FIXTURE BUILDER")
    print(f"base_dataset_csv: {base_csv}")
    print("base_rows: 35")
    print("round1_completed_results: 5")
    print("expected_dataset_v002_rows: 40")
    print(f"candidate_pool_rows: {len(pool_rows)}")
    print("explicit_observed_feature_duplicates: 5")
    print(f"base_best_冲击强度: {base_best:.6f}")
    print(f"round1_best_冲击强度: {max(measured):.6f}")
    print()
    print("V0.2-T24 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
