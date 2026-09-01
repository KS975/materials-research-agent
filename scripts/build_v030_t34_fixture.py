from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.build_v030_t33_fixture import response
from scripts.build_v030_t33_fixture import protocol_template as t33_protocol_template
from scripts.build_v030_t33_fixture import device_profile as t33_device_profile
from scripts.build_v030_t33_fixture import safety_policy as t33_safety_policy


CAMPAIGN_ID = "V030_T34_DEMO"
PROJECT_ID = 9034
TARGET = "冲击强度"
UNIT = "kJ/m²"
CONDITION = "T33_STANDARD_23C"
DATASET_VERSIONS = [
    "dataset_v001",
    "dataset_v002",
    "dataset_v003",
    "dataset_v004",
]
CHALLENGER_MODEL_VERSIONS = [
    "model_v002",
    "model_v003",
    "model_v004",
]


def protocol_template() -> dict:
    template = deepcopy(t33_protocol_template())
    template["template_id"] = "V030_T34_POLYMER_MULTI_ROUND_V1"
    template["name"] = "T34 多轮自主实验协议"
    template["project_id"] = PROJECT_ID
    template["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "multi_round": True,
    }
    return template


def device_profile() -> dict:
    profile = deepcopy(t33_device_profile())
    profile["device_id"] = "SIM_T34_MAIN"
    profile["name"] = "T34 Multi-Round Autonomous Simulator"
    profile["supported_template_ids"] = [protocol_template()["template_id"]]
    profile["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "multi_round": True,
    }
    return profile


def safety_policy() -> dict:
    policy = deepcopy(t33_safety_policy())
    policy["policy_id"] = "V030_T34_SAFETY_V1"
    policy["device_id"] = device_profile()["device_id"]
    policy["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "multi_round": True,
    }
    return policy


def campaign_create() -> dict:
    return {
        "campaign_id": CAMPAIGN_ID,
        "project_id": PROJECT_ID,
        "name": "T34 三轮连续自主实验闭环",
        "target_metrics": [TARGET],
        "metadata": {
            "fixture": True,
            "purpose": "V0.3-T34 multi-round autonomous acceptance",
            "real_measurement": False,
            "target_rounds": 3,
        },
    }


def round1_features() -> list[tuple[float, float, float, float]]:
    return [
        (31.0, 16.0, 240.0, 270.0),
        (34.0, 17.0, 246.0, 300.0),
        (36.0, 18.0, 250.0, 320.0),
        (33.0, 19.0, 252.0, 330.0),
        (38.0, 17.5, 255.0, 310.0),
    ]


def round1_plan() -> dict:
    return {
        "planned_experiment_count": 5,
        "dataset_version": DATASET_VERSIONS[0],
        "model_versions": {TARGET: "model_v001"},
        "search_space_snapshot": {
            "version": "T34_SPACE_V1",
            "variables": [
                "formula::ABS",
                "formula::PC",
                "formula::增韧剂",
                "process::加工温度",
                "process::螺杆转速",
            ],
        },
        "constraints_snapshot": {
            "version": "T34_CONSTRAINTS_V1",
            "hard": ["formula sum = 100"],
        },
        "optimizer_config": {
            "engine": "GaussianProcess",
            "acquisition": "EI",
            "batch_size": 5,
        },
        "source": "V0.3-T34_fixture",
    }


def planned_experiments() -> list[dict]:
    rows = []
    for i, (abs_v, tough, temp, speed) in enumerate(round1_features(), start=1):
        pc = 100.0 - abs_v - tough
        prediction = float(response(abs_v, tough, temp, speed))
        rows.append({
            "candidate_id": f"V030_T34_R1_{i:02d}",
            "required_metrics": [TARGET],
            "expected_test_condition_signature": CONDITION,
            "units": {TARGET: UNIT},
            "features": {
                "formula::ABS": abs_v,
                "formula::PC": pc,
                "formula::增韧剂": tough,
                "process::加工温度": temp,
                "process::螺杆转速": speed,
            },
            "prediction_snapshot": {
                TARGET: {
                    "value": prediction,
                    "posterior_std": 1.5,
                    "source": "T34_FIXTURE_PRIOR_MODEL",
                }
            },
        })
    return rows


def gate_pass() -> dict:
    return {
        "stage": "V0.1.3-B_modeling_gate",
        "project_id": PROJECT_ID,
        "target_metric": TARGET,
        "decision": "PASS",
        "training_allowed": True,
        "official_model_allowed": True,
    }


def write_csvs(out: Path) -> None:
    rng = np.random.default_rng(34)
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
        TARGET,
    ]

    base_rows = []
    for i in range(35):
        abs_v = float(rng.uniform(24, 41))
        tough = float(rng.uniform(11, 20))
        pc = 100.0 - abs_v - tough
        temp = float(rng.uniform(225, 270))
        speed = float(rng.integers(190, 351))
        y = float(
            response(abs_v, tough, temp, speed)
            + rng.normal(0, 0.75)
        )
        base_rows.append({
            "candidate_id": f"T34_BASE_{i+1:03d}",
            "project_id": str(PROJECT_ID),
            "test_condition_signature": CONDITION,
            "source_campaign": "BASE_IMPORT",
            "source_round": "BASE",
            "formula::ABS": f"{abs_v:.10f}",
            "formula::PC": f"{pc:.10f}",
            "formula::增韧剂": f"{tough:.10f}",
            "process::加工温度": f"{temp:.10f}",
            "process::螺杆转速": f"{speed:.10f}",
            TARGET: f"{y:.10f}",
        })

    with (out / "dataset_v001.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(base_rows)

    pool_rows = []
    for i in range(1200):
        if i < 120:
            # Deliberately broad points exercise T24 AD exclusion.
            abs_v = float(rng.uniform(17, 49))
            tough = float(rng.uniform(6, 25))
            temp = float(rng.uniform(278, 304))
            speed = float(rng.integers(145, 401))
        else:
            abs_v = float(rng.uniform(25, 40))
            tough = float(rng.uniform(11.5, 19.2))
            temp = float(rng.uniform(228, 268))
            speed = float(rng.integers(195, 346))
        pc = 100.0 - abs_v - tough
        pool_rows.append({
            "candidate_id": f"V030_T34_POOL_{i+1:04d}",
            "hard_valid": (
                "false" if i in {101, 302, 503, 704, 905} else "true"
            ),
            "soft_penalty": "0.10" if tough > 18.8 else "0",
            "formula::ABS": f"{abs_v:.10f}",
            "formula::PC": f"{pc:.10f}",
            "formula::增韧剂": f"{tough:.10f}",
            "process::加工温度": f"{temp:.10f}",
            "process::螺杆转速": f"{speed:.10f}",
        })

    pool_columns = [
        "candidate_id",
        "hard_valid",
        "soft_penalty",
        "formula::ABS",
        "formula::PC",
        "formula::增韧剂",
        "process::加工温度",
        "process::螺杆转速",
    ]
    with (out / "candidate_pool.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=pool_columns)
        writer.writeheader()
        writer.writerows(pool_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".runtime/v030/fixtures/t34",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if args.reset and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    write_csvs(out)
    payloads = {
        "campaign_create.json": campaign_create(),
        "round1_plan.json": round1_plan(),
        "round1_planned_experiments.json": planned_experiments(),
        "protocol_template.json": protocol_template(),
        "device_profile.json": device_profile(),
        "safety_policy.json": safety_policy(),
        "gate_pass.json": gate_pass(),
    }
    for name, payload in payloads.items():
        (out / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("V0.3-T34 FIXTURE BUILDER")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"project_id: {PROJECT_ID}")
    print("base_dataset_rows: 35")
    print("target_rounds: 3")
    print("experiments_per_round: 5")
    print("target_total_experiments: 15")
    print("candidate_pool_rows: 1200")
    print("device_count: 1")
    print("device_adapter: SimulatorDeviceAdapter")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T34 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
