from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.device import DEVICE_PROFILE_STAGE
from experiments.protocol import PROTOCOL_TEMPLATE_STAGE
from experiments.safety import SAFETY_POLICY_STAGE


CAMPAIGN_ID = "V030_T33_DEMO"
PROJECT_ID = 9033
TARGET = "冲击强度"
UNIT = "kJ/m²"
CONDITION = "T33_STANDARD_23C"
BASE_DATASET_VERSION = "dataset_v001"
CHILD_DATASET_VERSION = "dataset_v002"


def response(abs_v, tough, temp, speed):
    return (
        25.0
        + 0.30 * abs_v
        + 0.90 * tough
        - 0.008 * (temp - 248.0) ** 2
        + 0.012 * speed
        + 1.6 * np.sin(abs_v / 6.0)
    )


def protocol_template() -> dict:
    return {
        "stage": PROTOCOL_TEMPLATE_STAGE,
        "template_id": "V030_T33_POLYMER_AUTO_ROUND_V1",
        "name": "T33 自动轮次聚合物实验协议",
        "project_id": PROJECT_ID,
        "parameters": [
            {
                "name": "ABS",
                "source_feature": "formula::ABS",
                "section": "material_recipe",
                "kind": "continuous",
                "canonical_unit": "%",
                "default_input_unit": "%",
                "required": True,
                "safety": {"min": 0, "max": 100},
            },
            {
                "name": "PC",
                "source_feature": "formula::PC",
                "section": "material_recipe",
                "kind": "continuous",
                "canonical_unit": "%",
                "default_input_unit": "%",
                "required": True,
                "safety": {"min": 0, "max": 100},
            },
            {
                "name": "增韧剂",
                "source_feature": "formula::增韧剂",
                "section": "material_recipe",
                "kind": "continuous",
                "canonical_unit": "%",
                "default_input_unit": "%",
                "required": True,
                "safety": {"min": 0, "max": 30},
            },
            {
                "name": "加工温度",
                "source_feature": "process::加工温度",
                "section": "process_parameter",
                "kind": "continuous",
                "canonical_unit": "°C",
                "default_input_unit": "°C",
                "required": True,
                "safety": {"min": 180, "max": 280},
            },
            {
                "name": "螺杆转速",
                "source_feature": "process::螺杆转速",
                "section": "process_parameter",
                "kind": "integer",
                "canonical_unit": "rpm",
                "default_input_unit": "rpm",
                "required": True,
                "safety": {"min": 50, "max": 800},
            },
        ],
        "process_steps": [
            {
                "step_id": "weigh",
                "name": "称量",
                "device_role": "material_dispenser",
                "parameters": ["ABS", "PC", "增韧剂"],
                "instructions": "按 fixture 配方称量。",
            },
            {
                "step_id": "compound",
                "name": "混炼",
                "device_role": "compounder",
                "parameters": ["加工温度", "螺杆转速"],
                "instructions": "执行 deterministic simulator 混炼。",
            },
        ],
        "measurement_steps": [
            {
                "step_id": "impact",
                "name": "冲击强度测试",
                "device_role": "impact_tester",
                "metric": TARGET,
                "unit": UNIT,
                "condition_signature": CONDITION,
                "instructions": "T33 fixture 固定测试条件。",
            }
        ],
        "expected_outputs": [
            {"metric": TARGET, "unit": UNIT, "required": True}
        ],
        "metadata": {
            "fixture_only": True,
            "real_device": False,
        },
    }


def device_profile() -> dict:
    return {
        "stage": DEVICE_PROFILE_STAGE,
        "device_id": "SIM_T33_MAIN",
        "name": "T33 Autonomous Round Simulator",
        "adapter_type": "simulator",
        "online": True,
        "supported_roles": [
            "material_dispenser", "compounder", "impact_tester"
        ],
        "supported_template_ids": [protocol_template()["template_id"]],
        "capabilities": {
            "pause": True,
            "cancel": True,
            "progress_per_tick": 10,
        },
        "metadata": {
            "fixture_only": True,
            "real_device": False,
            "autonomous_round": True,
        },
    }


def safety_policy() -> dict:
    return {
        "stage": SAFETY_POLICY_STAGE,
        "policy_id": "V030_T33_SAFETY_V1",
        "device_id": device_profile()["device_id"],
        "protocol_limits": {
            "process::加工温度": {
                "min": 200, "max": 275, "unit": "°C"
            },
            "process::螺杆转速": {
                "min": 100, "max": 500, "unit": "rpm"
            },
        },
        "runtime_limits": {
            "temperature_c": {"min": 0, "max": 280},
            "rpm": {"min": 0, "max": 550},
        },
        "blocked_alarm_codes": [
            "DEVICE_ERROR", "SENSOR_FAULT", "COMMUNICATION_LOSS"
        ],
        "require_operator_ack": True,
        "metadata": {
            "fixture_only": True,
            "real_device": False,
        },
    }


def campaign_create() -> dict:
    return {
        "campaign_id": CAMPAIGN_ID,
        "project_id": PROJECT_ID,
        "name": "T33 单轮自主实验闭环",
        "target_metrics": [TARGET],
        "metadata": {
            "fixture": True,
            "purpose": "V0.3-T33 autonomous round acceptance",
            "real_measurement": False,
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
        "dataset_version": BASE_DATASET_VERSION,
        "model_versions": {TARGET: "model_v001"},
        "search_space_snapshot": {
            "version": "T33_SPACE_V1",
            "variables": [
                "formula::ABS",
                "formula::PC",
                "formula::增韧剂",
                "process::加工温度",
                "process::螺杆转速",
            ],
        },
        "constraints_snapshot": {
            "version": "T33_CONSTRAINTS_V1",
            "hard": ["formula sum = 100"],
        },
        "optimizer_config": {
            "engine": "GaussianProcess",
            "acquisition": "EI",
            "batch_size": 5,
        },
        "source": "V0.3-T33_fixture",
    }


def planned_experiments() -> list[dict]:
    result = []
    for i, (abs_v, tough, temp, speed) in enumerate(
        round1_features(), start=1
    ):
        pc = 100.0 - abs_v - tough
        prediction = float(response(abs_v, tough, temp, speed))
        result.append({
            "candidate_id": f"V030_T33_R1_{i:02d}",
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
                    "source": "T33_FIXTURE_PRIOR_MODEL",
                }
            },
        })
    return result


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
    rng = np.random.default_rng(33)
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
        speed = float(rng.uniform(190, 350))
        y = float(
            response(abs_v, tough, temp, speed)
            + rng.normal(0, 0.8)
        )
        base_rows.append({
            "candidate_id": f"T33_BASE_{i+1:03d}",
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

    base_csv = out / "dataset_v001.csv"
    with base_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(base_rows)

    pool_rows = []
    for i in range(900):
        if i < 90:
            abs_v = float(rng.uniform(18, 48))
            tough = float(rng.uniform(7, 24))
            temp = float(rng.uniform(278, 305))
            speed = float(rng.uniform(150, 390))
        else:
            abs_v = float(rng.uniform(25, 40))
            tough = float(rng.uniform(11.5, 19.5))
            temp = float(rng.uniform(228, 268))
            speed = float(rng.uniform(195, 345))
        pc = 100.0 - abs_v - tough
        pool_rows.append({
            "candidate_id": f"V030_T33_POOL_{i+1:04d}",
            "hard_valid": (
                "false" if i in {111, 333, 555} else "true"
            ),
            "soft_penalty": "0.10" if tough > 19.0 else "0",
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
    pool_csv = out / "candidate_pool.csv"
    with pool_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pool_columns)
        writer.writeheader()
        writer.writerows(pool_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".runtime/v030/fixtures/t33",
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

    print("V0.3-T33 FIXTURE BUILDER")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"project_id: {PROJECT_ID}")
    print("base_dataset_rows: 35")
    print("round1_planned_experiments: 5")
    print("candidate_pool_rows: 900")
    print("device_count: 1")
    print("device_adapter: SimulatorDeviceAdapter")
    print("result_capture: automatic")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T33 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
