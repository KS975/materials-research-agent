from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from experiments.protocol import ExperimentProtocolBuilder
from scripts.build_v030_t30_fixture import (
    telemetry_candidate,
    telemetry_template,
)
from scripts.build_v030_t31_fixture import device_profile


CAMPAIGN_ID = "V030_T32_DEMO"
PROJECT_ID = 9032
ROUND_CONDITION = "ISO180_23C_NOTCHED"
TARGET = "冲击强度"
UNIT = "kJ/m²"


def protocols() -> list[dict]:
    template = telemetry_template()
    docs = []
    for i in range(1, 5):
        candidate = deepcopy(telemetry_candidate())
        candidate["candidate_id"] = f"V030_T32_EXP{i:02d}"
        candidate["features"]["formula::ABS"] = 30.0 + i
        candidate["features"]["formula::PC"] = 57.0 - i
        candidate["features"]["process::加工温度"] = {
            "value": 224.0 + i,
            "unit": "°C",
        }
        docs.append(ExperimentProtocolBuilder(template).build(candidate))
    return docs


def campaign_create() -> dict:
    return {
        "campaign_id": CAMPAIGN_ID,
        "project_id": PROJECT_ID,
        "name": "T32 自动设备结果回流工程验收",
        "target_metrics": [TARGET],
        "metadata": {
            "fixture": True,
            "purpose": "V0.3-T32 automatic result capture acceptance",
            "real_measurement": False,
        },
    }


def round_plan() -> dict:
    return {
        "planned_experiment_count": 4,
        "dataset_version": "dataset_v001",
        "model_versions": {TARGET: "model_v001"},
        "search_space_snapshot": {"version": "V030_T32_SPACE"},
        "constraints_snapshot": {"version": "V030_T32_CONSTRAINTS"},
        "optimizer_config": {"engine": "fixture"},
        "source": "V0.3-T32",
        "notes": "Device result capture fixture only",
    }


def planned_experiments() -> list[dict]:
    prediction_values = [40.0, 42.0, 44.0, 46.0]
    result = []
    for protocol, prediction in zip(protocols(), prediction_values):
        features = {}
        for item in protocol.get("material_recipe") or []:
            features[str(item.get("source_feature") or ("formula::" + str(item["name"])))] = item["value"]
        for item in protocol.get("process_parameters") or []:
            features[str(item.get("source_feature") or item["name"])] = item["value"]
        result.append({
            "candidate_id": protocol["candidate_id"],
            "required_metrics": [TARGET],
            "expected_test_condition_signature": ROUND_CONDITION,
            "units": {TARGET: UNIT},
            "features": features,
            "prediction_snapshot": {
                TARGET: {
                    "value": prediction,
                    "std": 2.0,
                    "source": "V030_T32_FIXTURE_PREDICTION",
                }
            },
        })
    return result


def simulator_profile() -> dict:
    profile = device_profile("SIM_T32_CAPTURE")
    profile["name"] = "T32 Automatic Result Capture Simulator"
    profile["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "result_capture": True,
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root) / "v030" / "fixtures" / "t32"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    payloads = {
        "campaign_create.json": campaign_create(),
        "round_plan.json": round_plan(),
        "planned_experiments.json": planned_experiments(),
        "protocols.json": protocols(),
        "device_profile.json": simulator_profile(),
    }
    for name, payload in payloads.items():
        (root/name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("V0.3-T32 FIXTURE BUILDER")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"project_id: {PROJECT_ID}")
    print("planned_experiments: 4")
    print(f"target_metric: {TARGET}")
    print(f"test_condition: {ROUND_CONDITION}")
    print("device_adapter: SimulatorDeviceAdapter")
    print("result_origin: SIMULATOR_FIXTURE")
    print("is_real_measurement: false")
    print("fixture_only: true")
    print()
    print("V0.3-T32 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
