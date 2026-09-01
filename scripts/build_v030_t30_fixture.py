from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from experiments.device import DEVICE_PROFILE_STAGE
from experiments.protocol import ExperimentProtocolBuilder
from scripts.build_v030_t27_fixture import candidates, fixture_template


def telemetry_template() -> dict:
    template = deepcopy(fixture_template())
    template["template_id"] = "V030_T30_POLYMER_TELEMETRY_V1"
    template["name"] = "T30 聚合物实验遥测 fixture"
    template["parameters"].append({
        "name": "压力",
        "source_feature": "process::压力",
        "section": "process_parameter",
        "kind": "continuous",
        "canonical_unit": "MPa",
        "default_input_unit": "MPa",
        "required": False,
        "safety": {"min": 0, "max": 50},
    })
    for step in template["process_steps"]:
        if step["step_id"] == "compound":
            step["parameters"].append("压力")
    template["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "note": "T30 deterministic telemetry fixture; not a real instrument profile.",
    }
    return template


def telemetry_candidate() -> dict:
    candidate = deepcopy(candidates()["valid"])
    candidate["candidate_id"] = "V030_T30_EXP01"
    candidate["source_context"] = {
        "campaign_id": "V030_T30_DEMO",
        "round_id": "R001",
    }
    candidate["features"]["process::压力"] = {
        "value": 8.5,
        "unit": "MPa",
    }
    return candidate


def protocol_document() -> dict:
    return ExperimentProtocolBuilder(
        telemetry_template()
    ).build(telemetry_candidate())


def device_profiles() -> dict[str, dict]:
    roles = ["material_dispenser", "compounder", "impact_tester"]
    common = {
        "stage": DEVICE_PROFILE_STAGE,
        "adapter_type": "simulator",
        "online": True,
        "supported_roles": roles,
        "supported_template_ids": [telemetry_template()["template_id"]],
        "capabilities": {
            "pause": True,
            "cancel": True,
            "progress_per_tick": 10,
        },
        "metadata": {
            "fixture_only": True,
            "real_device": False,
            "telemetry_source": "deterministic_simulator",
        },
    }
    return {
        "normal": {
            **common,
            "device_id": "SIM_T30_MAIN",
            "name": "T30 遥测模拟设备",
        },
        "fault": {
            **common,
            "device_id": "SIM_T30_FAULT",
            "name": "T30 故障遥测模拟设备",
            "fault_injection": {"tick": True},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root) / "v030" / "fixtures" / "t30"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    protocol = protocol_document()
    profiles = device_profiles()
    (root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "device_profiles.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("V0.3-T30 FIXTURE BUILDER")
    print(f"protocol: {root / 'protocol.json'}")
    print(f"device_profiles: {len(profiles)}")
    print("progress_per_tick: 10")
    print("telemetry_time_source: SIMULATOR_VIRTUAL_CLOCK")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T30 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
