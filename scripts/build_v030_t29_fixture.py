from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from experiments.device import DEVICE_PROFILE_STAGE
from experiments.protocol import ExperimentProtocolBuilder
from scripts.build_v030_t27_fixture import candidates, fixture_template


def scheduler_candidates() -> list[dict]:
    base = candidates()["valid"]
    rows = []
    settings = [
        ("V030_T29_EXP01", 44.0, 51.0, 5.0, 225.0, 280),
        ("V030_T29_EXP02", 42.0, 53.0, 5.0, 230.0, 320),
        ("V030_T29_EXP03", 40.0, 55.0, 5.0, 235.0, 360),
        ("V030_T29_EXP04", 46.0, 50.0, 4.0, 220.0, 260),
        ("V030_T29_EXP05", 41.0, 54.0, 5.0, 240.0, 400),
        ("V030_T29_EXP06", 43.0, 52.0, 5.0, 228.0, 300),
        ("V030_T29_EXP07", 45.0, 50.0, 5.0, 232.0, 340),
    ]
    for candidate_id, abs_v, pc_v, tough, temp, rpm in settings:
        row = deepcopy(base)
        row["candidate_id"] = candidate_id
        row["source_context"] = {
            "campaign_id": "V030_T29_DEMO",
            "round_id": "R001",
        }
        row["features"]["formula::ABS"] = abs_v
        row["features"]["formula::PC"] = pc_v
        row["features"]["formula::增韧剂"] = tough
        row["features"]["process::加工温度"] = {"value": temp, "unit": "°C"}
        row["features"]["process::螺杆转速"] = {"value": rpm, "unit": "rpm"}
        rows.append(row)
    return rows


def protocol_documents() -> list[dict]:
    builder = ExperimentProtocolBuilder(fixture_template())
    return [builder.build(row) for row in scheduler_candidates()]


def device_profiles() -> dict[str, dict]:
    roles = ["material_dispenser", "compounder", "impact_tester"]
    template_id = fixture_template()["template_id"]
    common = {
        "stage": DEVICE_PROFILE_STAGE,
        "adapter_type": "simulator",
        "online": True,
        "supported_roles": roles,
        "supported_template_ids": [template_id],
        "metadata": {"fixture_only": True, "real_device": False},
    }
    return {
        "fast": {
            **common,
            "device_id": "SIM_T29_FAST",
            "name": "T29 快速模拟设备",
            "capabilities": {"pause": True, "cancel": True, "progress_per_tick": 50},
        },
        "aux": {
            **common,
            "device_id": "SIM_T29_AUX",
            "name": "T29 辅助模拟设备",
            "capabilities": {"pause": True, "cancel": True, "progress_per_tick": 34},
        },
        "slow": {
            **common,
            "device_id": "SIM_T29_SLOW",
            "name": "T29 超时模拟设备",
            "capabilities": {"pause": True, "cancel": True, "progress_per_tick": 20},
        },
        "flaky": {
            **common,
            "device_id": "SIM_T29_FLAKY",
            "name": "T29 瞬时故障模拟设备",
            "capabilities": {"pause": True, "cancel": True, "progress_per_tick": 50},
            "fault_injection": {"tick": True},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root) / "v030" / "fixtures" / "t29"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    protocols = protocol_documents()
    profiles = device_profiles()
    (root / "protocols.json").write_text(
        json.dumps(protocols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "device_profiles.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("V0.3-T29 FIXTURE BUILDER")
    print(f"protocols: {len(protocols)}")
    print(f"device_profiles: {len(profiles)}")
    print("scheduler_devices_for_priority_demo: 1")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T29 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
