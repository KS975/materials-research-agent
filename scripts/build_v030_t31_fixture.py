from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from experiments.protocol import ExperimentProtocolBuilder
from experiments.safety import SAFETY_POLICY_STAGE
from scripts.build_v030_t30_fixture import (
    device_profiles as t30_device_profiles,
    telemetry_candidate,
    telemetry_template,
)


def safety_policy(device_id: str = "SIM_T31_MAIN") -> dict:
    return {
        "stage": SAFETY_POLICY_STAGE,
        "policy_id": "V030_T31_POLICY_V1",
        "device_id": device_id,
        "protocol_limits": {
            "process::加工温度": {"min": 180, "max": 240, "unit": "°C"},
            "process::压力": {"min": 0, "max": 12, "unit": "MPa"},
            "process::螺杆转速": {"min": 50, "max": 500, "unit": "rpm"},
        },
        "runtime_limits": {
            "temperature_c": {"min": 0, "max": 245},
            "pressure_mpa": {"min": 0, "max": 12},
            "rpm": {"min": 0, "max": 550},
        },
        "blocked_alarm_codes": [
            "DEVICE_ERROR",
            "SENSOR_FAULT",
            "COMMUNICATION_LOSS",
        ],
        "require_operator_ack": True,
        "metadata": {
            "fixture_only": True,
            "real_device": False,
            "note": "T31 deterministic simulator safety policy.",
        },
    }


def normal_protocol() -> dict:
    candidate = telemetry_candidate()
    candidate["candidate_id"] = "V030_T31_SAFE_EXP"
    return ExperimentProtocolBuilder(telemetry_template()).build(candidate)


def overlimit_protocol() -> dict:
    candidate = deepcopy(telemetry_candidate())
    candidate["candidate_id"] = "V030_T31_PROTOCOL_OVERLIMIT"
    # T27 template permits <= 280 °C, while T31 operational policy permits <= 240 °C.
    # This proves T31 is an independent stricter safety boundary.
    candidate["features"]["process::加工温度"] = {
        "value": 250,
        "unit": "°C",
    }
    return ExperimentProtocolBuilder(telemetry_template()).build(candidate)


def device_profile(device_id: str = "SIM_T31_MAIN", *, fault: bool = False) -> dict:
    base = deepcopy(t30_device_profiles()["fault" if fault else "normal"])
    base["device_id"] = device_id
    base["name"] = "T31 Safety Simulator " + device_id
    base["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "safety_test": True,
    }
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root) / "v030" / "fixtures" / "t31"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    payloads = {
        "safety_policy.json": safety_policy(),
        "normal_protocol.json": normal_protocol(),
        "overlimit_protocol.json": overlimit_protocol(),
        "device_profile.json": device_profile(),
        "fault_device_profile.json": device_profile("SIM_T31_FAULT", fault=True),
    }
    for name, payload in payloads.items():
        (root / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("V0.3-T31 FIXTURE BUILDER")
    print(f"policy: {root / 'safety_policy.json'}")
    print("protocols: 2")
    print("device_profiles: 2")
    print("protocol_temperature_limit_c: 240")
    print("runtime_temperature_stop_c: 245")
    print("runtime_pressure_stop_mpa: 12")
    print("operator_ack_required: true")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T31 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
