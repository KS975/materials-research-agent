from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments.device import DEVICE_PROFILE_STAGE
from experiments.protocol import ExperimentProtocolBuilder
from scripts.build_v030_t27_fixture import candidates, fixture_template


def device_profiles() -> dict:
    common_roles = ["material_dispenser", "compounder", "impact_tester"]
    template_id = fixture_template()["template_id"]
    return {
        "main": {
            "stage": DEVICE_PROFILE_STAGE,
            "device_id": "SIM_T28_MAIN",
            "name": "T28 全流程模拟设备",
            "adapter_type": "simulator",
            "online": True,
            "supported_roles": common_roles,
            "supported_template_ids": [template_id],
            "capabilities": {"pause": True, "cancel": True, "progress_per_tick": 40},
            "metadata": {"fixture_only": True},
        },
        "offline": {
            "stage": DEVICE_PROFILE_STAGE,
            "device_id": "SIM_T28_OFFLINE",
            "name": "T28 离线模拟设备",
            "adapter_type": "simulator",
            "online": False,
            "supported_roles": common_roles,
            "supported_template_ids": [template_id],
            "metadata": {"fixture_only": True},
        },
        "limited": {
            "stage": DEVICE_PROFILE_STAGE,
            "device_id": "SIM_T28_LIMITED",
            "name": "T28 能力不足模拟设备",
            "adapter_type": "simulator",
            "online": True,
            "supported_roles": ["compounder"],
            "supported_template_ids": [template_id],
            "metadata": {"fixture_only": True},
        },
        "fault": {
            "stage": DEVICE_PROFILE_STAGE,
            "device_id": "SIM_T28_FAULT",
            "name": "T28 故障注入模拟设备",
            "adapter_type": "simulator",
            "online": True,
            "supported_roles": common_roles,
            "supported_template_ids": [template_id],
            "fault_injection": {"start": True},
            "metadata": {"fixture_only": True},
        },
    }


def protocol_documents() -> dict:
    builder = ExperimentProtocolBuilder(fixture_template())
    docs = candidates()
    return {
        "ready": builder.build(docs["valid"]),
        "second_ready": builder.build(docs["optional_missing"]),
        "blocked": builder.build(docs["unsafe"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root) / "v030" / "fixtures" / "t28"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    profiles = device_profiles()
    protocols = protocol_documents()
    (root / "device_profiles.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "protocols.json").write_text(
        json.dumps(protocols, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("V0.3-T28 FIXTURE BUILDER")
    print(f"profiles: {len(profiles)}")
    print(f"protocols: {len(protocols)}")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T28 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
