from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.build_v030_t34_fixture import (
    device_profile as t34_device_profile,
    gate_pass as t34_gate_pass,
    planned_experiments as t34_planned_experiments,
    protocol_template as t34_protocol_template,
    round1_plan as t34_round1_plan,
    safety_policy as t34_safety_policy,
    write_csvs as t34_write_csvs,
)


VALIDATION_ID = "V030_T36_FINAL"
NORMAL_CAMPAIGN_ID = "V030_T36_NORMAL"
NORMAL_PROJECT_ID = 9036
RECOVERY_CAMPAIGN_ID = "V030_T36_RECOVERY"
RECOVERY_PROJECT_ID = 9037
TARGET = "冲击强度"
UNIT = "kJ/m²"
DATASET_VERSIONS = [
    "dataset_v001", "dataset_v002", "dataset_v003", "dataset_v004"
]
CHALLENGER_MODELS = ["model_v002", "model_v003", "model_v004"]


def protocol_template(project_id: int, label: str) -> dict:
    template = deepcopy(t34_protocol_template())
    template["template_id"] = f"V030_T36_{label}_PROTOCOL_V1"
    template["name"] = f"T36 {label} End-to-End Protocol"
    template["project_id"] = int(project_id)
    template["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "t36_final_validation": True,
        "scenario": label,
    }
    return template


def device_profile(project_id: int, label: str) -> dict:
    profile = deepcopy(t34_device_profile())
    profile["device_id"] = f"SIM_T36_{label}"
    profile["name"] = f"T36 {label} Simulator"
    profile["supported_template_ids"] = [
        protocol_template(project_id, label)["template_id"]
    ]
    profile["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "t36_final_validation": True,
        "scenario": label,
    }
    return profile


def safety_policy(project_id: int, label: str) -> dict:
    policy = deepcopy(t34_safety_policy())
    policy["policy_id"] = f"V030_T36_{label}_SAFETY_V1"
    policy["device_id"] = device_profile(project_id, label)["device_id"]
    policy["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "t36_final_validation": True,
        "scenario": label,
    }
    return policy


def campaign_create(campaign_id: str, project_id: int, scenario: str) -> dict:
    return {
        "campaign_id": campaign_id,
        "project_id": int(project_id),
        "name": f"T36 {scenario} final validation",
        "target_metrics": [TARGET],
        "metadata": {
            "fixture": True,
            "purpose": "V0.3-T36 end-to-end acceptance",
            "scenario": scenario,
            "real_measurement": False,
        },
    }


def round1_plan() -> dict:
    plan = deepcopy(t34_round1_plan())
    plan["dataset_version"] = DATASET_VERSIONS[0]
    plan["source"] = "V0.3-T36_final_validation"
    return plan


def planned_experiments(prefix: str) -> list[dict]:
    rows = deepcopy(t34_planned_experiments())
    for index, row in enumerate(rows, start=1):
        row["candidate_id"] = f"{prefix}_R1_{index:02d}"
    return rows


def gate_pass(project_id: int) -> dict:
    gate = deepcopy(t34_gate_pass())
    gate["project_id"] = int(project_id)
    return gate


def write_csvs(out: Path, project_id: int, prefix: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t34_write_csvs(out)

    base = out / "dataset_v001.csv"
    with base.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    for index, row in enumerate(rows, start=1):
        row["candidate_id"] = f"{prefix}_BASE_{index:03d}"
        row["project_id"] = str(project_id)
    with base.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

    pool = out / "candidate_pool.csv"
    with pool.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    for index, row in enumerate(rows, start=1):
        row["candidate_id"] = f"{prefix}_POOL_{index:04d}"
    with pool.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default=".runtime/v030/fixtures/t36"
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if args.reset and out.exists():
        shutil.rmtree(out)
    normal = out / "normal"
    recovery = out / "recovery"
    write_csvs(normal, NORMAL_PROJECT_ID, "V030_T36_N")
    write_csvs(recovery, RECOVERY_PROJECT_ID, "V030_T36_R")

    payloads = {
        "normal/campaign_create.json": campaign_create(
            NORMAL_CAMPAIGN_ID, NORMAL_PROJECT_ID, "NORMAL"
        ),
        "normal/protocol_template.json": protocol_template(
            NORMAL_PROJECT_ID, "NORMAL"
        ),
        "normal/device_profile.json": device_profile(
            NORMAL_PROJECT_ID, "NORMAL"
        ),
        "normal/safety_policy.json": safety_policy(
            NORMAL_PROJECT_ID, "NORMAL"
        ),
        "normal/gate_pass.json": gate_pass(NORMAL_PROJECT_ID),
        "normal/round1_plan.json": round1_plan(),
        "normal/planned_experiments.json": planned_experiments("V030_T36_N"),
        "recovery/campaign_create.json": campaign_create(
            RECOVERY_CAMPAIGN_ID, RECOVERY_PROJECT_ID, "RECOVERY"
        ),
        "recovery/protocol_template.json": protocol_template(
            RECOVERY_PROJECT_ID, "RECOVERY"
        ),
        "recovery/device_profile.json": device_profile(
            RECOVERY_PROJECT_ID, "RECOVERY"
        ),
        "recovery/safety_policy.json": safety_policy(
            RECOVERY_PROJECT_ID, "RECOVERY"
        ),
        "recovery/gate_pass.json": gate_pass(RECOVERY_PROJECT_ID),
        "recovery/round1_plan.json": round1_plan(),
        "recovery/planned_experiments.json": planned_experiments("V030_T36_R"),
    }
    for relative, payload in payloads.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("V0.3-T36 FIXTURE BUILDER")
    print(f"validation_id: {VALIDATION_ID}")
    print(f"normal_campaign: {NORMAL_CAMPAIGN_ID}")
    print(f"recovery_campaign: {RECOVERY_CAMPAIGN_ID}")
    print("normal_target_rounds: 3")
    print("normal_target_experiments: 15")
    print("normal_dataset_rows: 35 -> 40 -> 45 -> 50")
    print("recovery_crash_point: R002 / experiment 3 / 40%")
    print("operator_override_cases: RESUME / CANCEL_JOB / ABORT_ROUND / SAFETY_BLOCK")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T36 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
