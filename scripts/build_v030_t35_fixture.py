from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.build_v030_t34_fixture import (
    gate_pass as t34_gate_pass,
    planned_experiments as t34_planned_experiments,
    protocol_template as t34_protocol_template,
    round1_plan as t34_round1_plan,
    safety_policy as t34_safety_policy,
    device_profile as t34_device_profile,
    write_csvs as t34_write_csvs,
)


CAMPAIGN_ID = "V030_T35_DEMO"
PROJECT_ID = 9035
TARGET = "冲击强度"
UNIT = "kJ/m²"
DATASET_VERSIONS = [
    "dataset_v001",
    "dataset_v002",
    "dataset_v003",
]
CHALLENGER_MODEL_VERSIONS = [
    "model_v002",
    "model_v003",
]


def protocol_template():
    template = deepcopy(t34_protocol_template())
    template["template_id"] = "V030_T35_CRASH_RESUME_V1"
    template["name"] = "T35 Crash Resume Protocol"
    template["project_id"] = PROJECT_ID
    template["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "crash_resume": True,
    }
    return template


def device_profile():
    profile = deepcopy(t34_device_profile())
    profile["device_id"] = "SIM_T35_MAIN"
    profile["name"] = "T35 Crash Resume Simulator"
    profile["supported_template_ids"] = [
        protocol_template()["template_id"]
    ]
    profile["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "crash_resume": True,
    }
    return profile


def safety_policy():
    policy = deepcopy(t34_safety_policy())
    policy["policy_id"] = "V030_T35_SAFETY_V1"
    policy["device_id"] = device_profile()["device_id"]
    policy["metadata"] = {
        "fixture_only": True,
        "real_device": False,
        "crash_resume": True,
    }
    return policy


def campaign_create(campaign_id=CAMPAIGN_ID, project_id=PROJECT_ID):
    return {
        "campaign_id": campaign_id,
        "project_id": project_id,
        "name": "T35 Crash/Resume + Operator Override",
        "target_metrics": [TARGET],
        "metadata": {
            "fixture": True,
            "purpose": "V0.3-T35 crash resume acceptance",
            "real_measurement": False,
        },
    }


def round1_plan():
    plan = deepcopy(t34_round1_plan())
    plan["dataset_version"] = DATASET_VERSIONS[0]
    plan["source"] = "V0.3-T35_fixture"
    return plan


def planned_experiments():
    rows = deepcopy(t34_planned_experiments())
    for index, row in enumerate(rows, start=1):
        row["candidate_id"] = f"V030_T35_R1_{index:02d}"
    return rows


def gate_pass():
    gate = deepcopy(t34_gate_pass())
    gate["project_id"] = PROJECT_ID
    return gate


def write_csvs(out: Path):
    t34_write_csvs(out)
    # CSV content is synthetic fixture data and can be reused across project
    # ids because T22 overwrites lineage metadata at registration time.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".runtime/v030/fixtures/t35",
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

    print("V0.3-T35 FIXTURE BUILDER")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"project_id: {PROJECT_ID}")
    print("base_dataset_rows: 35")
    print("round1_planned_experiments: 5")
    print("crash_target_round: R002")
    print("crash_after_completed_results: 2")
    print("crash_active_experiment_index: 3")
    print("crash_active_progress_percent: 40")
    print("operator_actions: RESUME / CANCEL_JOB / ABORT_ROUND")
    print("real_device_connected: false")
    print("fixture_only: true")
    print()
    print("V0.3-T35 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
