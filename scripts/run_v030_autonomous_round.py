from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from experiments import (
    AutonomousRoundController,
    CampaignStore,
    DatasetVersionStore,
    ExperimentalResultService,
)
from scripts.build_v030_t33_fixture import (
    BASE_DATASET_VERSION,
    CAMPAIGN_ID,
    CHILD_DATASET_VERSION,
    PROJECT_ID,
    TARGET,
    UNIT,
    campaign_create,
    device_profile,
    gate_pass,
    planned_experiments,
    protocol_template,
    round1_plan,
    safety_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument(
        "--fixture-dir",
        default=".runtime/v030/fixtures/t33",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root)
    fixture = Path(args.fixture_dir)
    base_csv = fixture / "dataset_v001.csv"
    pool_csv = fixture / "candidate_pool.csv"

    if not base_csv.exists() or not pool_csv.exists():
        raise SystemExit(
            "ERROR: 请先运行 python -m scripts.build_v030_t33_fixture --reset"
        )

    campaign_store = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    results = ExperimentalResultService(str(root))

    if args.reset:
        cleanup = [
            campaign_store.campaign_dir(CAMPAIGN_ID),
            datasets.project_dir(PROJECT_ID),
            root / "v020" / "evaluations" / CAMPAIGN_ID,
            root / "v020" / "model_promotion" / f"project_{PROJECT_ID}",
            root / "v020" / "model_registry" / f"project_{PROJECT_ID}",
            root / "v020" / "closed_loop_bo" / CAMPAIGN_ID,
            root / "v030" / "protocols",
            root / "v030" / "scheduler" / f"{CAMPAIGN_ID}_{CAMPAIGN_ID}-R001_T33",
            root / "v030" / "telemetry",
            root / "v030" / "safety",
            root / "v030" / "result_capture" / CAMPAIGN_ID,
            root / "v030" / "autonomous_round" / CAMPAIGN_ID,
        ]
        for target in cleanup:
            if target.exists():
                shutil.rmtree(target)

    c = campaign_create()
    campaign_store.create(
        campaign_id=c["campaign_id"],
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    round1 = campaign_store.add_round(
        CAMPAIGN_ID,
        plan=round1_plan(),
    )
    results.register_planned_experiments(
        CAMPAIGN_ID,
        round_id=round1["round_id"],
        experiments=planned_experiments(),
    )
    datasets.register_base_csv(
        project_id=PROJECT_ID,
        dataset_version=BASE_DATASET_VERSION,
        source_csv=base_csv,
        metadata={"fixture": True, "stage": "V0.3-T33"},
    )

    controller = AutonomousRoundController(root)

    print("V0.3-T33 AUTONOMOUS ROUND CONTROLLER")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"round_id: {round1['round_id']}")
    print()
    print("BOUNDARY")
    print("device_adapter: SimulatorDeviceAdapter only")
    print("real_device_connected: false")
    print("manual_result_submission_required: false")
    print("automatic_model_activation: false")
    print("one_round_only: true")
    print()

    report = controller.run_one_round(
        campaign_id=CAMPAIGN_ID,
        round_id=round1["round_id"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=CHILD_DATASET_VERSION,
        incumbent_model_version="model_v001",
        challenger_model_version="model_v002",
        next_batch_size=5,
        scheduler_timeout_ticks=30,
        max_scheduler_ticks=500,
    )

    print("AUTONOMOUS EXECUTION")
    print(f"protocol_ready: {report['protocol']['ready_count']}")
    print(f"protocol_blocked: {report['protocol']['blocked_count']}")
    print(f"scheduler_completed: {report['scheduler']['counts']['COMPLETED']}")
    print(f"scheduler_ticks: {report['scheduler']['ticks']}")
    print(f"safety_preflight_pass: {report['safety']['preflight_pass_count']}")
    print(f"runtime_safety_stop_count: {report['safety']['runtime_safety_stop_count']}")
    print(f"telemetry_all_completed: {str(report['telemetry']['all_completed']).lower()}")
    print(f"telemetry_hash_chain_valid: {str(report['telemetry']['all_hash_chains_valid']).lower()}")
    print()

    print("AUTOMATIC RESULT CAPTURE")
    print(f"automatic_capture_count: {report['result_capture']['automatic_capture_count']}")
    print(f"manual_result_submission_count: {report['result_capture']['manual_result_submission_count']}")
    print(f"capture_receipts: {report['result_capture']['receipt_count']}")
    print(f"measurement_origin: {report['result_capture']['measurement_origin']}")
    print(f"is_real_measurement: {str(report['result_capture']['is_real_measurement']).lower()}")
    print()

    print("T21 EVALUATION")
    print(f"evaluated: {report['evaluation']['evaluated']}")
    print(f"MAE: {report['evaluation']['mae']:.6f}")
    print(f"RMSE: {report['evaluation']['rmse']:.6f}")
    r2 = report["evaluation"]["r2"]
    print(f"R2: {r2:.6f}" if r2 is not None else "R2: null")
    print()

    print("T22 DATASET UPDATE")
    print(f"parent_dataset: {report['dataset']['parent_dataset_version']}")
    print(f"child_dataset: {report['dataset']['child_dataset_version']}")
    print(f"rows_before: {report['dataset']['row_count_before']}")
    print(f"rows_added: {report['dataset']['added_row_count']}")
    print(f"rows_after: {report['dataset']['row_count_after']}")
    print()

    print("T23 MODEL GOVERNANCE")
    print(f"decision: {report['model_governance']['decision']}")
    print(f"active_model: {report['model_governance']['active_model_version']}")
    print(f"challenger_model: {report['model_governance']['challenger_model_version']}")
    print(f"automatic_activation: {str(report['model_governance']['automatic_activation']).lower()}")
    print()

    print("T24 NEXT ROUND")
    print(f"next_round_id: {report['next_round']['round_id']}")
    print(f"next_round_status: {report['next_round']['status']}")
    print(f"next_round_dataset: {report['next_round']['dataset_version']}")
    print(f"next_round_planned_experiments: {report['next_round']['planned_experiments']}")
    print("next_round_auto_started: false")
    print()

    replay = controller.run_one_round(
        campaign_id=CAMPAIGN_ID,
        round_id=round1["round_id"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=CHILD_DATASET_VERSION,
        next_batch_size=5,
    )
    campaign = campaign_store.load(CAMPAIGN_ID)

    print("IDEMPOTENT REPLAY")
    print(f"idempotent_replay: {str(replay['idempotent_replay']).lower()}")
    print(f"round_count_after_replay: {len(campaign['rounds'])}")
    print(f"same_next_round: {str(replay['next_round']['round_id'] == report['next_round']['round_id']).lower()}")
    print()

    print("OUTPUT")
    print(f"report_json: {report['report_json']}")
    print()
    print("EXECUTION BOUNDARY")
    print("T33 autonomously executes exactly one simulator Round.")
    print("It creates the next Round as PLANNED but never auto-starts it.")
    print("It never auto-approves a T23 model promotion.")
    print("Synthetic simulator measurements are engineering fixtures, not real material data.")

    if report["protocol"]["ready_count"] != 5:
        raise SystemExit("ERROR: expected 5 READY protocols")
    if report["scheduler"]["counts"]["COMPLETED"] != 5:
        raise SystemExit("ERROR: scheduler expected 5 COMPLETED jobs")
    if report["safety"]["runtime_safety_stop_count"] != 0:
        raise SystemExit("ERROR: normal T33 fixture should have no safety stop")
    if not report["telemetry"]["all_completed"]:
        raise SystemExit("ERROR: telemetry sessions not all completed")
    if not report["telemetry"]["all_hash_chains_valid"]:
        raise SystemExit("ERROR: telemetry hash chain invalid")
    if report["result_capture"]["automatic_capture_count"] != 5:
        raise SystemExit("ERROR: expected 5 automatic captures")
    if report["result_capture"]["manual_result_submission_count"] != 0:
        raise SystemExit("ERROR: manual submission count must be 0")
    if report["evaluation"]["evaluated"] != 5:
        raise SystemExit("ERROR: expected 5 T21 evaluated results")
    if report["dataset"]["row_count_before"] != 35:
        raise SystemExit("ERROR: dataset parent rows should be 35")
    if report["dataset"]["added_row_count"] != 5:
        raise SystemExit("ERROR: dataset should add 5 rows")
    if report["dataset"]["row_count_after"] != 40:
        raise SystemExit("ERROR: dataset child rows should be 40")
    if report["model_governance"]["decision"] == "BLOCKED":
        raise SystemExit("ERROR: T33 model governance unexpectedly BLOCKED")
    if report["model_governance"]["automatic_activation"]:
        raise SystemExit("ERROR: model must not auto-activate")
    if report["next_round"]["status"] != "PLANNED":
        raise SystemExit("ERROR: next round must remain PLANNED")
    if report["next_round"]["planned_experiments"] != 5:
        raise SystemExit("ERROR: expected 5 BO next experiments")
    if not replay["idempotent_replay"]:
        raise SystemExit("ERROR: completed controller replay should be idempotent")
    if len(campaign["rounds"]) != 2:
        raise SystemExit("ERROR: replay must not create Round 3")

    print()
    print("V0.3-T33 AUTONOMOUS ROUND CONTROLLER PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
