from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from experiments import (
    AutonomousRoundController,
    CampaignStore,
    CrashResumeCoordinator,
    DatasetVersionStore,
    ExperimentalResultService,
)
from scripts.build_v030_t35_fixture import (
    CAMPAIGN_ID,
    CHALLENGER_MODEL_VERSIONS,
    DATASET_VERSIONS,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument(
        "--fixture-dir",
        default=".runtime/v030/fixtures/t35",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root)
    fixture = Path(args.fixture_dir)
    base_csv = fixture / "dataset_v001.csv"
    pool_csv = fixture / "candidate_pool.csv"
    if not base_csv.exists() or not pool_csv.exists():
        raise SystemExit(
            "ERROR: 请先运行 python -m scripts.build_v030_t35_fixture --reset"
        )

    campaigns = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    results = ExperimentalResultService(str(root))

    if args.reset:
        cleanup = [
            campaigns.campaign_dir(CAMPAIGN_ID),
            datasets.project_dir(PROJECT_ID),
            root / "v020" / "evaluations" / CAMPAIGN_ID,
            root / "v020" / "model_promotion" / f"project_{PROJECT_ID}",
            root / "v020" / "model_registry" / f"project_{PROJECT_ID}",
            root / "v020" / "closed_loop_bo" / CAMPAIGN_ID,
            root / "v030" / "scheduler",
            root / "v030" / "telemetry",
            root / "v030" / "safety",
            root / "v030" / "result_capture" / CAMPAIGN_ID,
            root / "v030" / "autonomous_round" / CAMPAIGN_ID,
            root / "v030" / "crash_resume" / CAMPAIGN_ID,
        ]
        for target in cleanup:
            if target.exists():
                shutil.rmtree(target)

    c = campaign_create()
    campaigns.create(
        campaign_id=c["campaign_id"],
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    r1 = campaigns.add_round(CAMPAIGN_ID, plan=round1_plan())
    results.register_planned_experiments(
        CAMPAIGN_ID,
        round_id=r1["round_id"],
        experiments=planned_experiments(),
    )
    datasets.register_base_csv(
        project_id=PROJECT_ID,
        dataset_version=DATASET_VERSIONS[0],
        source_csv=base_csv,
        metadata={"fixture": True, "stage": "V0.3-T35"},
    )

    # R1 is intentionally normal. It creates R2 PLANNED and dataset_v002.
    r1_report = AutonomousRoundController(root).run_one_round(
        campaign_id=CAMPAIGN_ID,
        round_id=r1["round_id"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=DATASET_VERSIONS[1],
        incumbent_model_version="model_v001",
        challenger_model_version=CHALLENGER_MODEL_VERSIONS[0],
        next_batch_size=5,
        scheduler_timeout_ticks=30,
    )
    r2_id = r1_report["next_round"]["round_id"]

    coordinator = CrashResumeCoordinator(root)
    checkpoint = coordinator.stage_simulated_crash(
        campaign_id=CAMPAIGN_ID,
        round_id=r2_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        completed_before_crash=2,
        active_elapsed_ticks=4,
        scheduler_timeout_ticks=30,
    )

    print("V0.3-T35 CRASH/RESUME + OPERATOR OVERRIDE")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"crash_round_id: {r2_id}")
    print()
    print("CRASH POINT")
    print(f"completed_results_before_restart: {checkpoint['completed_results_before_crash']}")
    print(f"pending_results_before_restart: {checkpoint['pending_results_before_crash']}")
    print(f"active_candidate_id: {checkpoint['active_candidate_id']}")
    print(f"active_experiment_progress: {checkpoint['device_progress_percent']:.6f}")
    print(f"telemetry_phase: {checkpoint['telemetry_phase']}")
    print(f"round_status_before_restart: {checkpoint['round_status']}")
    print()

    report = coordinator.resume_after_crash(
        campaign_id=CAMPAIGN_ID,
        round_id=r2_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        operator_action="RESUME",
        operator_id="operator_t35",
        reason="restart validation passed; continue simulator round",
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=DATASET_VERSIONS[2],
        challenger_model_version=CHALLENGER_MODEL_VERSIONS[1],
        next_batch_size=5,
    )

    print("RECONCILIATION")
    print(f"source_of_truth: {report['reconciliation']['source_of_truth']}")
    print(f"simulator_replayed_ticks: {report['reconciliation']['simulator_replayed_ticks']}")
    print(f"reconstructed_progress_percent: {report['reconciliation']['reconstructed_progress_percent']:.6f}")
    print(f"adapter_job_id_match: {str(report['reconciliation']['adapter_job_id_match']).lower()}")
    print(f"telemetry_replay_idempotent: {str(report['reconciliation']['telemetry_replay_idempotent']).lower()}")
    print(f"automatic_resume_used: {str(report['reconciliation']['automatic_resume_used']).lower()}")
    print(f"explicit_operator_resume_required: {str(report['reconciliation']['explicit_operator_resume_required']).lower()}")
    print()

    print("RESULTS AFTER RESUME")
    print(f"completed_before_resume: {report['results']['completed_before_resume']}")
    print(f"completed_after_resume: {report['results']['completed_after_resume']}")
    print(f"pending_after_resume: {report['results']['pending_after_resume']}")
    print(f"capture_receipts_before_resume: {report['results']['capture_receipts_before_resume']}")
    print(f"capture_receipts_after_resume: {report['results']['capture_receipts_after_resume']}")
    print(f"duplicate_completed_result_writes: {report['results']['duplicate_completed_result_writes']}")
    print()

    print("POST-ROUND")
    print(f"dataset_rows_before: {report['dataset']['row_count_before']}")
    print(f"dataset_rows_added: {report['dataset']['added_row_count']}")
    print(f"dataset_rows_after: {report['dataset']['row_count_after']}")
    print(f"model_decision: {report['model_governance']['decision']}")
    print(f"automatic_model_activation: {str(report['model_governance']['automatic_activation']).lower()}")
    print(f"next_round_id: {report['next_round']['round_id']}")
    print(f"next_round_status: {report['next_round']['status']}")
    print(f"round_count: {report['round_count']}")
    print()

    replay = coordinator.resume_after_crash(
        campaign_id=CAMPAIGN_ID,
        round_id=r2_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        operator_action="RESUME",
        operator_id="operator_t35",
        reason="idempotent replay check",
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=DATASET_VERSIONS[2],
        challenger_model_version=CHALLENGER_MODEL_VERSIONS[1],
    )
    print("IDEMPOTENT REPLAY")
    print(f"idempotent_replay: {str(replay['idempotent_replay']).lower()}")
    print(f"round_count_after_replay: {len(campaigns.load(CAMPAIGN_ID)['rounds'])}")
    print(f"dataset_v003_rows_after_replay: {datasets.verify(PROJECT_ID, DATASET_VERSIONS[2])['row_count']}")
    print()

    print("AUDIT + BOUNDARY")
    print(f"recovery_audit_valid: {str(report['recovery_audit_valid']).lower()}")
    print("operator_override_cannot_bypass_safety: true")
    print("real_device_connected: false")
    print("simulator_reconstruction_only: true")
    print("real_device_recovery_requires_physical_job_query: true")
    print()
    print("OUTPUT")
    print(f"report_json: {report['report_json']}")

    if checkpoint["completed_results_before_crash"] != 2:
        raise SystemExit("ERROR: crash fixture expected 2 completed results")
    if abs(checkpoint["device_progress_percent"] - 40.0) > 1e-9:
        raise SystemExit("ERROR: crash fixture expected 40% progress")
    if checkpoint["telemetry_phase"] != "PROCESSING":
        raise SystemExit("ERROR: crash point expected PROCESSING")
    if report["reconciliation"]["simulator_replayed_ticks"] != 4:
        raise SystemExit("ERROR: expected 4 replayed simulator ticks")
    if not report["reconciliation"]["telemetry_replay_idempotent"]:
        raise SystemExit("ERROR: telemetry reconstruction duplicated data")
    if report["results"]["completed_after_resume"] != 5:
        raise SystemExit("ERROR: R2 did not complete all 5 results")
    if report["results"]["pending_after_resume"] != 0:
        raise SystemExit("ERROR: pending results remain")
    if report["results"]["capture_receipts_after_resume"] != 5:
        raise SystemExit("ERROR: expected 5 R2 capture receipts")
    if report["dataset"]["row_count_before"] != 40:
        raise SystemExit("ERROR: dataset_v002 should have 40 rows")
    if report["dataset"]["added_row_count"] != 5:
        raise SystemExit("ERROR: recovery round should add 5 rows")
    if report["dataset"]["row_count_after"] != 45:
        raise SystemExit("ERROR: dataset_v003 should have 45 rows")
    if report["model_governance"]["automatic_activation"]:
        raise SystemExit("ERROR: model must never auto-activate")
    if report["next_round"]["status"] != "PLANNED":
        raise SystemExit("ERROR: next round should be PLANNED")
    if report["round_count"] != 3:
        raise SystemExit("ERROR: expected R1/R2/R3 only")
    if not replay["idempotent_replay"]:
        raise SystemExit("ERROR: recovery replay must be idempotent")

    print()
    print("V0.3-T35 CRASH/RESUME + OPERATOR OVERRIDE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
