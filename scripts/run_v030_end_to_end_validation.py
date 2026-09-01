from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from experiments import (
    AutonomousMultiRoundLoop,
    AutonomousRoundController,
    CampaignStore,
    CrashResumeCoordinator,
    DatasetVersionStore,
    ExperimentalResultService,
    FinalAutonomousValidationService,
    OperatorOverrideBlockedError,
    SafetyInterlock,
)
from scripts.build_v030_t36_fixture import (
    CHALLENGER_MODELS,
    DATASET_VERSIONS,
    NORMAL_CAMPAIGN_ID,
    NORMAL_PROJECT_ID,
    RECOVERY_CAMPAIGN_ID,
    RECOVERY_PROJECT_ID,
    TARGET,
    UNIT,
    VALIDATION_ID,
    campaign_create,
    device_profile,
    gate_pass,
    planned_experiments,
    protocol_template,
    round1_plan,
    safety_policy,
)


def _setup_campaign(root, campaign_id, project_id, prefix, fixture_dir):
    campaigns = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    results = ExperimentalResultService(str(root))
    c = campaign_create(campaign_id, project_id, prefix)
    campaigns.create(
        campaign_id=campaign_id,
        project_id=project_id,
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    r1 = campaigns.add_round(campaign_id, plan=round1_plan())
    results.register_planned_experiments(
        campaign_id,
        round_id=r1["round_id"],
        experiments=planned_experiments(prefix),
    )
    datasets.register_base_csv(
        project_id=project_id,
        dataset_version=DATASET_VERSIONS[0],
        source_csv=fixture_dir / "dataset_v001.csv",
        metadata={"fixture": True, "stage": "V0.3-T36"},
    )
    return campaigns, datasets, results, r1["round_id"]


def _cleanup(root, campaign_id, project_id):
    campaigns = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    paths = [
        campaigns.campaign_dir(campaign_id),
        datasets.project_dir(project_id),
        root / "v020" / "evaluations" / campaign_id,
        root / "v020" / "model_promotion" / f"project_{project_id}",
        root / "v020" / "model_registry" / f"project_{project_id}",
        root / "v020" / "closed_loop_bo" / campaign_id,
        root / "v030" / "result_capture" / campaign_id,
        root / "v030" / "autonomous_round" / campaign_id,
        root / "v030" / "autonomous_loop" / campaign_id,
        root / "v030" / "crash_resume" / campaign_id,
    ]
    for path in paths:
        if path.exists():
            shutil.rmtree(path)

    # T29/T30/T31 stores are shared top-level namespaces. Remove only entries
    # belonging to this T36 campaign so --reset is repeatable without deleting
    # unrelated user runtime data.
    for shared_name in ("scheduler", "telemetry", "safety"):
        shared = root / "v030" / shared_name
        if not shared.exists():
            continue
        for child in list(shared.iterdir()):
            if campaign_id in child.name and child.exists():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()


def _operator_case(
    *, root, fixture_dir, campaign_id, project_id, label, action
):
    campaigns, datasets, results, r1_id = _setup_campaign(
        root, campaign_id, project_id, label, fixture_dir
    )
    coordinator = CrashResumeCoordinator(root)
    cp = coordinator.stage_simulated_crash(
        campaign_id=campaign_id,
        round_id=r1_id,
        protocol_template=protocol_template(project_id, label),
        device_profile=device_profile(project_id, label),
        safety_policy=safety_policy(project_id, label),
        completed_before_crash=1,
        active_elapsed_ticks=4,
    )
    out = coordinator.resume_after_crash(
        campaign_id=campaign_id,
        round_id=r1_id,
        protocol_template=protocol_template(project_id, label),
        device_profile=device_profile(project_id, label),
        safety_policy=safety_policy(project_id, label),
        operator_action=action,
        operator_id=f"operator_{label.lower()}",
        reason=f"T36 {action} acceptance",
    )
    return out


def _safety_block_case(*, root, fixture_dir, campaign_id, project_id, label):
    campaigns, datasets, results, r1_id = _setup_campaign(
        root, campaign_id, project_id, label, fixture_dir
    )
    coordinator = CrashResumeCoordinator(root)
    cp = coordinator.stage_simulated_crash(
        campaign_id=campaign_id,
        round_id=r1_id,
        protocol_template=protocol_template(project_id, label),
        device_profile=device_profile(project_id, label),
        safety_policy=safety_policy(project_id, label),
        completed_before_crash=1,
        active_elapsed_ticks=4,
    )
    scheduler_state = json.loads(
        Path(cp["scheduler_state_path"]).read_text(encoding="utf-8")
    )
    active_job = next(
        job for job in scheduler_state["jobs"].values()
        if job["candidate_id"] == cp["active_candidate_id"]
    )
    interlock = SafetyInterlock(
        interlock_id=coordinator.safety_interlock_id(
            campaign_id, r1_id, cp["active_candidate_id"]
        ),
        policy=safety_policy(project_id, label),
        runtime_root=root,
    )
    tampered = deepcopy(active_job["protocol"])
    tampered["content_sha256"] = "T36_FORCED_SAFETY_LATCH"
    blocked = interlock.check_protocol(tampered)
    if blocked["state"] != "SAFETY_STOP":
        raise RuntimeError("T36 safety block fixture failed to latch")
    try:
        coordinator.resume_after_crash(
            campaign_id=campaign_id,
            round_id=r1_id,
            protocol_template=protocol_template(project_id, label),
            device_profile=device_profile(project_id, label),
            safety_policy=safety_policy(project_id, label),
            operator_action="RESUME",
            operator_id="operator_safety",
            reason="must be blocked by T31",
            candidate_pool_csv=fixture_dir / "candidate_pool.csv",
            target_metric=TARGET,
            target_unit=UNIT,
            gate=gate_pass(project_id),
            child_dataset_version=DATASET_VERSIONS[1],
            challenger_model_version="model_safety",
        )
    except OperatorOverrideBlockedError as exc:
        return True, exc.code
    return False, "NO_ERROR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument(
        "--fixture-dir", default=".runtime/v030/fixtures/t36"
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(args.runtime_root)
    fixture_root = Path(args.fixture_dir)
    normal_fixture = fixture_root / "normal"
    recovery_fixture = fixture_root / "recovery"
    if not (normal_fixture / "dataset_v001.csv").exists():
        raise SystemExit(
            "ERROR: 请先运行 python -m scripts.build_v030_t36_fixture --reset"
        )

    operator_specs = [
        ("V030_T36_CANCEL", 9038),
        ("V030_T36_ABORT", 9039),
        ("V030_T36_SAFETY", 9040),
    ]
    if args.reset:
        _cleanup(root, NORMAL_CAMPAIGN_ID, NORMAL_PROJECT_ID)
        _cleanup(root, RECOVERY_CAMPAIGN_ID, RECOVERY_PROJECT_ID)
        for cid, pid in operator_specs:
            _cleanup(root, cid, pid)
        final_dir = root / "v030" / "final_validation" / VALIDATION_ID
        if final_dir.exists():
            shutil.rmtree(final_dir)

    # Scenario A: exact 3-round healthy autonomous loop.
    normal_campaigns, normal_datasets, _, normal_r1 = _setup_campaign(
        root,
        NORMAL_CAMPAIGN_ID,
        NORMAL_PROJECT_ID,
        "V030_T36_N",
        normal_fixture,
    )
    normal_loop = AutonomousMultiRoundLoop(root).run(
        campaign_id=NORMAL_CAMPAIGN_ID,
        first_round_id=normal_r1,
        protocol_template=protocol_template(NORMAL_PROJECT_ID, "NORMAL"),
        device_profile=device_profile(NORMAL_PROJECT_ID, "NORMAL"),
        safety_policy=safety_policy(NORMAL_PROJECT_ID, "NORMAL"),
        candidate_pool_csv=normal_fixture / "candidate_pool.csv",
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(NORMAL_PROJECT_ID),
        dataset_versions=DATASET_VERSIONS,
        challenger_model_versions=CHALLENGER_MODELS,
        rounds_to_run=3,
        active_incumbent_model_version="model_v001",
        batch_size=5,
        scheduler_timeout_ticks=30,
    )

    # Scenario B: R1 normal, R2 crashes after 2 results + active third at 40%.
    recovery_campaigns, recovery_datasets, _, recovery_r1 = _setup_campaign(
        root,
        RECOVERY_CAMPAIGN_ID,
        RECOVERY_PROJECT_ID,
        "V030_T36_R",
        recovery_fixture,
    )
    recovery_r1_report = AutonomousRoundController(root).run_one_round(
        campaign_id=RECOVERY_CAMPAIGN_ID,
        round_id=recovery_r1,
        protocol_template=protocol_template(RECOVERY_PROJECT_ID, "RECOVERY"),
        device_profile=device_profile(RECOVERY_PROJECT_ID, "RECOVERY"),
        safety_policy=safety_policy(RECOVERY_PROJECT_ID, "RECOVERY"),
        candidate_pool_csv=recovery_fixture / "candidate_pool.csv",
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(RECOVERY_PROJECT_ID),
        child_dataset_version=DATASET_VERSIONS[1],
        incumbent_model_version="model_v001",
        challenger_model_version=CHALLENGER_MODELS[0],
        next_batch_size=5,
        scheduler_timeout_ticks=30,
    )
    recovery_r2 = recovery_r1_report["next_round"]["round_id"]
    coordinator = CrashResumeCoordinator(root)
    checkpoint = coordinator.stage_simulated_crash(
        campaign_id=RECOVERY_CAMPAIGN_ID,
        round_id=recovery_r2,
        protocol_template=protocol_template(RECOVERY_PROJECT_ID, "RECOVERY"),
        device_profile=device_profile(RECOVERY_PROJECT_ID, "RECOVERY"),
        safety_policy=safety_policy(RECOVERY_PROJECT_ID, "RECOVERY"),
        completed_before_crash=2,
        active_elapsed_ticks=4,
    )
    recovery = coordinator.resume_after_crash(
        campaign_id=RECOVERY_CAMPAIGN_ID,
        round_id=recovery_r2,
        protocol_template=protocol_template(RECOVERY_PROJECT_ID, "RECOVERY"),
        device_profile=device_profile(RECOVERY_PROJECT_ID, "RECOVERY"),
        safety_policy=safety_policy(RECOVERY_PROJECT_ID, "RECOVERY"),
        operator_action="RESUME",
        operator_id="operator_t36_recovery",
        reason="T36 verified recovery continue",
        candidate_pool_csv=recovery_fixture / "candidate_pool.csv",
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(RECOVERY_PROJECT_ID),
        child_dataset_version=DATASET_VERSIONS[2],
        challenger_model_version=CHALLENGER_MODELS[1],
        next_batch_size=5,
    )
    recovery_replay = coordinator.resume_after_crash(
        campaign_id=RECOVERY_CAMPAIGN_ID,
        round_id=recovery_r2,
        protocol_template=protocol_template(RECOVERY_PROJECT_ID, "RECOVERY"),
        device_profile=device_profile(RECOVERY_PROJECT_ID, "RECOVERY"),
        safety_policy=safety_policy(RECOVERY_PROJECT_ID, "RECOVERY"),
        operator_action="RESUME",
        operator_id="operator_t36_recovery",
        reason="T36 recovery idempotent replay",
        candidate_pool_csv=recovery_fixture / "candidate_pool.csv",
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(RECOVERY_PROJECT_ID),
        child_dataset_version=DATASET_VERSIONS[2],
        challenger_model_version=CHALLENGER_MODELS[1],
    )

    cancel = _operator_case(
        root=root,
        fixture_dir=recovery_fixture,
        campaign_id="V030_T36_CANCEL",
        project_id=9038,
        label="CANCEL",
        action="CANCEL_JOB",
    )
    abort = _operator_case(
        root=root,
        fixture_dir=recovery_fixture,
        campaign_id="V030_T36_ABORT",
        project_id=9039,
        label="ABORT",
        action="ABORT_ROUND",
    )
    safety_blocked, safety_code = _safety_block_case(
        root=root,
        fixture_dir=recovery_fixture,
        campaign_id="V030_T36_SAFETY",
        project_id=9040,
        label="SAFETY",
    )

    validator = FinalAutonomousValidationService(root)
    report = validator.build_report(
        validation_id=VALIDATION_ID,
        normal_loop_report=normal_loop,
        recovery_report=recovery,
        recovery_replay_report=recovery_replay,
        cancel_job_report=cancel,
        abort_round_report=abort,
        safety_resume_blocked=safety_blocked,
        safety_resume_error_code=safety_code,
        expected_normal_rounds=3,
        expected_experiments_per_round=5,
        expected_normal_dataset_rows=[35, 40, 45, 50],
        expected_recovery_dataset_rows=(40, 45),
    )
    replay = validator.build_report(
        validation_id=VALIDATION_ID,
        normal_loop_report=normal_loop,
        recovery_report=recovery,
        recovery_replay_report=recovery_replay,
        cancel_job_report=cancel,
        abort_round_report=abort,
        safety_resume_blocked=safety_blocked,
        safety_resume_error_code=safety_code,
        expected_normal_rounds=3,
        expected_experiments_per_round=5,
        expected_normal_dataset_rows=[35, 40, 45, 50],
        expected_recovery_dataset_rows=(40, 45),
    )

    print("V0.3-T36 END-TO-END AUTONOMOUS VALIDATION")
    print(f"validation_id: {VALIDATION_ID}")
    print()
    print("COMPONENT MATRIX")
    for name, row in report["component_checks"].items():
        print(f"{name}: {'PASS' if row['pass'] else 'FAIL'}")
    print(f"component_pass_count: {report['component_pass_count']}/{report['component_total']}")
    print()
    print("NORMAL 3-ROUND LOOP")
    print(f"round_count: {report['normal_loop']['round_count']}")
    print(f"total_experiments: {report['normal_loop']['total_experiments']}")
    print(f"automatic_captures: {report['normal_loop']['automatic_captures']}")
    print(f"manual_submissions: {report['normal_loop']['manual_submissions']}")
    print(f"dataset_row_counts: {report['normal_loop']['dataset_row_counts']}")
    print(f"bo_transition_count: {report['normal_loop']['bo_transition_count']}")
    print(f"bo_ood_selected: {report['normal_loop']['bo_ood_selected']}")
    print(f"automatic_model_activation_count: {report['normal_loop']['automatic_model_activation_count']}")
    print()
    print("CRASH / RESUME")
    print(f"completed_before_restart: {report['crash_resume']['completed_before_restart']}")
    print(f"pending_before_restart: {report['crash_resume']['pending_before_restart']}")
    print(f"crash_progress_percent: {report['crash_resume']['crash_progress_percent']:.6f}")
    print(f"completed_after_resume: {report['crash_resume']['completed_after_resume']}")
    print(f"dataset_rows_after_resume: {report['crash_resume']['dataset_rows_after_resume']}")
    print(f"recovery_replay_idempotent: {str(report['crash_resume']['replay_idempotent']).lower()}")
    print()
    print("OPERATOR OVERRIDE")
    print(f"cancel_job_status: {report['operator_override']['cancel_job']['status']}")
    print(f"cancel_automatic_continuation: {str(report['operator_override']['cancel_job']['automatic_continuation']).lower()}")
    print(f"abort_round_status: {report['operator_override']['abort_round']['round_status']}")
    print(f"abort_created_next_round: {str(report['operator_override']['abort_round']['next_round_created']).lower()}")
    print(f"safety_resume_blocked: {str(report['operator_override']['safety_resume_blocked']).lower()}")
    print(f"safety_resume_error_code: {report['operator_override']['safety_resume_error_code']}")
    print()
    print("BOUNDARY")
    print("simulator_only: true")
    print("real_device_connected: false")
    print("real_device_validation_completed: false")
    print("simulator_results_are_not_real_material_measurements: true")
    print("model_promotion_auto_approval_forbidden: true")
    print("operator_override_cannot_bypass_safety: true")
    print()
    print("FINAL")
    print(f"status: {report['status']}")
    print(f"report_sha256: {report['report_sha256']}")
    print(f"final_report_replay_idempotent: {str(replay['idempotent_replay']).lower()}")
    print(f"report_json: {report['report_json']}")

    if report["status"] != "PASS":
        raise SystemExit("ERROR: T36 final validation did not PASS")
    if report["component_pass_count"] != 9:
        raise SystemExit("ERROR: expected 9/9 component checks")
    if not replay["idempotent_replay"]:
        raise SystemExit("ERROR: final validation report replay not idempotent")

    print()
    print("V0.3-T36 END-TO-END AUTONOMOUS VALIDATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
