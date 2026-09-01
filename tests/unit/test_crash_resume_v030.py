from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments import (
    AutonomousRoundController,
    CampaignStore,
    CrashResumeCoordinator,
    DatasetVersionStore,
    ExperimentalResultService,
    OperatorOverrideBlockedError,
    SafetyInterlock,
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
    write_csvs,
)


def _setup_base(root: Path, campaign_id=CAMPAIGN_ID, project_id=PROJECT_ID):
    fixture = root / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    write_csvs(fixture)
    campaigns = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    results = ExperimentalResultService(str(root))

    c = campaign_create(campaign_id, project_id)
    campaigns.create(
        campaign_id=c["campaign_id"],
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    plan = round1_plan()
    r1 = campaigns.add_round(campaign_id, plan=plan)
    rows = planned_experiments()
    results.register_planned_experiments(
        campaign_id,
        round_id=r1["round_id"],
        experiments=rows,
    )
    datasets.register_base_csv(
        project_id=project_id,
        dataset_version=DATASET_VERSIONS[0],
        source_csv=fixture / "dataset_v001.csv",
        metadata={"fixture": True},
    )
    return campaigns, datasets, results, r1["round_id"], fixture / "candidate_pool.csv"


@pytest.fixture(scope="module")
def resume_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("t35_resume")
    campaigns, datasets, results, r1_id, pool = _setup_base(root)
    r1_report = AutonomousRoundController(root).run_one_round(
        campaign_id=CAMPAIGN_ID,
        round_id=r1_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool,
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
    )
    report = coordinator.resume_after_crash(
        campaign_id=CAMPAIGN_ID,
        round_id=r2_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        operator_action="RESUME",
        operator_id="operator_test",
        reason="test recovery",
        candidate_pool_csv=pool,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=DATASET_VERSIONS[2],
        challenger_model_version=CHALLENGER_MODEL_VERSIONS[1],
        next_batch_size=5,
    )
    return {
        "root": root,
        "campaigns": campaigns,
        "datasets": datasets,
        "results": results,
        "pool": pool,
        "r2_id": r2_id,
        "coordinator": coordinator,
        "checkpoint": checkpoint,
        "report": report,
    }


def _setup_operator_case(tmp_path, suffix):
    campaign_id = f"V030_T35_{suffix}"
    project_id = PROJECT_ID + sum(ord(ch) for ch in suffix)
    campaigns, datasets, results, r1_id, pool = _setup_base(
        tmp_path, campaign_id, project_id
    )
    coordinator = CrashResumeCoordinator(tmp_path)
    checkpoint = coordinator.stage_simulated_crash(
        campaign_id=campaign_id,
        round_id=r1_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        completed_before_crash=1,
        active_elapsed_ticks=4,
    )
    return (
        campaign_id,
        project_id,
        campaigns,
        coordinator,
        checkpoint,
        r1_id,
        pool,
    )


def test_crash_point_is_r2_third_experiment_at_40_percent(resume_case):
    cp = resume_case["checkpoint"]
    assert cp["completed_results_before_crash"] == 2
    assert cp["pending_results_before_crash"] == 3
    assert cp["device_progress_percent"] == 40.0
    assert cp["telemetry_phase"] == "PROCESSING"


def test_reconcile_uses_persisted_scheduler_and_same_device_job(resume_case):
    report = resume_case["report"]
    rec = report["reconciliation"]
    assert rec["source_of_truth"] == "PERSISTED_T29_SCHEDULER"
    assert rec["simulator_replayed_ticks"] == 4
    assert rec["reconstructed_progress_percent"] == 40.0
    assert rec["adapter_job_id_match"] is True


def test_telemetry_replay_at_crash_point_is_idempotent(resume_case):
    assert (
        resume_case["report"]["reconciliation"][
            "telemetry_replay_idempotent"
        ]
        is True
    )


def test_completed_results_are_not_written_twice(resume_case):
    report = resume_case["report"]
    assert report["results"]["completed_before_resume"] == 2
    assert report["results"]["completed_after_resume"] == 5
    assert report["results"]["pending_after_resume"] == 0
    assert report["results"]["capture_receipts_before_resume"] == 2
    assert report["results"]["capture_receipts_after_resume"] == 5
    assert report["results"]["duplicate_completed_result_writes"] == 0


def test_dataset_updates_once_after_resumed_round(resume_case):
    report = resume_case["report"]
    assert report["dataset"]["row_count_before"] == 40
    assert report["dataset"]["added_row_count"] == 5
    assert report["dataset"]["row_count_after"] == 45
    assert resume_case["datasets"].verify(
        PROJECT_ID, DATASET_VERSIONS[1]
    )["row_count"] == 40
    assert resume_case["datasets"].verify(
        PROJECT_ID, DATASET_VERSIONS[2]
    )["row_count"] == 45


def test_model_is_never_auto_activated_after_recovery(resume_case):
    report = resume_case["report"]
    assert report["model_governance"]["automatic_activation"] is False
    assert report["boundary"]["model_promotion_auto_approved"] is False


def test_recovery_creates_exactly_one_next_planned_round(resume_case):
    report = resume_case["report"]
    campaign = resume_case["campaigns"].load(CAMPAIGN_ID)
    assert len(campaign["rounds"]) == 3
    assert report["next_round"]["status"] == "PLANNED"
    assert report["next_round"]["round_id"] == campaign["rounds"][2]["round_id"]


def test_recovery_replay_is_idempotent(resume_case):
    report = resume_case["report"]
    replay = resume_case["coordinator"].resume_after_crash(
        campaign_id=CAMPAIGN_ID,
        round_id=resume_case["r2_id"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        operator_action="RESUME",
        operator_id="operator_test",
        reason="replay",
        candidate_pool_csv=resume_case["pool"],
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=DATASET_VERSIONS[2],
        challenger_model_version=CHALLENGER_MODEL_VERSIONS[1],
    )
    assert replay["idempotent_replay"] is True
    assert len(resume_case["campaigns"].load(CAMPAIGN_ID)["rounds"]) == 3
    assert resume_case["datasets"].verify(
        PROJECT_ID, DATASET_VERSIONS[2]
    )["row_count"] == 45


def test_cancel_job_override_stops_automatic_continuation(tmp_path):
    (
        campaign_id,
        project_id,
        campaigns,
        coordinator,
        checkpoint,
        round_id,
        pool,
    ) = _setup_operator_case(tmp_path, "CANCEL")
    out = coordinator.resume_after_crash(
        campaign_id=campaign_id,
        round_id=round_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        operator_action="CANCEL_JOB",
        operator_id="op_cancel",
        reason="operator chooses to cancel current experiment",
    )
    assert out["scheduler_job_status"] == "CANCELLED"
    assert out["automatic_continuation"] is False
    assert out["dataset_updated"] is False
    assert out["next_round_created"] is False


def test_abort_round_override_cancels_round_without_dataset_or_bo(tmp_path):
    (
        campaign_id,
        project_id,
        campaigns,
        coordinator,
        checkpoint,
        round_id,
        pool,
    ) = _setup_operator_case(tmp_path, "ABORT")
    out = coordinator.resume_after_crash(
        campaign_id=campaign_id,
        round_id=round_id,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        operator_action="ABORT_ROUND",
        operator_id="op_abort",
        reason="operator aborts round",
    )
    assert out["round_status"] == "CANCELLED"
    assert out["active_jobs_after_abort"] == 0
    assert out["dataset_updated"] is False
    assert out["model_governance_run"] is False
    assert out["next_round_created"] is False


def test_operator_resume_cannot_bypass_latched_safety_stop(tmp_path):
    (
        campaign_id,
        project_id,
        campaigns,
        coordinator,
        checkpoint,
        round_id,
        pool,
    ) = _setup_operator_case(tmp_path, "SAFETY")
    active_candidate = checkpoint["active_candidate_id"]
    scheduler_path = Path(checkpoint["scheduler_state_path"])
    scheduler_state = __import__("json").loads(
        scheduler_path.read_text(encoding="utf-8")
    )
    active_job = next(
        job for job in scheduler_state["jobs"].values()
        if job["candidate_id"] == active_candidate
    )

    interlock = SafetyInterlock(
        interlock_id=coordinator.safety_interlock_id(
            campaign_id, round_id, active_candidate
        ),
        policy=safety_policy(),
        runtime_root=tmp_path,
    )
    tampered_protocol = deepcopy(active_job["protocol"])
    tampered_protocol["content_sha256"] = "tampered"
    blocked = interlock.check_protocol(tampered_protocol)
    assert blocked["state"] == "SAFETY_STOP"

    with pytest.raises(OperatorOverrideBlockedError):
        coordinator.resume_after_crash(
            campaign_id=campaign_id,
            round_id=round_id,
            protocol_template=protocol_template(),
            device_profile=device_profile(),
            safety_policy=safety_policy(),
            operator_action="RESUME",
            operator_id="op_safety",
            reason="must not bypass safety",
            candidate_pool_csv=pool,
            target_metric=TARGET,
            target_unit=UNIT,
            gate=gate_pass(),
            child_dataset_version=DATASET_VERSIONS[1],
            challenger_model_version="model_safety",
        )


def test_recovery_audit_hash_chain_is_valid(resume_case):
    assert resume_case["report"]["recovery_audit_valid"] is True
    assert resume_case["coordinator"].verify_integrity(
        CAMPAIGN_ID, resume_case["r2_id"]
    ) is True


def test_boundary_explicitly_limits_reconstruction_to_simulator(resume_case):
    boundary = resume_case["report"]["boundary"]
    assert boundary["real_device_connected"] is False
    assert boundary["simulator_reconstruction_only"] is True
    assert boundary["real_device_recovery_requires_physical_job_query"] is True
    assert boundary["operator_override_cannot_bypass_safety"] is True
