from __future__ import annotations

from pathlib import Path

import pytest

from experiments import (
    AutonomousRoundConflictError,
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
    write_csvs,
)


def _setup(root: Path):
    fixture = root / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    write_csvs(fixture)

    store = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    results = ExperimentalResultService(str(root))

    c = campaign_create()
    store.create(
        campaign_id=c["campaign_id"],
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    r = store.add_round(CAMPAIGN_ID, plan=round1_plan())
    results.register_planned_experiments(
        CAMPAIGN_ID,
        round_id=r["round_id"],
        experiments=planned_experiments(),
    )
    datasets.register_base_csv(
        project_id=PROJECT_ID,
        dataset_version=BASE_DATASET_VERSION,
        source_csv=fixture / "dataset_v001.csv",
        metadata={"fixture": True},
    )
    return store, datasets, r["round_id"], fixture / "candidate_pool.csv"


def _run(root: Path):
    store, datasets, rid, pool = _setup(root)
    controller = AutonomousRoundController(root)
    report = controller.run_one_round(
        campaign_id=CAMPAIGN_ID,
        round_id=rid,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=CHILD_DATASET_VERSION,
        incumbent_model_version="model_v001",
        challenger_model_version="model_v002",
        next_batch_size=5,
        scheduler_timeout_ticks=30,
    )
    return {
        "root": root,
        "store": store,
        "datasets": datasets,
        "rid": rid,
        "pool": pool,
        "controller": controller,
        "report": report,
    }


@pytest.fixture(scope="module")
def autonomous_case(tmp_path_factory):
    # T23 model retraining + T24 BO are intentionally executed once for this
    # module; the assertions below inspect the same immutable T33 report.
    root = tmp_path_factory.mktemp("t33_autonomous_case")
    return _run(root)


def test_full_autonomous_round_completes_five_jobs(autonomous_case):
    report = autonomous_case["report"]
    assert report["scheduler"]["counts"]["COMPLETED"] == 5
    assert report["source_round_status"] == "COMPLETED"


def test_protocol_and_safety_preflight_all_pass(autonomous_case):
    report = autonomous_case["report"]
    assert report["protocol"]["ready_count"] == 5
    assert report["protocol"]["blocked_count"] == 0
    assert report["safety"]["preflight_pass_count"] == 5
    assert report["safety"]["runtime_safety_stop_count"] == 0


def test_telemetry_is_complete_and_integrity_valid(autonomous_case):
    report = autonomous_case["report"]
    assert report["telemetry"]["all_completed"] is True
    assert report["telemetry"]["all_hash_chains_valid"] is True
    assert report["telemetry"]["all_simulator"] is True


def test_no_manual_result_submission(autonomous_case):
    report = autonomous_case["report"]
    assert report["result_capture"]["automatic_capture_count"] == 5
    assert report["result_capture"]["manual_result_submission_count"] == 0
    assert report["result_capture"]["receipt_count"] == 5
    assert report["result_capture"]["is_real_measurement"] is False


def test_t21_evaluates_all_round_results(autonomous_case):
    report = autonomous_case["report"]
    assert report["evaluation"]["evaluated"] == 5
    assert report["evaluation"]["mae"] >= 0
    assert Path(report["evaluation"]["report_json"]).exists()


def test_t22_creates_immutable_child_dataset(autonomous_case):
    report = autonomous_case["report"]
    datasets = autonomous_case["datasets"]
    assert report["dataset"]["row_count_before"] == 35
    assert report["dataset"]["added_row_count"] == 5
    assert report["dataset"]["row_count_after"] == 40
    assert datasets.verify(
        PROJECT_ID, BASE_DATASET_VERSION
    )["row_count"] == 35
    assert datasets.verify(
        PROJECT_ID, CHILD_DATASET_VERSION
    )["row_count"] == 40


def test_t23_never_auto_activates_challenger(autonomous_case):
    report = autonomous_case["report"]
    assert report["model_governance"]["decision"] != "BLOCKED"
    assert report["model_governance"]["automatic_activation"] is False
    assert report["model_governance"]["active_model_version"] == "model_v001"


def test_t24_creates_one_planned_next_round(autonomous_case):
    report = autonomous_case["report"]
    campaign = autonomous_case["store"].load(CAMPAIGN_ID)
    assert len(campaign["rounds"]) == 2
    next_round = campaign["rounds"][1]
    assert next_round["round_id"] == report["next_round"]["round_id"]
    assert next_round["status"] == "PLANNED"
    assert next_round["plan"]["dataset_version"] == CHILD_DATASET_VERSION
    assert len(next_round["experiments"]) == 5


def test_completed_controller_replay_is_idempotent(autonomous_case):
    controller = autonomous_case["controller"]
    rid = autonomous_case["rid"]
    pool = autonomous_case["pool"]
    report = autonomous_case["report"]
    replay = controller.run_one_round(
        campaign_id=CAMPAIGN_ID,
        round_id=rid,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        child_dataset_version=CHILD_DATASET_VERSION,
        next_batch_size=5,
    )
    assert replay["idempotent_replay"] is True
    assert replay["next_round"]["round_id"] == report["next_round"]["round_id"]
    assert len(autonomous_case["store"].load(CAMPAIGN_ID)["rounds"]) == 2


def test_gate_block_prevents_autonomous_execution(tmp_path):
    store, datasets, rid, pool = _setup(tmp_path)
    bad_gate = dict(gate_pass())
    bad_gate["training_allowed"] = False
    with pytest.raises(AutonomousRoundConflictError):
        AutonomousRoundController(tmp_path).run_one_round(
            campaign_id=CAMPAIGN_ID,
            round_id=rid,
            protocol_template=protocol_template(),
            device_profile=device_profile(),
            safety_policy=safety_policy(),
            candidate_pool_csv=pool,
            target_metric=TARGET,
            target_unit=UNIT,
            gate=bad_gate,
            child_dataset_version=CHILD_DATASET_VERSION,
        )
    assert store.load(CAMPAIGN_ID)["rounds"][0]["status"] == "PLANNED"


def test_report_boundary_explicitly_stays_one_round_only(autonomous_case):
    report = autonomous_case["report"]
    assert report["boundary"]["one_round_only"] is True
    assert report["boundary"]["next_round_not_auto_started"] is True
    assert report["boundary"]["real_device_connected"] is False
    assert report["boundary"]["model_promotion_never_auto_approved"] is True
