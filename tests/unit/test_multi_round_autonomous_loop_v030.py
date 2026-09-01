from __future__ import annotations

from pathlib import Path

import pytest

from experiments import (
    AutonomousLoopConflictError,
    AutonomousMultiRoundLoop,
    CampaignStore,
    DatasetVersionStore,
    ExperimentalResultService,
)
from scripts.build_v030_t34_fixture import (
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
    r1 = store.add_round(CAMPAIGN_ID, plan=round1_plan())
    results.register_planned_experiments(
        CAMPAIGN_ID,
        round_id=r1["round_id"],
        experiments=planned_experiments(),
    )
    datasets.register_base_csv(
        project_id=PROJECT_ID,
        dataset_version=DATASET_VERSIONS[0],
        source_csv=fixture / "dataset_v001.csv",
        metadata={"fixture": True},
    )
    return store, datasets, r1["round_id"], fixture / "candidate_pool.csv"


def _run(root: Path):
    store, datasets, rid, pool = _setup(root)
    loop = AutonomousMultiRoundLoop(root)
    report = loop.run(
        campaign_id=CAMPAIGN_ID,
        first_round_id=rid,
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        dataset_versions=DATASET_VERSIONS,
        challenger_model_versions=CHALLENGER_MODEL_VERSIONS,
        rounds_to_run=3,
        active_incumbent_model_version="model_v001",
        batch_size=5,
        scheduler_timeout_ticks=30,
    )
    return {
        "root": root,
        "store": store,
        "datasets": datasets,
        "rid": rid,
        "pool": pool,
        "loop": loop,
        "report": report,
    }


@pytest.fixture(scope="module")
def loop_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("t34_multi_round_case")
    return _run(root)


def test_three_rounds_all_complete(loop_case):
    report = loop_case["report"]
    assert report["round_count"] == 3
    assert report["round_statuses"] == ["COMPLETED"] * 3
    assert report["all_rounds_completed"] is True


def test_fifteen_results_are_automatically_captured(loop_case):
    experiments = loop_case["report"]["experiments"]
    assert experiments["total_planned"] == 15
    assert experiments["automatic_capture_count"] == 15
    assert experiments["receipt_count"] == 15
    assert experiments["manual_result_submission_count"] == 0
    assert experiments["is_real_measurement"] is False


def test_no_cross_round_candidate_or_feature_duplicates(loop_case):
    experiments = loop_case["report"]["experiments"]
    assert experiments["duplicate_candidate_ids"] == 0
    assert experiments["duplicate_feature_points"] == 0


def test_dataset_lineage_is_35_40_45_50(loop_case):
    dataset = loop_case["report"]["dataset"]
    assert dataset["lineage"] == DATASET_VERSIONS
    assert dataset["row_counts"] == [35, 40, 45, 50]
    assert dataset["rows_added_per_round"] == [5, 5, 5]


def test_all_dataset_versions_verify(loop_case):
    datasets = loop_case["datasets"]
    expected = [35, 40, 45, 50]
    for version, rows in zip(DATASET_VERSIONS, expected):
        assert datasets.verify(PROJECT_ID, version)["row_count"] == rows


def test_safety_and_telemetry_remain_clean(loop_case):
    safety = loop_case["report"]["safety"]
    assert safety["runtime_safety_stop_count"] == 0
    assert safety["all_telemetry_completed"] is True
    assert safety["all_telemetry_hash_chains_valid"] is True
    assert safety["automatic_resume_used"] is False
    assert safety["real_device_connected"] is False


def test_model_governance_never_auto_activates(loop_case):
    model = loop_case["report"]["model_governance"]
    assert len(model["decisions"]) == 3
    assert "BLOCKED" not in model["decisions"]
    assert model["automatic_activation_count"] == 0
    assert model["all_rounds_no_auto_activation"] is True
    assert model["active_model_versions"] == ["model_v001"] * 3


def test_bo_never_selects_ood(loop_case):
    bo = loop_case["report"]["bayesian_optimization"]
    assert bo["bo_transition_count"] == 2
    assert bo["selected_out_of_domain_count"] == 0


def test_final_round_does_not_create_round4(loop_case):
    report = loop_case["report"]
    campaign = loop_case["store"].load(CAMPAIGN_ID)
    assert report["bayesian_optimization"]["final_round_created_next_round"] is False
    assert len(campaign["rounds"]) == 3
    assert report["round_reports"][-1]["next_round_id"] is None


def test_each_round_report_has_expected_dataset_growth(loop_case):
    reports = loop_case["report"]["round_reports"]
    assert [r["dataset"]["row_count_before"] for r in reports] == [35, 40, 45]
    assert [r["dataset"]["row_count_after"] for r in reports] == [40, 45, 50]
    assert [r["dataset"]["added_row_count"] for r in reports] == [5, 5, 5]


def test_completed_loop_replay_is_idempotent(loop_case):
    loop = loop_case["loop"]
    replay = loop.run(
        campaign_id=CAMPAIGN_ID,
        first_round_id=loop_case["rid"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=loop_case["pool"],
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        dataset_versions=DATASET_VERSIONS,
        challenger_model_versions=CHALLENGER_MODEL_VERSIONS,
        rounds_to_run=3,
    )
    assert replay["idempotent_replay"] is True
    assert len(loop_case["store"].load(CAMPAIGN_ID)["rounds"]) == 3
    assert loop_case["datasets"].verify(PROJECT_ID, DATASET_VERSIONS[-1])["row_count"] == 50


def test_gate_block_stops_before_first_round(tmp_path):
    store, datasets, rid, pool = _setup(tmp_path)
    gate = gate_pass()
    gate["official_model_allowed"] = False
    with pytest.raises(AutonomousLoopConflictError):
        AutonomousMultiRoundLoop(tmp_path).run(
            campaign_id=CAMPAIGN_ID,
            first_round_id=rid,
            protocol_template=protocol_template(),
            device_profile=device_profile(),
            safety_policy=safety_policy(),
            candidate_pool_csv=pool,
            target_metric=TARGET,
            target_unit=UNIT,
            gate=gate,
            dataset_versions=DATASET_VERSIONS,
            challenger_model_versions=CHALLENGER_MODEL_VERSIONS,
            rounds_to_run=3,
        )
    assert store.load(CAMPAIGN_ID)["rounds"][0]["status"] == "PLANNED"


def test_boundary_is_explicit(loop_case):
    boundary = loop_case["report"]["boundary"]
    assert boundary["bounded_rounds"] is True
    assert boundary["rounds_to_run"] == 3
    assert boundary["no_round_after_final"] is True
    assert boundary["real_device_connected"] is False
    assert boundary["model_promotion_never_auto_approved"] is True
    assert boundary["crash_resume_owned_by_t35"] is True
