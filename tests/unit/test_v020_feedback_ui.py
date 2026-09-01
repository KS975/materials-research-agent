import json
import shutil
from pathlib import Path

import pytest

from experiments import CampaignStore, ExperimentalResultService
from runtime.v020_ui import (
    V020UIError,
    build_campaign_overview,
    close_round_for_ui,
    latest_campaign_id_for_project,
    start_round_for_ui,
    submit_result_for_ui,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / ".runtime"


def test_latest_campaign_by_project_reads_t26_runtime():
    assert latest_campaign_id_for_project(FIXTURE_ROOT, 9026) == "V020_T26_DEMO"


def test_t26_overview_contains_full_closed_loop_evidence():
    data = build_campaign_overview(FIXTURE_ROOT, campaign_id="V020_T26_DEMO")
    assert data["kind"] == "v020_feedback_loop"
    assert data["campaign"]["status"] == "COMPLETED"
    assert data["campaign"]["round_count"] == 3
    assert [x["row_count"] for x in data["datasets"]] == [35, 40, 45, 50]
    assert data["evaluation"]["aggregate"]["mae"] == pytest.approx(0.1600262151660985)
    assert data["model_promotion"]["decision"] == "REVIEW_REQUIRED"
    assert data["checkpoint"]["status"] == "COMPLETED"
    assert data["end_to_end"]["decision"] == "PASS"


def test_t24_overview_uses_latest_runtime_round_state():
    data = build_campaign_overview(FIXTURE_ROOT, campaign_id="V020_T24_DEMO")
    latest = data["latest_round"]
    status = latest["status"]

    # T24 runtime is intentionally mutable during UI acceptance.
    # Validate consistency instead of assuming the fixture is still PLANNED.
    assert status in {"PLANNED", "RUNNING", "PARTIALLY_COMPLETED", "COMPLETED"}
    assert latest["can_start"] is (status == "PLANNED")

    if data["campaign"]["status"] == "ACTIVE":
        assert f"状态为 {status}" in data["answer"]
        pending = len(latest["pending_experiments"])
        assert f"待处理实验 {pending} 组" in data["answer"]


def _copy_campaign_runtime(tmp_path: Path, campaign_id: str):
    src = FIXTURE_ROOT / "v020" / "campaigns" / campaign_id
    dst = tmp_path / "v020" / "campaigns" / campaign_id
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def test_ui_can_start_round_and_ingest_result(tmp_path):
    _copy_campaign_runtime(tmp_path, "V020_T24_DEMO")
    before = build_campaign_overview(tmp_path, campaign_id="V020_T24_DEMO")
    round_id = before["latest_round"]["round_id"]
    first = before["latest_round"]["pending_experiments"][0]
    started = start_round_for_ui(tmp_path, campaign_id="V020_T24_DEMO", round_id=round_id)
    assert started["latest_round"]["status"] == "RUNNING"
    assert "状态为 RUNNING" in started["answer"]
    assert "状态为 PLANNED" not in started["answer"]
    metric = first["required_metrics"][0]
    submitted = submit_result_for_ui(
        tmp_path,
        campaign_id="V020_T24_DEMO",
        round_id=round_id,
        payload={
            "candidate_id": first["candidate_id"],
            "status": "COMPLETED",
            "test_condition_signature": first["expected_test_condition_signature"],
            "measurements": {metric: 50.0},
            "units": {metric: first["units"][metric]},
            "failure_reason": "",
            "notes": "ui unit test",
        },
    )
    assert submitted["latest_round"]["status"] == "PARTIALLY_COMPLETED"
    assert "状态为 PARTIALLY_COMPLETED" in submitted["answer"]
    assert len(submitted["latest_round"]["pending_experiments"]) == 4


def test_ui_rejects_early_round_close(tmp_path):
    _copy_campaign_runtime(tmp_path, "V020_T24_DEMO")
    data = build_campaign_overview(tmp_path, campaign_id="V020_T24_DEMO")
    round_id = data["latest_round"]["round_id"]
    start_round_for_ui(tmp_path, campaign_id="V020_T24_DEMO", round_id=round_id)
    with pytest.raises(V020UIError):
        close_round_for_ui(tmp_path, campaign_id="V020_T24_DEMO", round_id=round_id)


def test_overview_safety_never_claims_automatic_model_replacement():
    data = build_campaign_overview(FIXTURE_ROOT, campaign_id="V020_T26_DEMO")
    assert data["safety"]["automatic_model_replacement"] is False
    assert data["safety"]["result_ingestion_uses_t20"] is True


def test_project_mismatch_is_rejected():
    with pytest.raises(V020UIError):
        build_campaign_overview(
            FIXTURE_ROOT,
            campaign_id="V020_T26_DEMO",
            project_id=9999,
        )
