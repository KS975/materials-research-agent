from pathlib import Path

import pytest

from runtime.v030_ui import (
    V030UIError,
    build_autonomy_overview,
    latest_v030_campaign_id_for_project,
    resolve_operator_inputs,
)


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / ".runtime"


def test_latest_v030_campaign_by_project_reads_t36_runtime():
    assert (
        latest_v030_campaign_id_for_project(
            FIXTURE_ROOT, 9035
        )
        == "V030_T35_DEMO"
    )


def test_normal_t36_overview_contains_three_round_autonomous_evidence():
    data = build_autonomy_overview(
        FIXTURE_ROOT,
        campaign_id="V030_T35_DEMO",
    )
    assert data["kind"] == "v030_autonomy"
    assert data["campaign"]["project_id"] == 9035
    assert data["summary"]["round_count"] == 3
    assert data["summary"]["automatic_capture_count"] == 10
    assert [x["status"] for x in data["rounds"]] == [
        "COMPLETED",
        "COMPLETED",
        "PLANNED",
    ]
    assert [x["row_count"] for x in data["datasets"]] == [
        35,
        40,
        45,
    ]
    assert data["autonomous_loop"] is None
    assert data["boundary"]["simulator_only"] is True
    assert data["boundary"]["automatic_model_activation"] is False


def test_round_runtime_exposes_scheduler_telemetry_and_safety():
    data = build_autonomy_overview(
        FIXTURE_ROOT,
        campaign_id="V030_T35_DEMO",
    )
    r1 = data["rounds"][0]
    assert r1["scheduler"]["counts"]["COMPLETED"] == 5
    assert r1["telemetry"]["session_count"] == 5
    assert r1["telemetry"]["completed_sessions"] == 5
    assert r1["telemetry"]["all_simulator"] is True
    assert r1["safety"]["state"] == "SAFE"
    assert r1["capture_receipts"] == 5


def test_recovery_campaign_exposes_crash_and_recovery_report():
    data = build_autonomy_overview(
        FIXTURE_ROOT,
        campaign_id="V030_T35_DEMO",
    )
    r2 = data["rounds"][1]
    assert r2["crash_checkpoint"] is not None
    assert r2["crash_checkpoint"][
        "completed_results_before_crash"
    ] == 2
    assert r2["recovery_report"] is not None
    assert r2["recovery_report"]["recovery_audit_valid"] is True


def test_safety_view_exposes_fail_closed_boundary():
    data = build_autonomy_overview(
        FIXTURE_ROOT,
        campaign_id="V030_T35_DEMO",
    )
    r2 = data["rounds"][1]
    assert r2["safety"]["state"] == "SAFE"
    assert r2["safety"]["automatic_resume_allowed"] is False
    assert (
        data["boundary"]["operator_override_cannot_bypass_safety"]
        is True
    )


def test_operator_inputs_are_fail_closed_when_not_configured():
    data = resolve_operator_inputs(
        FIXTURE_ROOT,
        project_id=9035,
    )
    assert data["base_ready"] is False
    assert "protocol_template" in data["missing_base"]


def test_project_mismatch_is_rejected():
    with pytest.raises(V030UIError):
        build_autonomy_overview(
            FIXTURE_ROOT,
            campaign_id="V030_T35_DEMO",
            project_id=9999,
        )


def test_v030_status_never_claims_real_device_validation():
    data = build_autonomy_overview(
        FIXTURE_ROOT,
        campaign_id="V030_T35_DEMO",
    )
    assert data["summary"]["real_device_connected"] is False
    assert data["boundary"]["real_device_validation_completed"] is False
    assert (
        data["boundary"][
            "simulator_results_are_not_real_material_measurements"
        ]
        is True
    )
