from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import REQUIRED_COMPONENTS


@pytest.fixture(scope="module")
def t36_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("t36_case")
    runtime = root / ".runtime"
    fixture = runtime / "v030" / "fixtures" / "t36"
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_v030_t36_fixture",
            "--output-dir",
            str(fixture),
            "--reset",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_v030_end_to_end_validation",
            "--runtime-root",
            str(runtime),
            "--fixture-dir",
            str(fixture),
            "--reset",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    report_path = (
        runtime
        / "v030"
        / "final_validation"
        / "V030_T36_FINAL"
        / "v030_final_validation_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "runtime": runtime,
        "report": report,
        "stdout": run.stdout,
        "report_path": report_path,
    }


def test_t36_final_status_pass(t36_case):
    assert t36_case["report"]["status"] == "PASS"


def test_all_nine_v030_components_pass(t36_case):
    report = t36_case["report"]
    assert report["component_pass_count"] == 9
    assert report["component_total"] == 9
    assert set(report["component_checks"]) == set(REQUIRED_COMPONENTS)
    assert all(row["pass"] is True for row in report["component_checks"].values())


def test_normal_loop_is_exactly_three_rounds_and_fifteen_results(t36_case):
    normal = t36_case["report"]["normal_loop"]
    assert normal["round_count"] == 3
    assert normal["total_experiments"] == 15
    assert normal["automatic_captures"] == 15
    assert normal["manual_submissions"] == 0


def test_dataset_lineage_is_35_40_45_50(t36_case):
    assert t36_case["report"]["normal_loop"]["dataset_row_counts"] == [35, 40, 45, 50]


def test_bo_never_selects_ood_and_model_never_auto_activates(t36_case):
    normal = t36_case["report"]["normal_loop"]
    assert normal["bo_transition_count"] == 2
    assert normal["bo_ood_selected"] == 0
    assert normal["automatic_model_activation_count"] == 0


def test_crash_resume_is_at_40_percent_and_finishes_round(t36_case):
    recovery = t36_case["report"]["crash_resume"]
    assert recovery["completed_before_restart"] == 2
    assert recovery["pending_before_restart"] == 3
    assert recovery["crash_progress_percent"] == 40.0
    assert recovery["completed_after_resume"] == 5
    assert recovery["dataset_rows_after_resume"] == 45
    assert recovery["replay_idempotent"] is True


def test_cancel_job_stops_automatic_continuation(t36_case):
    cancel = t36_case["report"]["operator_override"]["cancel_job"]
    assert cancel["status"] == "CANCELLED"
    assert cancel["automatic_continuation"] is False


def test_abort_round_cancels_without_next_round(t36_case):
    abort = t36_case["report"]["operator_override"]["abort_round"]
    assert abort["round_status"] == "CANCELLED"
    assert abort["next_round_created"] is False


def test_operator_resume_cannot_bypass_safety(t36_case):
    operator = t36_case["report"]["operator_override"]
    assert operator["safety_resume_blocked"] is True
    assert operator["safety_resume_error_code"] == "OPERATOR_OVERRIDE_BLOCKED"


def test_simulator_boundary_remains_explicit(t36_case):
    boundary = t36_case["report"]["boundary"]
    assert boundary["simulator_only"] is True
    assert boundary["real_device_connected"] is False
    assert boundary["real_device_validation_completed"] is False
    assert boundary["simulator_results_are_not_real_material_measurements"] is True


def test_final_report_has_sha256_and_persisted_file(t36_case):
    report = t36_case["report"]
    assert len(report["report_sha256"]) == 64
    assert t36_case["report_path"].exists()


def test_runner_explicitly_reports_final_pass(t36_case):
    assert "V0.3-T36 END-TO-END AUTONOMOUS VALIDATION PASS" in t36_case["stdout"]
