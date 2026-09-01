from pathlib import Path

from demo import MondayDemoService
from runtime.monday_demo_ui import build_monday_demo_overview


def test_demo_status_contains_all_versions():
    report = MondayDemoService().status()
    assert [x["version"] for x in report["versions"]] == [
        "V0.1.1",
        "V0.1.2",
        "V0.1.3",
        "V0.1.4",
        "V0.2",
        "V0.3",
    ]


def test_internal_demo_versions_are_ready_after_prepare():
    report = MondayDemoService().status()
    by_version = {
        x["version"]: x for x in report["versions"]
    }
    for version in ("V0.1.3", "V0.1.4", "V0.2", "V0.3"):
        assert by_version[version]["status"] == "READY"
    assert report["status"] == "READY"
    assert report["prepared_internal_versions"] == "4/4"


def test_v013_demo_has_gate_cv_and_ad():
    report = MondayDemoService().status()
    item = next(x for x in report["versions"] if x["version"] == "V0.1.3")
    assert item["project_id"] == 9010
    assert item["summary"]["gate_decision"] == "PASS"
    assert item["summary"]["official_model_allowed"] is True
    assert item["summary"]["best_cv_model"]
    assert {"IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN"} <= set(
        item["summary"]["ad_statuses"]
    )


def test_v014_demo_has_five_bo_experiments():
    report = MondayDemoService().status()
    item = next(x for x in report["versions"] if x["version"] == "V0.1.4")
    assert item["project_id"] == 9018
    assert item["summary"]["next_experiment_count"] == 5
    assert item["summary"]["ood_selected"] == 0


def test_v020_demo_shows_automatic_learning_dataset_chain():
    report = MondayDemoService().status()
    item = next(x for x in report["versions"] if x["version"] == "V0.2")
    assert item["project_id"] == 9026
    assert item["summary"]["round_count"] == 3
    assert item["summary"]["total_experiments"] == 15
    assert item["summary"]["dataset_row_counts"] == [35, 40, 45, 50]
    assert item["summary"]["model_auto_activation"] is False


def test_v030_demo_is_9_of_9_and_simulator_only():
    report = MondayDemoService().status()
    item = next(x for x in report["versions"] if x["version"] == "V0.3")
    assert item["project_id"] == 9036
    assert item["summary"]["component_pass"] == "9/9"
    assert item["summary"]["round_count"] == 3
    assert item["summary"]["automatic_captures"] == 15
    assert item["summary"]["manual_submissions"] == 0
    assert item["summary"]["real_device_connected"] is False
    assert item["summary"]["safety_bypass_forbidden"] is True


def test_demo_scope_has_all_projects():
    report = MondayDemoService().status()
    assert report["demo_scope_project_ids"] == [
        115, 9010, 9018, 9026, 9036, 930066
    ]


def test_demo_ui_answer_is_ready():
    data = build_monday_demo_overview()
    assert data["kind"] == "monday_demo"
    assert data["status"] == "READY"
    assert "周一 Demo Runtime 已准备完成" in data["answer"]


def test_boundary_is_explicit():
    report = MondayDemoService().status()
    b = report["boundaries"]
    assert b["business_mysql_read_only"] is True
    assert b["v030_simulator_only"] is True
    assert b["real_device_connected"] is False
    assert b["automatic_model_promotion_forbidden"] is True
    assert b["operator_override_cannot_bypass_safety"] is True
