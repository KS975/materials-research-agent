import json
from pathlib import Path

from runtime.v013_reports import load_v013_status


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_v013_status_aggregates_gate_cv_and_latest_ad(tmp_path):
    root = tmp_path / "v013"
    stem = "project_115_冲击强度"
    _write(root / "gates" / f"{stem}_modeling_gate.json", {"decision": "FAIL"})
    _write(root / "cross_validation" / stem / "cv_report.json", {"best_cv_model": {"model_name": "ExtraTreesRegressor"}})
    _write(root / "applicability_domain" / stem / "sample_ad_report.json", {"applicability_domain": {"status": "OUT_OF_DOMAIN"}})

    status = load_v013_status(root, 115, "冲击强度")

    assert status["kind"] == "v013_modeling_status"
    assert status["gate"]["decision"] == "FAIL"
    assert status["cross_validation"]["best_cv_model"]["model_name"] == "ExtraTreesRegressor"
    assert status["latest_applicability_domain"]["applicability_domain"]["status"] == "OUT_OF_DOMAIN"
    assert status["availability"]["gate"] is True
    assert status["availability"]["cross_validation"] is True


def test_v013_status_allows_partial_reports(tmp_path):
    status = load_v013_status(tmp_path / "v013", 115, "冲击强度")
    assert status["gate"] is None
    assert status["cross_validation"] is None
    assert status["latest_applicability_domain"] is None
    assert status["availability"]["gate"] is False
