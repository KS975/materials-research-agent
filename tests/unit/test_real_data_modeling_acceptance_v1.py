from __future__ import annotations

from pathlib import Path

import pytest

from company_data import resolve_company_data_runtime_root
from real_modeling import (
    RealDataModelingAcceptance,
    RealModelingAcceptanceError,
)


ROOT = resolve_company_data_runtime_root()


@pytest.fixture(scope="module")
def report():
    return RealDataModelingAcceptance(ROOT).run(
        product_name="PC/ABS FR303",
        target_metric="悬臂梁冲击强度",
        test_size=0.20,
        random_state=42,
    )


def test_source_is_real_company_data_only(report):
    source = report["source"]
    assert source["source_kind"] == "company_real_data"
    assert source["canonical_source"] == "海科数据整理/总库"
    assert source["company_dataset_id"] == "haike_784db6db8807"
    assert source["simulator_rows"] == 0


def test_exact_product_and_target(report):
    assert report["product_type"] == "PC/ABS FR303"
    assert report["target_metric"] == "悬臂梁冲击强度"
    assert report["source"]["source_rows"] == 83
    assert report["source"]["target_numeric_rows"] == 82


def test_holdout_is_strictly_disjoint(report):
    source = report["source"]
    assert source["train_rows"] + source["holdout_rows"] == 82
    assert source["train_rows"] == 65
    assert source["holdout_rows"] == 17
    assert source["split_overlap_count"] == 0
    assert set(source["train_sample_ids"]).isdisjoint(
        source["holdout_sample_ids"]
    )


def test_feature_selection_is_train_only(report):
    features = report["feature_selection"]
    assert features["selection_fit_scope"] == "TRAIN_ONLY"
    assert features["all_active_formula_features"] == 103
    assert features["selected_feature_count"] >= 2
    assert all(
        name.startswith("formula::")
        for name in features["selected_features"]
    )
    assert "NOT silently converted to zero".lower() in (
        features["missing_value_policy"].lower()
    )


def test_model_selection_is_cv_only(report):
    comparison = report["model_comparison"]
    assert comparison["selection_scope"] == "TRAIN_CV_ONLY"
    names = {
        item["model_name"]
        for item in comparison["leaderboard"]
    }
    assert {
        "DummyMedian",
        "Ridge",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
    } <= names
    assert comparison["selected_model"] != "DummyMedian"


def test_holdout_metrics_are_finite(report):
    metrics = report["holdout"]["selected_model_metrics"]
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert isinstance(metrics["r2"], float)


def test_ad_classifies_every_holdout_row(report):
    counts = report["holdout"]["domain_counts"]
    assert sum(counts.values()) == report["source"]["holdout_rows"]
    assert set(counts) == {
        "IN_DOMAIN",
        "BORDERLINE",
        "OUT_OF_DOMAIN",
    }


def test_target_regime_diagnostic_detects_current_scale_discontinuity(report):
    regime = report["data_quality_diagnostics"]["target_regime"]
    assert regime["suspected"] is True
    assert regime["largest_gap"] > 300
    assert regime["largest_gap_left"] == 111.0
    assert regime["largest_gap_right"] == 440.0


def test_units_are_not_invented(report):
    quality = report["data_quality_diagnostics"]
    assert quality["target_unit_metadata_available"] is False
    assert "TARGET_UNIT_METADATA_UNAVAILABLE" in report["review_reasons"]


def test_diagnostic_model_is_not_official(report):
    boundary = report["official_boundary"]
    assert boundary["formal_reality_core_closed_samples"] == 0
    assert boundary["process_parameter_rows"] == 0
    assert boundary["explicit_test_condition_rows"] == 0
    assert boundary["official_model_allowed"] is False
    assert boundary["bo_allowed"] is False
    assert boundary["autonomous_model_use_allowed"] is False
    assert boundary["model_registry_write"] is False


def test_current_real_data_acceptance_requires_review(report):
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["review_reasons"]


def test_artifacts_exist(report):
    for path in report["artifacts"].values():
        assert Path(path).exists()


def test_ambiguous_target_name_is_rejected():
    with pytest.raises(RealModelingAcceptanceError):
        RealDataModelingAcceptance(ROOT).run(
            product_name="PC/ABS FR303",
            target_metric="冲击强度",
        )
