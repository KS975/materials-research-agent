from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from engine.contracts import GateDecision, QualityThresholdConfig
from engine.contracts import ClosureConfig, LeakageConfig, TestConsistencySpec
from engine.governance.gate import apply_modeling_gate
from engine.governance.quality import run_quality_checks


def _base_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "identifier_001": [f"id_{index:03d}" for index in range(30)],
        "feature_001": np.linspace(0, 1, 30),
        "feature_002": np.zeros(30),
        "target_001": np.full(30, 20.0),
    })


class GovernanceTests(unittest.TestCase):
    def test_valid_dataset_passes_gate(self) -> None:
        report = run_quality_checks(
            _base_frame(),
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
                min_feature_count=2,
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.passed)
        self.assertEqual(gate.recommended_tier, 3)

    def test_target_missing_fails_gate(self) -> None:
        dataframe = _base_frame()
        dataframe.loc[dataframe.index[:10], "target_001"] = np.nan
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
                max_target_missing_ratio=0.1,
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.failed)
        self.assertIn("target_missing", gate.blocking_items)

    def test_target_conflict_fails_gate(self) -> None:
        dataframe = _base_frame()
        dataframe.loc[1, "target_001"] = dataframe.loc[0, "target_001"]
        dataframe.loc[1, "identifier_001"] = dataframe.loc[0, "identifier_001"]
        dataframe.loc[1, "target_001"] += 100
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.failed)
        self.assertIn("target_conflict", gate.blocking_items)

    def test_outlier_returns_conditional_gate(self) -> None:
        dataframe = _base_frame()
        dataframe.loc[0, "feature_002"] = 1000
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
                max_single_feature_outlier_ratio=0.0,
                max_global_outlier_ratio=0.2,
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.conditional)
        self.assertIn("feature_outlier", gate.warning_items)
        outlier_finding = next(
            item for item in report.findings if item.check == "feature_outlier"
        )
        self.assertEqual(outlier_finding.suggested_action, "delete_rows")
        self.assertFalse(outlier_finding.executed)
        self.assertTrue(outlier_finding.affected_rows)
        self.assertEqual(
            report.details["outliers"]["suggested_action"],
            "delete_rows",
        )
        self.assertFalse(report.details["outliers"]["executed"])
        self.assertIn("technical_summary", report.to_dict())
        self.assertEqual(
            {record.rule_name for record in report.rule_records},
            {
                "schema_validation",
                "sample_and_dimension_gate",
                "missing_value_detection",
                "duplicate_and_conflict_detection",
                "iqr_outlier_detection",
                "sample_closure_validation",
                "test_consistency_validation",
                "target_leakage_detection",
            },
        )
        self.assertTrue(any(
            "建议删除相关样本" in item["message"]
            and "未执行删除" in item["message"]
            for item in report.technical_summary()
        ))

    def test_missing_closure_field_fails_gate(self) -> None:
        dataframe = _base_frame()
        dataframe.loc[dataframe.index[:5], "feature_001"] = np.nan
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
            ),
            closure_config=ClosureConfig(
                identifier_fields=["identifier_001"],
                required_fields=["feature_001"],
                min_closure_ratio=0.99,
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.failed)
        self.assertIn("sample_closure", gate.blocking_items)
        self.assertEqual(
            report.details["sample_closure"]["closed_count"],
            25,
        )

    def test_test_consistency_mismatch_fails_gate(self) -> None:
        dataframe = _base_frame()
        dataframe["test_name_001"] = "unexpected_method"
        dataframe["unit_001"] = "unexpected_unit"
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
            ),
            consistency_specs=[
                TestConsistencySpec(
                    target_field="target_001",
                    test_field="test_name_001",
                    expected_test="expected_method",
                    unit_field="unit_001",
                    expected_unit="expected_unit",
                )
            ],
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.failed)
        self.assertIn("test_consistency", gate.blocking_items)
        self.assertEqual(
            report.details["test_consistency"]["specs"][0]["mismatches"]["test"],
            30,
        )

    def test_explicit_leakage_field_fails_gate(self) -> None:
        dataframe = _base_frame()
        dataframe["post_experiment_result_001"] = np.linspace(100, 130, 30)
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            feature_fields=[
                "feature_001",
                "feature_002",
                "post_experiment_result_001",
            ],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
            ),
            leakage_config=LeakageConfig(
                post_experiment_fields=["post_experiment_result_001"],
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.failed)
        self.assertIn("explicit_leakage", gate.blocking_items)

    def test_sample_size_reduction_fails_gate(self) -> None:
        dataframe = _base_frame().iloc[:5].reset_index(drop=True)
        report = run_quality_checks(
            dataframe,
            target_fields=["target_001"],
            identifier_fields=["identifier_001"],
            thresholds=QualityThresholdConfig(
                min_total_samples=30,
                min_samples_per_target=30,
            ),
        )
        gate = apply_modeling_gate(report)
        self.assertEqual(gate.decision, GateDecision.failed)
        self.assertIn("sample_count", gate.blocking_items)


if __name__ == "__main__":
    unittest.main()
