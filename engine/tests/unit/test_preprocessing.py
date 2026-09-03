from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.contracts import GateDecision, TestConsistencySpec
from engine.dataset.preprocessing import run_dataset_preprocessing


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    target = rng.normal(20, 5, size=20)
    dataframe = pd.DataFrame({
        "record_id": [f"item_{index:02d}" for index in range(20)],
        "feature_numeric": np.linspace(0, 1, 20),
        "feature_categorical": ["category_a"] * 20,
        "feature_high_missing": np.nan,
        "target_001": target,
    })
    dataframe.loc[2, "feature_numeric"] = np.nan
    dataframe.loc[3, "feature_categorical"] = None
    dataframe.loc[4, "target_001"] = np.nan
    dataframe["post_experiment_result"] = dataframe["target_001"]
    return pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)


class PreprocessingTests(unittest.TestCase):
    def test_default_strategy_produces_one_clean_dataset_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _frame()
            result = run_dataset_preprocessing(
                source,
                metadata={"target_fields": ["target_001"]},
                source_uri="test-source",
                output_dir=temporary,
            )
            self.assertEqual(result.final_gate.decision, GateDecision.passed)
            self.assertIsNotNone(result.artifact)
            artifact_dir = Path(result.artifact.artifact_dir)
            self.assertTrue((artifact_dir / "dataset.parquet").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())
            self.assertTrue((artifact_dir / "lineage.json").exists())
            self.assertIn("feature_high_missing", result.execution_report.removed_fields)
            self.assertIn("post_experiment_result", result.execution_report.removed_fields)
            self.assertIn("feature_numeric_was_missing", result.execution_report.added_fields)
            self.assertEqual(result.execution_report.dropped_duplicate_count, 1)
            self.assertEqual(result.execution_report.dropped_missing_target_count, 1)
            self.assertEqual(
                result.resolved_config.cleaning_config.profile_name,
                "default_safe_v1",
            )
            operation_names = {
                record.operation for record in result.cleaning_operations
            }
            self.assertIn("drop_fields", operation_names)
            self.assertIn("drop_exact_duplicates", operation_names)
            self.assertIn("drop_missing_target_rows", operation_names)
            self.assertIn("add_missing_indicators", operation_names)
            self.assertIn("impute_numeric_median", operation_names)
            self.assertIn(
                "cleaning_operation_records",
                result.to_dict(),
            )

    def test_user_can_disable_missing_indicator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_dataset_preprocessing(
                _frame(),
                user_config={
                    "target_fields": ["target_001"],
                    "cleaning": {"add_missing_indicators": False},
                },
                source_uri="test-source",
                output_dir=temporary,
            )
            self.assertEqual(result.execution_report.added_fields, [])

    def test_consistent_test_mismatch_blocks_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataframe = _frame()
            dataframe["test_name_001"] = "unexpected_method"
            result = run_dataset_preprocessing(
                dataframe,
                user_config={
                    "target_fields": ["target_001"],
                    "identifier_fields": ["record_id"],
                    "consistency_specs": [
                        {
                            "target_field": "target_001",
                            "test_field": "test_name_001",
                            "expected_test": "expected_method",
                        }
                    ],
                },
                source_uri="test-source",
                output_dir=temporary,
            )
            self.assertEqual(result.final_gate.decision, GateDecision.failed)
            self.assertIsNone(result.artifact)
            self.assertIn("test_consistency", result.final_gate.blocking_items)


if __name__ == "__main__":
    unittest.main()
