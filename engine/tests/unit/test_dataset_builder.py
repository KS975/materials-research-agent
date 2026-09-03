from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.contracts import CleaningConfig, QualityThresholdConfig
from engine.dataset.builder import build_dataset
from engine.exceptions import ValidationError
from engine.governance.gate import apply_modeling_gate
from engine.governance.quality import run_quality_checks


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "identifier_001": [f"item_{index}" for index in range(30)],
        "feature_001": np.linspace(0, 1, 30),
        "feature_002": rng.normal(size=30),
        "target_001": rng.normal(20, 5, size=30),
    })


class DatasetBuilderTests(unittest.TestCase):
    def test_build_dataset_writes_artifact_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataframe = _frame()
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
            artifact = build_dataset(
                dataframe,
                target_fields=["target_001"],
                identifier_fields=["identifier_001"],
                quality_report=report,
                gate_result=gate,
                cleaning_config=CleaningConfig(drop_fields=["feature_002"]),
                source_uri="test-source",
                output_dir=temporary,
            )
            artifact_dir = Path(artifact.artifact_dir)
            self.assertTrue((artifact_dir / "dataset.parquet").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())
            self.assertTrue((artifact_dir / "lineage.json").exists())
            self.assertNotIn("feature_002", artifact.feature_fields)
            lineage = __import__("json").loads(
                (artifact_dir / "lineage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                lineage["cleaning_summary"]["removed_fields"],
                ["feature_002"],
            )

    def test_fail_gate_blocks_dataset_build(self) -> None:
        dataframe = _frame().iloc[:5].reset_index(drop=True)
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
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValidationError):
                build_dataset(
                    dataframe,
                    target_fields=["target_001"],
                    identifier_fields=["identifier_001"],
                    quality_report=report,
                    gate_result=gate,
                    output_dir=temporary,
                )


if __name__ == "__main__":
    unittest.main()
