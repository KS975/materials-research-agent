from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.dataset.test_data_factory import (
    PerturbationSpec,
    generate_perturbation,
    save_perturbation,
)


class TestDataFactoryTests(unittest.TestCase):
    def test_normal_jitter_is_deterministic_and_source_is_untouched(self) -> None:
        source = pd.DataFrame({
            "feature_001": np.linspace(0, 1, 30),
            "target_001": np.linspace(10, 40, 30),
        })
        source_before = source.copy(deep=True)
        spec = PerturbationSpec(
            kind="normal_jitter",
            target_fields=["target_001"],
            affected_ratio=0.2,
            magnitude=0.05,
            random_seed=42,
        )
        first = generate_perturbation(source, spec)
        second = generate_perturbation(source, spec)
        pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
        pd.testing.assert_frame_equal(source, source_before)

    def test_missing_perturbation_adds_target_na(self) -> None:
        source = pd.DataFrame({
            "feature_001": np.linspace(0, 1, 20),
            "target_001": np.linspace(5, 25, 20),
        })
        result = generate_perturbation(source, PerturbationSpec(
            kind="missing",
            target_fields=["target_001"],
            affected_ratio=0.2,
            random_seed=7,
        ))
        self.assertTrue(result.dataframe["target_001"].isna().any())

    def test_save_perturbation_writes_artifact_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pd.DataFrame({
                "feature_001": np.linspace(0, 1, 15),
                "target_001": np.linspace(1, 15, 15),
            })
            result = generate_perturbation(source, PerturbationSpec(
                kind="normal_jitter",
                target_fields=["target_001"],
            ))
            artifact_dir = save_perturbation(result, temporary)
            self.assertTrue((artifact_dir / "dataset.parquet").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())
            self.assertTrue((artifact_dir / "lineage.json").exists())


if __name__ == "__main__":
    unittest.main()
