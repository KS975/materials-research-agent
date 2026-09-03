from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from engine.dataset.config_resolver import (
    MetadataRequiredError,
    resolve_preprocessing_config,
)


class ConfigResolverTests(unittest.TestCase):
    def test_system_metadata_and_inference_are_used_without_user_config(self) -> None:
        dataframe = pd.DataFrame({
            "record_id": [f"item_{index}" for index in range(20)],
            "feature_001": np.linspace(0, 1, 20),
            "target_001": np.random.default_rng(7).normal(size=20),
        })
        resolved = resolve_preprocessing_config(
            dataframe,
            metadata={"target_fields": ["target_001"]},
        )
        self.assertEqual(resolved.target_fields, ["target_001"])
        self.assertEqual(resolved.identifier_fields, ["record_id"])
        self.assertEqual(resolved.feature_fields, ["feature_001"])
        self.assertEqual(resolved.cleaning_config.profile_name, "default_safe_v1")
        self.assertIn(
            "identifier fields were inferred from uniqueness and type",
            resolved.resolution_reasons,
        )

    def test_user_overrides_take_precedence(self) -> None:
        dataframe = pd.DataFrame({
            "record_id": [str(index) for index in range(20)],
            "feature_001": np.linspace(0, 1, 20),
            "feature_002": np.linspace(1, 2, 20),
            "target_001": np.random.default_rng(9).normal(size=20),
        })
        resolved = resolve_preprocessing_config(
            dataframe,
            user_config={
                "target_fields": ["target_001"],
                "identifier_fields": ["record_id"],
                "thresholds": {"min_total_samples": 20},
                "cleaning": {
                    "drop_fields": ["feature_002"],
                    "outlier_strategy": "winsorize",
                    "impute_missing_features": False,
                },
            },
        )
        self.assertEqual(resolved.thresholds.min_total_samples, 20)
        self.assertEqual(resolved.cleaning_config.drop_fields, ["feature_002"])
        self.assertEqual(resolved.cleaning_config.outlier_strategy, "winsorize")
        self.assertFalse(resolved.cleaning_config.impute_missing_features)
        self.assertIn(
            "cleaning strategy was overridden by user configuration",
            resolved.resolution_reasons,
        )

    def test_missing_target_metadata_is_rejected(self) -> None:
        dataframe = pd.DataFrame({
            "feature_001": np.linspace(0, 1, 20),
        })
        with self.assertRaises(MetadataRequiredError):
            resolve_preprocessing_config(dataframe)


if __name__ == "__main__":
    unittest.main()
