from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.cli import main
from engine.contracts import TrainingConfig
from engine.dataset.preprocessing import run_dataset_preprocessing
from engine.exceptions import ValidationError
from engine.modeling.predictor import predict_with_model
from engine.modeling.registry import load_model_bundle, load_registered_model
from engine.modeling.registry import select_registry_entry
from engine.modeling.strategy import select_modeling_strategy
from engine.modeling.trainer import train_models


def _config(**overrides) -> TrainingConfig:
    payload = {
        "target_names": ["target_001"],
        "feature_names": ["feature_001", "feature_002", "feature_003"],
        "algorithms": None,
    }
    payload.update(overrides)
    return TrainingConfig(**payload)


def _training_frame() -> pd.DataFrame:
    x1 = np.linspace(0.05, 0.95, 24)
    x2 = np.sin(np.linspace(0, np.pi, 24))
    x3 = np.cos(np.linspace(0, np.pi, 24))
    return pd.DataFrame({
        "feature_001": x1,
        "feature_002": x2,
        "feature_003": x3,
        "target_001": 2 * x1 + 3 * x2 + 1,
        "target_002": 4 * x1 - 2 * x3 + 2,
    })


class StrategyTests(unittest.TestCase):
    def test_tier_selection_is_data_size_driven(self) -> None:
        tier_1 = select_modeling_strategy(
            target_name="target_001", sample_count=2, feature_count=3, config=_config()
        )
        tier_2 = select_modeling_strategy(
            target_name="target_001", sample_count=7, feature_count=3, config=_config()
        )
        tier_3 = select_modeling_strategy(
            target_name="target_001", sample_count=12, feature_count=3, config=_config()
        )
        self.assertEqual(tier_1.tier, 1)
        self.assertEqual(tier_1.cv_mode, "loocv")
        self.assertIn("linear_regression", tier_1.selected_algorithms)
        self.assertNotIn("xgboost", tier_1.selected_algorithms)
        self.assertEqual(tier_2.tier, 2)
        self.assertEqual(tier_2.cv_mode, "kfold")
        self.assertIn("xgboost", tier_2.selected_algorithms)
        self.assertEqual(tier_3.tier, 3)
        self.assertIn("svr", tier_3.selected_algorithms)

    def test_user_algorithm_override_and_filtering(self) -> None:
        overridden = select_modeling_strategy(
            target_name="target_001",
            sample_count=12,
            feature_count=3,
            config=_config(algorithms=["ridge"]),
        )
        self.assertEqual(overridden.selected_algorithms, ["ridge"])
        self.assertIn("user explicitly supplied", overridden.strategy_reason)

        filtered = select_modeling_strategy(
            target_name="target_001",
            sample_count=12,
            feature_count=3,
            config=_config(disabled_algorithms=["linear_regression"]),
        )
        self.assertNotIn("linear_regression", filtered.selected_algorithms)

    def test_unknown_algorithm_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            select_modeling_strategy(
                target_name="target_001",
                sample_count=12,
                feature_count=3,
                config=_config(algorithms=["unsupported_model"]),
            )


class TrainerTests(unittest.TestCase):
    def test_targets_are_trained_and_saved_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = train_models(
                _training_frame(),
                TrainingConfig(
                    target_names=["target_001", "target_002"],
                    feature_names=["feature_001", "feature_002", "feature_003"],
                    algorithms=["linear_regression", "ridge"],
                    cv_mode="kfold",
                ),
                dataset_artifact_id="dataset_test",
                output_dir=root / "models",
            )
            self.assertEqual(
                [item.target_name for item in result.strategies],
                ["target_001", "target_002"],
            )
            self.assertEqual(
                {item.target_name for item in result.model_artifacts},
                {"target_001", "target_002"},
            )
            for artifact in result.model_artifacts:
                metadata = json.loads(
                    (Path(artifact.artifact_dir) / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                schema = json.loads(
                    (Path(artifact.artifact_dir) / "feature_schema.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(metadata["target_name"], artifact.target_name)
                self.assertEqual(schema["target"], artifact.target_name)
                self.assertEqual(metadata["model_id"], artifact.model_id)
                self.assertIn("interpretability", metadata)
                self.assertIn("training_warnings", metadata)

            self.assertEqual(len(result.candidate_records), 4)
            self.assertEqual(
                len([item for item in result.candidate_records if item.selection_rank]),
                4,
            )

    def test_repeat_training_creates_new_version_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = TrainingConfig(
                target_names=["target_001"],
                feature_names=["feature_001", "feature_002", "feature_003"],
                algorithms=["ridge"],
                cv_mode="kfold",
            )
            first = train_models(
                _training_frame(), config, output_dir=root / "models"
            )
            second = train_models(
                _training_frame(), config, output_dir=root / "models"
            )
            registry = json.loads(
                (root / "models" / "model-registry.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first.model_artifacts[0].version, "v001")
            self.assertEqual(second.model_artifacts[0].version, "v002")
            self.assertEqual(len(registry["models"]), 2)
            self.assertTrue(
                (Path(first.model_artifacts[0].artifact_dir) / "model.joblib").is_file()
            )
            self.assertTrue(
                (Path(second.model_artifacts[0].artifact_dir) / "model.joblib").is_file()
            )


class PredictionTests(unittest.TestCase):
    def test_target_selector_prefers_latest_registered_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry_path = Path(temporary) / "model-registry.json"
            registry_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "models": [
                        {
                            "model_id": "model_a_target_legacy",
                            "version": "v001",
                            "target_name": "target_001",
                            "dataset_artifact_id": "dataset_a",
                            "created_at": "2026-01-01T00:00:00+00:00",
                        },
                        {
                            "model_id": "model_b_target_latest",
                            "version": "v001",
                            "target_name": "target_001",
                            "dataset_artifact_id": "dataset_b",
                            "created_at": "2026-01-02T00:00:00+00:00",
                        },
                    ],
                }),
                encoding="utf-8",
            )
            entry = select_registry_entry(registry_path, target_name="target_001")
            self.assertEqual(entry["model_id"], "model_b_target_latest")
            dataset_entry = select_registry_entry(
                registry_path,
                target_name="target_001",
                dataset_artifact_id="dataset_a",
            )
            self.assertEqual(dataset_entry["model_id"], "model_a_target_legacy")

    def test_registered_model_predicts_with_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = _training_frame()
            result = train_models(
                frame,
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002", "feature_003"],
                    algorithms=["ridge"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            artifact = result.model_artifacts[0]
            bundle = load_model_bundle(artifact.file_path)
            in_domain = predict_with_model(bundle, frame.head(2))
            self.assertEqual(len(in_domain), 2)
            self.assertTrue(
                all(np.isfinite(item.predicted_value) for item in in_domain)
            )
            self.assertIn(
                in_domain[0].applicability_domain,
                {"IN_DOMAIN", "EDGE", "OUT_OF_DOMAIN"},
            )

            far_frame = pd.DataFrame({
                "feature_001": [100.0],
                "feature_002": [100.0],
                "feature_003": [100.0],
            })
            out_domain = predict_with_model(bundle, far_frame)
            self.assertEqual(out_domain[0].applicability_domain, "OUT_OF_DOMAIN")

            loaded = load_registered_model(
                root / "models" / "model-registry.json",
                target_name="target_001",
            )
            self.assertEqual(loaded.bundle["model_id"], artifact.model_id)
            self.assertEqual(loaded.bundle["version"], artifact.version)


class ModelingCliTests(unittest.TestCase):
    def test_dataset_artifact_train_and_predict_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = _training_frame()
            preprocessed = run_dataset_preprocessing(
                frame,
                metadata={"target_fields": ["target_001", "target_002"]},
                source_uri="memory://generic",
                output_dir=root / "datasets",
            )
            assert preprocessed.artifact is not None
            config_path = root / "model-config.json"
            config_path.write_text(
                json.dumps({
                    "algorithms": ["linear_regression"],
                    "feature_fields": [
                        "feature_001", "feature_002", "feature_003"
                    ],
                }),
                encoding="utf-8",
            )
            report_path = root / "training-report.json"
            self.assertEqual(
                main([
                    "train",
                    "--input", preprocessed.artifact.artifact_dir,
                    "--config", str(config_path),
                    "--output-dir", str(root / "models"),
                    "--output", str(report_path),
                ]),
                0,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(report["training_run"]["model_artifacts"]), 2
            )
            registry_path = root / "models" / "model-registry.json"
            prediction_path = root / "prediction.json"
            self.assertEqual(
                main([
                    "predict",
                    "--input", preprocessed.artifact.file_path,
                    "--registry", str(registry_path),
                    "--target-name", "target_001",
                    "--output", str(prediction_path),
                ]),
                0,
            )
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            self.assertEqual(prediction["model"]["target_name"], "target_001")
            self.assertEqual(len(prediction["predictions"]), len(frame))


if __name__ == "__main__":
    unittest.main()
