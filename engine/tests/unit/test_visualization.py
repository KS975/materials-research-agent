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
from engine.modeling.trainer import train_models
from engine.tests.render_chart_data import render_bundle
from engine.visualization import build_visualization_bundle


def _frame() -> pd.DataFrame:
    x1 = np.linspace(0.05, 0.95, 24)
    x2 = np.sin(np.linspace(0, np.pi, 24))
    x3 = np.cos(np.linspace(0, np.pi, 24))
    return pd.DataFrame({
        "feature_001": x1,
        "feature_002": x2,
        "feature_003": x3,
        "target_001": 2 * x1 + 3 * x2 + 1,
    })


class VisualizationTests(unittest.TestCase):
    def test_training_run_exposes_evaluation_and_modeling_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = train_models(
                _frame(),
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002", "feature_003"],
                    algorithms=["linear_regression"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            artifact = result.model_artifacts[0]
            self.assertGreaterEqual(len(artifact.evaluation_records), 2)
            evaluation_path = Path(artifact.artifact_dir) / "evaluation.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.assertEqual(len(evaluation), len(artifact.evaluation_records))
            self.assertEqual(evaluation[0]["model_id"], artifact.model_id)

            payload = {"training_run": result.to_dict()}
            bundle = build_visualization_bundle(
                payload, source_kind="training", source_uri="memory://training"
            )
            dataset_ids = {item.dataset_id for item in bundle.datasets}
            self.assertIn("model_selected_metrics", dataset_ids)
            self.assertIn("model_target_001_predicted_vs_actual", dataset_ids)
            self.assertIn("model_target_001_residuals", dataset_ids)
            for dataset in bundle.datasets:
                rendered = dataset.to_dict()
                self.assertEqual(rendered["record_type"], "visualization_dataset")

    def test_prediction_report_builds_prediction_datasets(self) -> None:
        payload = {
            "model": {
                "model_id": "model_test",
                "version": "v001",
                "target_name": "target_001",
                "algorithm": "ridge",
            },
            "predictions": [
                {
                    "target_name": "target_001",
                    "predicted_value": 10.0,
                    "prediction_uncertainty": 1.0,
                    "applicability_domain": "IN_DOMAIN",
                    "warnings": [],
                },
                {
                    "target_name": "target_001",
                    "predicted_value": 20.0,
                    "prediction_uncertainty": 2.0,
                    "applicability_domain": "OUT_OF_DOMAIN",
                    "warnings": ["applicability domain is OUT_OF_DOMAIN"],
                },
            ],
        }
        bundle = build_visualization_bundle(
            payload, source_kind="prediction", source_uri="memory://prediction"
        )
        by_id = {item.dataset_id: item for item in bundle.datasets}
        self.assertIn("prediction_values", by_id)
        self.assertIn("prediction_detail", by_id)
        self.assertEqual(
            by_id["prediction_applicability_domain"].records,
            [
                {"applicability_domain": "IN_DOMAIN", "count": 1},
                {"applicability_domain": "OUT_OF_DOMAIN", "count": 1},
            ],
        )

    def test_preprocessing_report_builds_table_and_chart_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_dataset_preprocessing(
                _frame(),
                metadata={"target_fields": ["target_001"]},
                source_uri="memory://preprocessing",
                output_dir=root / "datasets",
            )
            payload = result.to_dict()
            bundle = build_visualization_bundle(
                payload,
                source_kind="preprocessing",
                source_uri="memory://preprocessing",
            )
            dataset_ids = {item.dataset_id for item in bundle.datasets}
            self.assertIn("preprocessing_before_after", dataset_ids)
            self.assertIn("preprocessing_modeling_gate", dataset_ids)
            self.assertIn("preprocessing_quality_findings", dataset_ids)

    def test_chart_data_cli_and_test_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "prediction-report.json"
            chart_data_path = root / "chart-data.json"
            output_dir = root / "charts"
            report_path.write_text(
                json.dumps({
                    "model": {
                        "model_id": "model_test",
                        "version": "v001",
                        "target_name": "target_001",
                        "algorithm": "ridge",
                    },
                    "predictions": [
                        {
                            "target_name": "target_001",
                            "predicted_value": float(value),
                            "prediction_uncertainty": 1.0,
                            "applicability_domain": "IN_DOMAIN",
                            "warnings": [],
                        }
                        for value in range(6)
                    ],
                }),
                encoding="utf-8",
            )
            self.assertEqual(main([
                "chart-data",
                "--input", str(report_path),
                "--source-kind", "prediction",
                "--output", str(chart_data_path),
            ]), 0)
            rendered = render_bundle(chart_data_path, output_dir)
            self.assertGreaterEqual(len(rendered), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in rendered))
            self.assertTrue((output_dir / "manifest.json").is_file())

    def test_formal_visualization_code_has_no_plotting_dependency(self) -> None:
        production_files = list(Path("engine/visualization").rglob("*.py"))
        production_files.extend([
            Path("engine/contracts.py"),
            Path("engine/cli.py"),
        ])
        for source_path in production_files:
            source = source_path.read_text(encoding="utf-8")
            self.assertNotIn("matplotlib", source)
            self.assertNotIn("pyecharts", source)
            self.assertNotIn("echarts", source)


if __name__ == "__main__":
    unittest.main()
