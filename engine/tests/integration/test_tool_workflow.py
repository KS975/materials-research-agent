from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.tools.optimize_formula import run_tool as optimize_formula
from engine.tools.get_chart_data import run_tool as get_chart_data
from engine.tools.list_artifacts import run_tool as list_artifacts
from engine.tools.preprocess_dataset import run_tool as preprocess_dataset
from engine.tools.predict_model import run_tool as predict_model
from engine.tools.recommend_next_experiments import (
    run_tool as recommend_next_experiments,
)
from engine.tools.train_model import run_tool as train_model


class ToolWorkflowTests(unittest.TestCase):
    def test_json_tools_complete_preprocess_train_predict_optimize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x1 = np.linspace(0.05, 0.95, 80)
            x2 = np.sin(2 * np.pi * x1)
            x3 = np.cos(2 * np.pi * x1)
            frame = pd.DataFrame({
                "feature_001": x1,
                "feature_002": x2,
                "feature_003": x3,
                "target_001": 3 * x1 + 2 * x2 + 1,
            })
            source = root / "source.csv"
            frame.to_csv(source, index=False)

            preprocessed = preprocess_dataset({
                "input_uri": str(source),
                "config": {
                    "target_fields": ["target_001"],
                    "feature_fields": [
                        "feature_001", "feature_002", "feature_003"
                    ],
                },
                "output_dir": str(root / "datasets"),
                "result_mode": "summary",
            })
            self.assertEqual(preprocessed["status"], "OK", preprocessed)
            dataset_artifact = preprocessed["result"]["dataset_artifact"]
            self.assertEqual(
                preprocessed["result"]["final_gate"]["decision"], "PASS"
            )
            self.assertNotIn("initial_quality_report", preprocessed["result"])

            registry_path = root / "models" / "model-registry.json"
            trained = train_model({
                "dataset_artifact_uri": dataset_artifact["artifact_dir"],
                "config": {
                    "algorithms": ["linear_regression", "ridge"],
                    "cv_mode": "kfold",
                },
                "output_dir": str(root / "models"),
                "model_registry_path": str(registry_path),
            })
            self.assertEqual(trained["status"], "OK", trained)
            model_artifacts = trained["result"]["training_run"]["model_artifacts"]
            self.assertEqual(len(model_artifacts), 1)

            predicted = predict_model({
                "model_registry_path": str(registry_path),
                "model_selector": {"target_name": "target_001"},
                "inputs": [{
                    "feature_001": 0.4,
                    "feature_002": 0.2,
                    "feature_003": 0.1,
                }],
                "result_mode": "summary",
            })
            self.assertEqual(predicted["status"], "OK", predicted)
            self.assertEqual(predicted["result"]["prediction_count"], 1)
            self.assertNotIn("predictions", predicted["result"])
            self.assertTrue(set(
                predicted["result"]["applicability_domain_counts"]
            ).issubset({"IN_DOMAIN", "EDGE", "OUT_OF_DOMAIN"}))

            optimized = optimize_formula({
                "request": {
                    "request_id": "integration-tool-flow",
                    "objectives": [{
                        "target_name": "target_001",
                        "operator": "equal",
                        "value": 2.5,
                    }],
                    "variables": [
                        {"name": "feature_001", "lower": 0.0, "upper": 1.0},
                        {"name": "feature_002", "lower": -1.0, "upper": 1.0},
                        {"name": "feature_003", "lower": -1.0, "upper": 1.0},
                    ],
                    "model_registry_path": str(registry_path),
                    "top_n": 2,
                    "max_evaluations": 300,
                },
                "output_dir": str(root / "optimizations"),
            })
            self.assertEqual(optimized["status"], "OK", optimized)
            self.assertGreaterEqual(
                len(optimized["result"]["selected_candidates"]), 1
            )

            listed = list_artifacts({
                "dataset_roots": [str(root / "datasets")],
                "model_registry_paths": [str(registry_path)],
            })
            self.assertEqual(listed["status"], "OK", listed)
            self.assertEqual(len(listed["result"]["datasets"]), 1)
            self.assertEqual(len(listed["result"]["models"]), 1)

            report_uri = optimized["result"]["artifact_ids"]["optimization_result"]
            charts = get_chart_data({"input_uri": report_uri})
            self.assertEqual(charts["status"], "OK", charts)
            self.assertGreaterEqual(len(charts["result"]["datasets"]), 1)

            next_experiments = recommend_next_experiments({
                "request": {
                    "request_id": "integration-next-experiments",
                    "objectives": [{
                        "target_name": "target_001",
                        "operator": "greater_or_equal",
                        "value": 3.0,
                    }],
                    "variables": [
                        {"name": "feature_001", "lower": 0.0, "upper": 1.0},
                        {"name": "feature_002", "lower": -1.0, "upper": 1.0},
                        {"name": "feature_003", "lower": -1.0, "upper": 1.0},
                    ],
                    "model_registry_path": str(registry_path),
                    "historical_experiments": [{
                        "experiment_id": "experiment_001",
                        "values": {
                            "feature_001": 0.2,
                            "feature_002": 0.1,
                            "feature_003": 0.0,
                        },
                        "observed_values": {"target_001": 1.8},
                    }],
                    "top_n": 2,
                    "max_evaluations": 200,
                },
                "output_dir": str(root / "optimizations"),
            })
            self.assertEqual(next_experiments["status"], "OK", next_experiments)
            self.assertEqual(
                next_experiments["result"]["diagnostics"]["selected_strategy"],
                "cold_start_design",
            )


if __name__ == "__main__":
    unittest.main()
