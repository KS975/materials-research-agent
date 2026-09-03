from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from engine.tools.get_chart_data import run_tool as get_chart_data
from engine.tools.list_artifacts import run_tool as list_artifacts
from engine.tools.optimize_formula import run_tool as optimize_formula
from engine.tools.predict_model import run_tool as predict_model
from engine.tools.preprocess_dataset import run_tool as preprocess_dataset
from engine.tools.train_model import run_tool as train_model


RAFM_SOURCE = Path("engine/artifacts/external/03_demo/RAFM-dataset.csv")
RAFM_CONSTRAINTS = Path(
    "engine/artifacts/external/03_demo/reports/constraints.json"
)


@unittest.skipUnless(
    RAFM_SOURCE.is_file() and RAFM_CONSTRAINTS.is_file(),
    "external RAFM acceptance fixture is not present",
)
class ExternalRafmToolTests(unittest.TestCase):
    def test_real_dataset_supports_full_tool_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.read_csv(RAFM_SOURCE)
            target_fields = ["UTS", "TE"]
            feature_fields = [
                column for column in frame.columns
                if column not in target_fields
            ]
            # Keep the first observation for each repeated input key so this
            # acceptance test exercises the formal gate without bypassing it.
            frame = frame.drop_duplicates(subset=feature_fields, keep="first")
            source = root / "rafm_unique_inputs.csv"
            frame.to_csv(source, index=False)

            preprocessed = preprocess_dataset({
                "input_uri": str(source),
                "config": {
                    "target_fields": target_fields,
                    "feature_fields": feature_fields,
                },
                "output_dir": str(root / "datasets"),
                "result_mode": "summary",
            })
            self.assertEqual(preprocessed["status"], "OK", preprocessed)
            self.assertIn(
                preprocessed["result"]["final_gate"]["decision"],
                {"PASS", "CONDITIONAL_PASS"},
            )

            registry_path = root / "models" / "model-registry.json"
            trained = train_model({
                "dataset_artifact_uri": (
                    preprocessed["result"]["dataset_artifact"]["artifact_dir"]
                ),
                "config": {
                    "algorithms": ["gradient_boosting"],
                    "cv_mode": "kfold",
                },
                "output_dir": str(root / "models"),
                "model_registry_path": str(registry_path),
                "result_mode": "summary",
            })
            self.assertEqual(trained["status"], "OK", trained)
            self.assertEqual(
                len(trained["result"]["training_run"]["model_artifacts"]), 2
            )

            predicted = predict_model({
                "model_registry_path": str(registry_path),
                "model_selector": {"target_name": "UTS"},
                "input_uri": str(source),
                "result_mode": "summary",
            })
            self.assertEqual(predicted["status"], "OK", predicted)
            self.assertEqual(predicted["result"]["prediction_count"], len(frame))

            constraints = json.loads(
                RAFM_CONSTRAINTS.read_text(encoding="utf-8")
            )
            variables = [
                {
                    "name": item["name"],
                    "lower": item["lower_bound"],
                    "upper": item["upper_bound"],
                }
                for item in constraints["variables"]
            ]
            for variable in variables:
                if variable["name"] == "Ttest":
                    variable["fixed_value"] = 25.0

            optimized = optimize_formula({
                "request": {
                    "request_id": "rafm-tool-acceptance",
                    "objectives": [
                        {
                            "target_name": "UTS",
                            "operator": "greater_or_equal",
                            "value": 650,
                            "requirement": "hard",
                        },
                        {
                            "target_name": "TE",
                            "operator": "greater_or_equal",
                            "value": 15,
                            "requirement": "hard",
                        },
                    ],
                    "variables": variables,
                    "model_registry_path": str(registry_path),
                    "top_n": 3,
                    "random_seed": 20260903,
                    "max_evaluations": 800,
                    "time_limit": 60,
                },
                "output_dir": str(root / "optimizations"),
                "result_mode": "summary",
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
            self.assertEqual(len(listed["result"]["models"]), 2)

            charts = get_chart_data({
                "input_uri": optimized["result"]["artifact_ids"][
                    "optimization_result"
                ],
            })
            self.assertEqual(charts["status"], "OK", charts)
            self.assertGreaterEqual(len(charts["result"]["datasets"]), 1)


if __name__ == "__main__":
    unittest.main()
