from __future__ import annotations

import unittest

from jsonschema import Draft202012Validator

from engine.config import EnginePathConfig
from engine.tools import TOOL_FUNCTIONS, TOOL_SPECS, run_tool_by_name


class ToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_json_tools(self) -> None:
        expected = {
            "preprocess_dataset",
            "train_model",
            "predict_model",
            "optimize_formula",
            "recommend_next_experiments",
            "list_artifacts",
            "get_chart_data",
        }
        self.assertEqual({item["name"] for item in TOOL_SPECS}, expected)
        self.assertEqual(set(TOOL_FUNCTIONS), expected)

    def test_unknown_tool_returns_structured_error(self) -> None:
        result = run_tool_by_name("not_a_tool", {})
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error"]["code"], "UNKNOWN_TOOL")
        self.assertIn("preprocess_dataset", result["details"]["available_tools"])

    def test_artifact_paths_are_resolved_from_one_root(self) -> None:
        config = EnginePathConfig(artifact_root="custom/artifacts")
        self.assertEqual(config.dataset_dir.as_posix(), "custom/artifacts/datasets")
        self.assertEqual(config.model_dir.as_posix(), "custom/artifacts/models")
        self.assertEqual(
            config.optimization_dir.as_posix(), "custom/artifacts/optimizations"
        )
        self.assertEqual(
            config.model_registry_path.as_posix(),
            "custom/artifacts/models/model-registry.json",
        )

    def test_tool_input_schemas_are_valid_json_schema(self) -> None:
        for spec in TOOL_SPECS:
            with self.subTest(tool=spec["name"]):
                Draft202012Validator.check_schema(spec["input_schema"])


if __name__ == "__main__":
    unittest.main()
