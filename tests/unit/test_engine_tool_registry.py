from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.engine_tool_registration import register_engine_tools
from agent.tool_registry import ToolRegistry


class EngineToolRegistryTests(unittest.TestCase):
    def test_engine_tools_are_registered_and_json_dispatchable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = ToolRegistry()
            register_engine_tools(registry)

            names = {item["name"] for item in registry.list_tools()}
            self.assertTrue({
                "preprocess_dataset",
                "train_model",
                "predict_model",
                "optimize_formula",
                "recommend_next_experiments",
                "list_artifacts",
                "get_chart_data",
            }.issubset(names))

            result = registry.execute(
                "list_artifacts",
                payload={
                    "dataset_roots": [str(root / "missing-datasets")],
                    "model_registry_paths": [
                        str(root / "missing-registry.json")
                    ],
                },
            )
            self.assertEqual(result["status"], "OK")
            self.assertEqual(result["result"], {"datasets": [], "models": []})


if __name__ == "__main__":
    unittest.main()
