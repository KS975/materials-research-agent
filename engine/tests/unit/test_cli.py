from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.cli import main


class CliTests(unittest.TestCase):
    def test_quality_command_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            output_path = root / "quality.json"
            pd.DataFrame({
                "feature_001": np.linspace(0, 1, 30),
                "target_001": np.linspace(10, 40, 30),
            }).to_parquet(input_path, index=False)
            exit_code = main([
                "quality",
                "--input", str(input_path),
                "--target-field", "target_001",
                "--output", str(output_path),
            ])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertIn("quality_report", payload)
            self.assertIn("modeling_gate", payload)

    def test_test_data_command_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            output_dir = root / "artifacts"
            pd.DataFrame({
                "feature_001": np.linspace(0, 1, 20),
                "target_001": np.linspace(1, 20, 20),
            }).to_parquet(input_path, index=False)
            exit_code = main([
                "test-data",
                "--input", str(input_path),
                "--kind", "normal_jitter",
                "--target-field", "target_001",
                "--output-dir", str(output_dir),
                "--seed", "42",
            ])
            artifact_files = list(output_dir.rglob("metadata.json"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(artifact_files)

    def test_constraints_command_writes_utf8_json(self) -> None:
        from engine.tests.unit.test_constraints_reader import (
            _write_constraints_xlsx,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "constraints.xlsx"
            output = root / "constraints.json"
            _write_constraints_xlsx(source)
            exit_code = main([
                "constraints",
                "--input", str(source),
                "--output", str(output),
            ])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                payload["source_headers"]["parameter_name"], "参数名称"
            )


if __name__ == "__main__":
    unittest.main()
