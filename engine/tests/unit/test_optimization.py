from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.contracts import TrainingConfig
from engine.cli import main
from engine.exceptions import ValidationError
from engine.modeling.trainer import train_models
from engine.optimization.contracts import (
    HardConstraintSpec,
    ModelQualityGate,
    ObjectiveSpec,
    OptimizationRequest,
    StrategyThresholds,
    VariableSpec,
)
from engine.optimization.constraints import repair_and_report, variable_violation
from engine.optimization.models import ModelSet
from engine.optimization.search_space import SearchSpace
from engine.optimization.strategy_selector import select_strategy
from engine.optimization.service import (
    optimize_formula,
    optimize_next_experiments,
)


def _training_frame() -> pd.DataFrame:
    x1 = np.linspace(0.05, 0.95, 28)
    x2 = np.sin(np.linspace(0, np.pi, 28))
    x3 = np.cos(np.linspace(0, np.pi, 28))
    return pd.DataFrame({
        "feature_001": x1,
        "feature_002": x2,
        "feature_003": x3,
        "target_001": 2 * x1 + 3 * x2 + 1,
    })


def _mixed_frame() -> pd.DataFrame:
    x1 = np.linspace(0.05, 0.95, 36)
    x2 = np.tile(np.arange(6), 6)
    x3 = np.tile(np.arange(3), 12)
    return pd.DataFrame({
        "feature_001": x1,
        "feature_002": x2,
        "feature_003": x3,
        "target_001": 10 + 3 * x1 + x2 + 2 * x3,
    })


def _bo_frame() -> pd.DataFrame:
    x1 = np.linspace(0.05, 0.95, 40)
    x2 = np.linspace(0.10, 0.90, 40)
    return pd.DataFrame({
        "feature_001": x1,
        "feature_002": x2,
        "target_001": 3 * x1 + 2 * x2 + 1,
    })


def _variables() -> list[VariableSpec]:
    return [
        VariableSpec(name="feature_001", lower=0.0, upper=1.0),
        VariableSpec(name="feature_002", lower=0.0, upper=1.0),
        VariableSpec(name="feature_003", lower=0.0, upper=1.0),
    ]


def _space(variables: list[VariableSpec] | None = None) -> SearchSpace:
    variables = variables or _variables()
    return SearchSpace(
        variables=variables,
        defaults={item.name: 0.5 for item in variables},
        model_feature_names=[item.name for item in variables],
    )


class ContractTests(unittest.TestCase):
    def test_request_parses_nested_contracts(self) -> None:
        request = OptimizationRequest.from_dict({
            "objectives": [{
                "target_name": "target_001",
                "operator": "greater_or_equal",
                "value": 2.0,
                "requirement": "hard",
            }],
            "variables": [
                {"name": "feature_001", "type": "continuous", "lower": 0, "upper": 1},
                {
                    "name": "feature_002",
                    "type": "categorical",
                    "categories": ["A", "B"],
                },
            ],
            "hard_constraints": [{
                "name": "sum",
                "kind": "linear_sum",
                "variables": ["feature_001"],
                "constant": 0.5,
            }],
            "strategy_thresholds": {"candidate_rank_max_count": 100},
        })
        request.validate()
        self.assertEqual(
            request.strategy_thresholds.candidate_rank_max_count, 100
        )

    def test_conflicting_objectives_are_rejected(self) -> None:
        request = OptimizationRequest(
            objectives=[
                ObjectiveSpec(
                    target_name="target_001",
                    operator="greater_or_equal",
                    value=5,
                ),
                ObjectiveSpec(
                    target_name="target_001",
                    operator="less_or_equal",
                    value=4,
                ),
            ]
        )
        with self.assertRaises(ValidationError):
            request.validate()


class ConstraintTests(unittest.TestCase):
    def test_linear_sum_is_repaired(self) -> None:
        space = _space()
        constraint = HardConstraintSpec(
            name="sum",
            kind="linear_sum",
            variables=["feature_001", "feature_002", "feature_003"],
            coefficients=[1, 1, 1],
            constant=1.5,
        )
        repaired, reports, violation = repair_and_report(
            {"feature_001": 0.8, "feature_002": 0.8, "feature_003": 0.8},
            space,
            [constraint],
        )
        self.assertLessEqual(abs(sum(repaired.values()) - 1.5), 1e-8)
        self.assertLessEqual(violation, 1e-8)
        self.assertTrue(all(item["satisfied"] for item in reports))

    def test_mutex_keeps_one_active_variable(self) -> None:
        space = _space()
        constraint = HardConstraintSpec(
            name="mutex",
            kind="mutex",
            variables=["feature_001", "feature_002"],
        )
        repaired, reports, violation = repair_and_report(
            {"feature_001": 0.2, "feature_002": 0.8, "feature_003": 0.1},
            space,
            [constraint],
        )
        active_count = sum(
            abs(float(repaired[name])) > 1e-12
            for name in ["feature_001", "feature_002"]
        )
        self.assertEqual(active_count, 1)
        self.assertLessEqual(violation, 1e-12)
        self.assertTrue(reports[0]["satisfied"])
        self.assertEqual(variable_violation(repaired, space), 0)


class StrategySelectionTests(unittest.TestCase):
    def test_low_dimensional_grid_uses_candidate_rank(self) -> None:
        request = OptimizationRequest(
            objectives=[
                ObjectiveSpec(target_name="target_001", operator="equal", value=1)
            ],
            variables=_variables(),
        )
        request.validate()
        strategy, _, _ = select_strategy(
            request, space=_space(), model_set=ModelSet(models={})
        )
        self.assertEqual(strategy, "candidate_rank")

    def test_high_dimensional_small_sample_uses_active_set_de(self) -> None:
        variables = [
            VariableSpec(name=f"feature_{index:03d}", lower=0, upper=1)
            for index in range(25)
        ]
        request = OptimizationRequest(
            objectives=[
                ObjectiveSpec(target_name="target_001", operator="equal", value=1)
            ],
            variables=variables,
        )
        request.validate()
        strategy, _, _ = select_strategy(
            request,
            space=_space(variables),
            model_set=ModelSet(models={}),
        )
        self.assertEqual(strategy, "active_set_de")


class FormulaOptimizationTests(unittest.TestCase):
    def test_registered_model_optimizes_with_constraints_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trained = train_models(
                _training_frame(),
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002", "feature_003"],
                    algorithms=["ridge"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            request = OptimizationRequest(
                request_id="formula-candidate-rank",
                objectives=[
                    ObjectiveSpec(
                        target_name="target_001", operator="equal", value=2.5
                    )
                ],
                variables=_variables(),
                hard_constraints=[
                    HardConstraintSpec(
                        name="component_sum",
                        kind="linear_sum",
                        variables=["feature_001", "feature_002", "feature_003"],
                        coefficients=[1, 1, 1],
                        upper=2.0,
                    )
                ],
                top_n=3,
            )
            request.model_registry_path = str(
                root / "models" / "model-registry.json"
            )
            result = optimize_formula(request, output_dir=root / "optimizations")
            self.assertEqual(result.status, "COMPLETE")
            self.assertGreaterEqual(len(result.selected_candidates), 1)
            for candidate in result.selected_candidates:
                total = sum(float(candidate.values[name]) for name in [
                    "feature_001", "feature_002", "feature_003"
                ])
                self.assertLessEqual(total, 2.0 + 1e-8)
                self.assertNotEqual(candidate.applicability_domain, "OUT_OF_DOMAIN")
                self.assertTrue(candidate.model_refs)
            artifact_dir = Path(result.artifact_ids["artifact_dir"])
            self.assertTrue((artifact_dir / "optimization_result.json").is_file())
            self.assertTrue((artifact_dir / "selected_candidates.csv").is_file())
            visualizations = json.loads(
                (artifact_dir / "visualization_datasets.json").read_text(
                    encoding="utf-8"
                )
            )
            dataset_ids = {item["dataset_id"] for item in visualizations}
            self.assertIn("selected_candidates", dataset_ids)
            self.assertIn("constraint_report", dataset_ids)
            chart_path = root / "optimization-chart-data.json"
            self.assertEqual(
                main([
                    "chart-data",
                    "--input", str(artifact_dir / "optimization_result.json"),
                    "--source-kind", "optimization",
                    "--output", str(chart_path),
                ]),
                0,
            )
            chart_payload = json.loads(chart_path.read_text(encoding="utf-8"))
            self.assertEqual(chart_payload["source_kind"], "optimization")
            self.assertGreaterEqual(len(chart_payload["datasets"]), 1)

    def test_model_quality_gate_can_block_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_models(
                _training_frame(),
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002", "feature_003"],
                    algorithms=["ridge"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            request = OptimizationRequest(
                objectives=[
                    ObjectiveSpec(
                        target_name="target_001", operator="equal", value=2.5
                    )
                ],
                variables=_variables(),
                model_quality_gate=ModelQualityGate(
                    mode="block", max_cv_rmse=1e-9
                ),
            )
            request.model_registry_path = str(
                root / "models" / "model-registry.json"
            )
            with self.assertRaises(ValidationError):
                optimize_formula(request, output_dir=None)

    def test_mixed_nsga2_preserves_integer_and_categorical_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_models(
                _mixed_frame(),
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002", "feature_003"],
                    algorithms=["ridge"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            request = OptimizationRequest(
                request_id="formula-mixed-nsga2",
                objectives=[
                    ObjectiveSpec(
                        target_name="target_001",
                        operator="greater_or_equal",
                        value=12.0,
                        requirement="hard",
                    )
                ],
                variables=[
                    VariableSpec(name="feature_001", lower=0.0, upper=1.0),
                    VariableSpec(name="feature_002", type="integer", lower=0, upper=5),
                    VariableSpec(
                        name="feature_003",
                        type="categorical",
                        categories=[0, 1, 2],
                    ),
                ],
                strategy_thresholds=StrategyThresholds(
                    candidate_rank_max_count=10,
                ),
                algorithm_override="mixed_nsga2",
                max_evaluations=180,
                top_n=3,
            )
            request.model_registry_path = str(
                root / "models" / "model-registry.json"
            )
            result = optimize_formula(request, output_dir=root / "optimizations")
            self.assertGreaterEqual(len(result.selected_candidates), 1)
            for candidate in result.selected_candidates:
                self.assertIsInstance(candidate.values["feature_002"], int)
                self.assertIn(candidate.values["feature_003"], [0, 1, 2])
                self.assertGreaterEqual(
                    candidate.predicted_values["target_001"],
                    12.0 - 1e-7,
                )


class BayesianOptimizationTests(unittest.TestCase):
    def test_bo_uses_complete_observed_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = _bo_frame()
            train_models(
                frame,
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002"],
                    algorithms=["ridge"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            history = [
                {
                    "experiment_id": f"experiment_{index:03d}",
                    "values": {
                        "feature_001": float(frame.iloc[index]["feature_001"]),
                        "feature_002": float(frame.iloc[index]["feature_002"]),
                    },
                    "observed_values": {
                        "target_001": float(frame.iloc[index]["target_001"])
                    },
                }
                for index in range(12)
            ]
            request = OptimizationRequest.from_dict({
                "request_id": "bo-next-experiments",
                "objectives": [{
                    "target_name": "target_001",
                    "operator": "equal",
                    "value": 3.5,
                }],
                "variables": [
                    {"name": "feature_001", "lower": 0.1, "upper": 0.9},
                    {"name": "feature_002", "lower": 0.1, "upper": 0.9},
                ],
                "historical_experiments": history,
                "top_n": 3,
                "max_evaluations": 1000,
            })
            request.model_registry_path = str(
                root / "models" / "model-registry.json"
            )
            result = optimize_next_experiments(
                request, output_dir=root / "optimizations"
            )
            self.assertEqual(result.status, "COMPLETE")
            self.assertEqual(len(result.selected_candidates), 3)
            self.assertEqual(
                result.diagnostics["selected_strategy"], "bo"
            )
            for candidate in result.selected_candidates:
                self.assertEqual(candidate.acquisition_name, "ei")
                self.assertIsNotNone(candidate.acquisition_value)

    def test_insufficient_history_falls_back_to_cold_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = _bo_frame()
            train_models(
                frame,
                TrainingConfig(
                    target_names=["target_001"],
                    feature_names=["feature_001", "feature_002"],
                    algorithms=["ridge"],
                    cv_mode="kfold",
                ),
                output_dir=root / "models",
            )
            history = [
                {
                    "experiment_id": "experiment_001",
                    "values": {
                        "feature_001": 0.2,
                        "feature_002": 0.4,
                    },
                    "observed_values": {"target_001": 2.4},
                }
            ]
            request = OptimizationRequest.from_dict({
                "request_id": "bo-cold-start",
                "objectives": [{
                    "target_name": "target_001",
                    "operator": "equal",
                    "value": 3.5,
                }],
                "variables": [
                    {"name": "feature_001", "lower": 0, "upper": 1},
                    {"name": "feature_002", "lower": 0, "upper": 1},
                ],
                "historical_experiments": history,
                "top_n": 3,
                "max_evaluations": 200,
            })
            request.model_registry_path = str(
                root / "models" / "model-registry.json"
            )
            result = optimize_next_experiments(
                request, output_dir=root / "optimizations"
            )
            self.assertEqual(
                result.diagnostics["selected_strategy"], "cold_start_design"
            )
            self.assertTrue(any(
                item["code"] == "BO_COLD_START_FALLBACK"
                for item in result.warnings
            ))
            for candidate in result.selected_candidates:
                self.assertIsNone(candidate.acquisition_name)


if __name__ == "__main__":
    unittest.main()
