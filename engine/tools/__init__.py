"""JSON tool adapters for host agents.

The engine remains framework-neutral. A host agent imports one function and
registers it with its own tool runtime; no FastAPI or LangGraph dependency is
required here.
"""

from __future__ import annotations

from typing import Any, Callable

from engine.tools.get_chart_data import GET_CHART_DATA_SPEC, run_tool as get_chart_data
from engine.tools.list_artifacts import LIST_ARTIFACTS_SPEC, run_tool as list_artifacts
from engine.tools.optimize_formula import OPTIMIZE_FORMULA_SPEC, run_tool as optimize_formula
from engine.tools.preprocess_dataset import (
    PREPROCESS_DATASET_SPEC,
    run_tool as preprocess_dataset,
)
from engine.tools.predict_model import PREDICT_MODEL_SPEC, run_tool as predict_model
from engine.tools.recommend_next_experiments import (
    RECOMMEND_NEXT_EXPERIMENTS_SPEC,
    run_tool as recommend_next_experiments,
)
from engine.tools.train_model import TRAIN_MODEL_SPEC, run_tool as train_model


TOOL_SPECS: list[dict[str, Any]] = [
    PREPROCESS_DATASET_SPEC,
    TRAIN_MODEL_SPEC,
    PREDICT_MODEL_SPEC,
    OPTIMIZE_FORMULA_SPEC,
    RECOMMEND_NEXT_EXPERIMENTS_SPEC,
    LIST_ARTIFACTS_SPEC,
    GET_CHART_DATA_SPEC,
]

TOOL_FUNCTIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    item["name"]: function
    for item, function in zip(
        TOOL_SPECS,
        [
            preprocess_dataset,
            train_model,
            predict_model,
            optimize_formula,
            recommend_next_experiments,
            list_artifacts,
            get_chart_data,
        ],
    )
}


def tool_specs() -> list[dict[str, Any]]:
    """Return framework-neutral tool metadata for host registration."""
    return [dict(item) for item in TOOL_SPECS]


def run_tool_by_name(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a JSON tool request without coupling the engine to an agent."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        from engine.tools.common import failure

        return failure(
            "tool_dispatch",
            ValueError(f"unknown engine tool: {name}"),
            error_code="UNKNOWN_TOOL",
            details={"available_tools": sorted(TOOL_FUNCTIONS)},
        )
    return function(payload)


__all__ = [
    "OPTIMIZE_FORMULA_SPEC",
    "GET_CHART_DATA_SPEC",
    "LIST_ARTIFACTS_SPEC",
    "PREPROCESS_DATASET_SPEC",
    "PREDICT_MODEL_SPEC",
    "RECOMMEND_NEXT_EXPERIMENTS_SPEC",
    "TRAIN_MODEL_SPEC",
    "TOOL_FUNCTIONS",
    "TOOL_SPECS",
    "get_chart_data",
    "list_artifacts",
    "optimize_formula",
    "preprocess_dataset",
    "predict_model",
    "recommend_next_experiments",
    "run_tool_by_name",
    "tool_specs",
    "train_model",
]
