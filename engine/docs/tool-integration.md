# Engine Tool Integration Guide

## Boundary

The independent engine exposes one JSON tool per capability. A host agent owns
upload handling, permissions, sessions, orchestration, and UI. The engine tools
own preprocessing, modeling, prediction, and optimization execution.

```text
host Tool Registry
  -> engine/tools/<capability>.py
     -> engine.dataset / engine.modeling / engine.optimization
        -> Dataset / Model / Optimization artifacts
```

The tool layer does not depend on FastAPI, LangGraph, Demo3, React, or a
database. It can be copied with `engine/` into another Python project.

## Available tools

| Tool | Input | Output |
|---|---|---|
| `preprocess_dataset` | tabular input URI, optional config/metadata | preprocessing report and DatasetArtifact reference |
| `train_model` | DatasetArtifact URI, optional config | strategy, candidate metrics, model artifacts, registry entries |
| `predict_model` | registry path, model selector, inline records or input URI | model metadata, predictions, applicability domains |
| `optimize_formula` | OptimizationRequest JSON | selected candidates, diagnostics, warnings, artifact references |
| `recommend_next_experiments` | OptimizationRequest JSON with history | BO or cold-start experiment recommendations |
| `list_artifacts` | dataset roots / model registries and optional selectors | lightweight dataset/model inventories |
| `get_chart_data` | persisted engine report URI | UI-neutral chart/table datasets |

Every function accepts a JSON object (or serialized JSON object) and returns:

```json
{
  "schema_version": 1,
  "tool": "tool_name",
  "status": "OK",
  "result": {}
}
```

Errors are returned as structured JSON instead of raised to the host agent.

## Host registration example

```python
from engine.tools import train_model

# LangChain/LangGraph example; other agents use their equivalent registration.
registered = StructuredTool.from_function(
    func=train_model.run_tool,
    name=train_model.TOOL_NAME,
    description=train_model.TRAIN_MODEL_SPEC["description"],
)
```

Hosts may also call `engine.tools.run_tool_by_name(name, payload)` or inspect
`engine.tools.tool_specs()` to generate registration metadata.

Long-running tools accept `result_mode: "summary"` when the host only needs the
decision, warnings, metrics, and Artifact references; `full` preserves all
technical records. Artifact defaults resolve from one configurable root:

```powershell
$env:ENGINE_ARTIFACT_ROOT = "engine/artifacts"
```

## Test commands

```powershell
# All tests
python -B -m unittest discover -s engine/tests -v

# Unit tests only
python -B -m unittest discover -s engine/tests/unit -v

# End-to-end tool workflow only
python -B -m unittest engine.tests.integration.test_tool_workflow -v
```

## XLSX constraints

Agent tools do not accept XLSX. A host upload pipeline first converts legacy
workbooks to JSON with the offline CLI:

```powershell
python -m engine.cli constraints --input RAFM-constraints.xlsx --output constraints.json
```

The command reads the workbook through the standard library and writes UTF-8
JSON without modifying the source file. The resulting `variables` and
`target_bounds` can be mapped by the host or planner into the optimization
request contract.
