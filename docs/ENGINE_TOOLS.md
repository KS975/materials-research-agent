# Independent Engine Tools

`engine/` is a framework-neutral package for preprocessing, modeling, prediction,
and formula optimization. It does not depend on FastAPI, the chat UI, MySQL, or
the existing material tools.

The application container registers these JSON tools into the existing
`ToolRegistry`:

| Tool | Purpose |
|---|---|
| `preprocess_dataset` | Build a versioned DatasetArtifact from CSV/Parquet input. |
| `train_model` | Train target models from a DatasetArtifact. |
| `predict_model` | Predict with a registered model and applicability domain. |
| `optimize_formula` | Generate ranked formula/process candidates. |
| `recommend_next_experiments` | Recommend BO or cold-start experiments. |
| `list_artifacts` | Discover available datasets and models. |
| `get_chart_data` | Convert a persisted report into UI-neutral chart/table data. |

Every tool handler accepts one `payload` object and returns a JSON object:

```python
result = registry.execute(
    "list_artifacts",
    payload={"dataset_roots": ["/path/to/datasets"]},
)
```

The raw registration remains framework-neutral. The Agent adopts the tools
through `EngineWorkflowAdapter`, rather than exposing path-bearing payloads to
the language model. The adapter owns permission checks, project selection,
artifact paths, model lookup and the fixed Tool order.

Public natural-language entry points are limited to:

| Intent | Public entry Tool | Host-controlled workflow |
|---|---|---|
| `engine_prepare_dataset` | `preprocess_dataset` | authorized snapshot -> preprocess -> Modeling Gate |
| `automl_training` | `train_model` | authorized snapshot -> preprocess -> Gate -> train/register |
| `predict_performance` | `predict_model` | list models -> select project model -> validate input -> predict |
| `optimize_formula` | `optimize_formula` | list models -> bind objectives -> optimize -> chart data |
| `recommend_next_experiments` | `recommend_next_experiments` | list models -> authorized history -> recommend -> chart data |

`list_artifacts` and `get_chart_data` are internal workflow steps and are not
valid DeepSeek routing choices. LLM-provided source paths, artifact roots and
model registry paths are discarded.

Engine-owned outputs default under `.runtime/engine_artifacts/`; set
`ENGINE_ARTIFACT_ROOT` to relocate them. The host scopes the files as:

```text
companies/<company>/projects/project_<id>/
  models/model-registry.json
  sessions/<conversation>/datasets/
  sessions/<conversation>/optimizations/
```

Model registries are reusable across conversations inside the same Company and
Project. A different Company cannot select the same registry path.
