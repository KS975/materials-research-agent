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

The registration is intentionally independent from intent routing and Skill
allow-lists. Chat orchestration can adopt the tools in a later, separately
reviewed change. Engine-owned outputs default under `engine/artifacts/`; set
`ENGINE_ARTIFACT_ROOT` to relocate them.
