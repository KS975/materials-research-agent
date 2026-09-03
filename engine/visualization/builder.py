from __future__ import annotations

from collections import Counter
from typing import Any

from engine.contracts import (
    VisualizationBundle,
    VisualizationColumn,
    VisualizationDataset,
)
from engine.exceptions import ValidationError


SOURCE_KINDS = {"preprocessing", "training", "prediction", "optimization"}


def build_visualization_bundle(
    payload: dict[str, Any],
    *,
    source_kind: str,
    source_uri: str,
) -> VisualizationBundle:
    """Build UI-neutral chart/table datasets from persisted engine reports."""
    if source_kind not in SOURCE_KINDS:
        raise ValidationError(
            f"unsupported visualization source_kind: {source_kind}"
        )
    if not isinstance(payload, dict):
        raise ValidationError("visualization payload must be a JSON object")

    if source_kind == "preprocessing":
        datasets = _preprocessing_datasets(payload)
    elif source_kind == "training":
        training_run = payload.get("training_run", payload)
        datasets = _training_datasets(training_run)
    elif source_kind == "prediction":
        datasets = _prediction_datasets(payload)
    else:
        datasets = _optimization_datasets(payload)
    return VisualizationBundle(
        source_kind=source_kind,
        source_uri=source_uri,
        datasets=datasets,
    )


def _preprocessing_datasets(payload: dict[str, Any]) -> list[VisualizationDataset]:
    datasets: list[VisualizationDataset] = []
    source = _source(payload.get("dataset_artifact"))

    execution = payload.get("execution_report", {})
    before_after = [
        {
            "metric": "row_count",
            "before": execution.get("input_row_count"),
            "after": execution.get("output_row_count"),
        },
        {
            "metric": "column_count",
            "before": execution.get("input_column_count"),
            "after": execution.get("output_column_count"),
        },
    ]
    datasets.append(_dataset(
        dataset_id="preprocessing_before_after",
        dataset_kind="chart_with_table",
        chart_type="bar",
        title="Rows and columns before and after preprocessing",
        records=before_after,
        x_field="metric",
        y_fields=["before", "after"],
        source_artifact=source,
    ))

    initial = payload.get("initial_quality_report", {})
    details = initial.get("details", {})
    missing_rows = _quality_field_rows(details.get("missing_values"))
    if missing_rows:
        datasets.append(_dataset(
            dataset_id="preprocessing_missing_values",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="Missing values by field",
            records=missing_rows,
            x_field="field",
            y_fields=["missing_count"],
            source_artifact=source,
        ))

    outlier_payload = details.get("outliers", {})
    outlier_rows = list(outlier_payload.get("fields", []))
    if outlier_rows:
        datasets.append(_dataset(
            dataset_id="preprocessing_outliers",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="IQR outliers by field",
            records=outlier_rows,
            x_field="field",
            y_fields=["outlier_count", "ratio"],
            source_artifact=source,
        ))

    finding_rows: list[dict[str, Any]] = []
    for stage_name, report in (
        ("initial", initial), ("final", payload.get("final_quality_report", {}))
    ):
        for finding in report.get("findings", []):
            finding_rows.append({"stage": stage_name, **finding})
    if finding_rows:
        datasets.append(_dataset(
            dataset_id="preprocessing_quality_findings",
            dataset_kind="table",
            title="Data quality findings",
            records=finding_rows,
            source_artifact=source,
        ))

    gate_rows = [
        {"stage": stage_name, **gate}
        for stage_name, gate in (
            ("initial", payload.get("initial_gate", {})),
            ("final", payload.get("final_gate", {})),
        )
        if gate
    ]
    if gate_rows:
        datasets.append(_dataset(
            dataset_id="preprocessing_modeling_gate",
            dataset_kind="table",
            title="Modeling gate decisions",
            records=gate_rows,
            source_artifact=source,
        ))
    return datasets


def _training_datasets(payload: dict[str, Any]) -> list[VisualizationDataset]:
    datasets: list[VisualizationDataset] = []
    artifacts = payload.get("model_artifacts", [])
    source = _source(artifacts[0]) if artifacts else {}

    metric_rows = []
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        metric_rows.append({
            "target_name": artifact.get("target_name"),
            "algorithm": artifact.get("algorithm"),
            "model_id": artifact.get("model_id"),
            "version": artifact.get("version"),
            **metrics,
        })
    if metric_rows:
        datasets.append(_dataset(
            dataset_id="model_selected_metrics",
            dataset_kind="table",
            title="Selected model metrics",
            records=metric_rows,
            source_artifact=source,
        ))
        datasets.append(_dataset(
            dataset_id="model_test_r2",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="Selected model test R2",
            records=metric_rows,
            x_field="target_name",
            y_fields=["r2"],
            source_artifact=source,
        ))
        datasets.append(_dataset(
            dataset_id="model_test_rmse",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="Selected model test RMSE",
            records=metric_rows,
            x_field="target_name",
            y_fields=["rmse"],
            source_artifact=source,
        ))

    candidate_rows = []
    for candidate in payload.get("candidate_records", []):
        metrics = candidate.get("metrics") or {}
        candidate_rows.append({
            "target_name": candidate.get("target_name"),
            "algorithm": candidate.get("algorithm"),
            "status": candidate.get("status"),
            "selection_rank": candidate.get("selection_rank"),
            **metrics,
            "error": candidate.get("error"),
        })
    if candidate_rows:
        datasets.append(_dataset(
            dataset_id="model_candidate_cv_rmse",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="Candidate model CV RMSE",
            records=candidate_rows,
            x_field="target_name",
            y_fields=["cv_rmse_mean"],
            series_field="algorithm",
            source_artifact=source,
        ))
        datasets.append(_dataset(
            dataset_id="model_candidate_comparison",
            dataset_kind="table",
            title="Candidate model comparison",
            records=candidate_rows,
            source_artifact=source,
        ))

    for artifact in artifacts:
        target = str(artifact.get("target_name", "target"))
        target_token = _token(target)
        records = list(artifact.get("evaluation_records", []))
        if not records:
            continue
        datasets.append(_dataset(
            dataset_id=f"model_{target_token}_predicted_vs_actual",
            dataset_kind="chart_with_table",
            chart_type="scatter",
            title=f"Predicted vs actual: {target}",
            records=records,
            x_field="y_true",
            y_fields=["y_pred"],
            source_artifact=_source(artifact),
        ))
        datasets.append(_dataset(
            dataset_id=f"model_{target_token}_residuals",
            dataset_kind="chart_with_table",
            chart_type="scatter",
            title=f"Residuals: {target}",
            records=records,
            x_field="y_pred",
            y_fields=["residual"],
            source_artifact=_source(artifact),
        ))

    interpretability_rows: list[dict[str, Any]] = []
    for candidate in payload.get("candidate_records", []):
        if candidate.get("selection_rank") != 1:
            continue
        interpretation = candidate.get("interpretability", {})
        items = interpretation.get("feature_importance")
        value_name = "importance"
        if items is None:
            items = interpretation.get("coefficient_fields")
            value_name = "coefficient"
        for item in items or []:
            feature_name = item.get("feature_name", item.get("feature"))
            interpretability_rows.append({
                "target_name": candidate.get("target_name"),
                "algorithm": candidate.get("algorithm"),
                "feature_name": feature_name,
                "value_name": value_name,
                "value": item.get(value_name),
            })
    if interpretability_rows:
        datasets.append(_dataset(
            dataset_id="model_feature_explanation",
            dataset_kind="chart_with_table",
            chart_type="horizontal_bar",
            title="Selected model feature explanation",
            records=interpretability_rows,
            x_field="value",
            y_fields=["feature_name"],
            series_field="target_name",
            source_artifact=source,
        ))

    warning_rows = []
    for candidate in payload.get("candidate_records", []):
        for warning in candidate.get("warnings", []):
            warning_rows.append({
                "target_name": candidate.get("target_name"),
                "algorithm": candidate.get("algorithm"),
                **warning,
            })
    if warning_rows:
        datasets.append(_dataset(
            dataset_id="model_training_warnings",
            dataset_kind="table",
            title="Training warnings",
            records=warning_rows,
            source_artifact=source,
        ))
    return datasets


def _prediction_datasets(payload: dict[str, Any]) -> list[VisualizationDataset]:
    model = payload.get("model", {})
    predictions = payload.get("predictions", [])
    source = _source(model)
    datasets: list[VisualizationDataset] = []

    rows = []
    for index, prediction in enumerate(predictions):
        rows.append({
            "row_index": index,
            "target_name": prediction.get("target_name"),
            "predicted_value": prediction.get("predicted_value"),
            "prediction_uncertainty": prediction.get("prediction_uncertainty"),
            "applicability_domain": prediction.get("applicability_domain"),
            "warning_count": len(prediction.get("warnings", [])),
        })
    if rows:
        datasets.append(_dataset(
            dataset_id="prediction_values",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="Prediction values and applicability domain",
            records=rows,
            x_field="row_index",
            y_fields=["predicted_value", "prediction_uncertainty"],
            source_artifact=source,
        ))
        datasets.append(_dataset(
            dataset_id="prediction_detail",
            dataset_kind="table",
            title="Prediction detail",
            records=rows,
            source_artifact=source,
        ))

        domain_counts = Counter(
            item.get("applicability_domain", "UNKNOWN") for item in rows
        )
        domain_rows = [
            {"applicability_domain": name, "count": count}
            for name, count in sorted(domain_counts.items())
        ]
        datasets.append(_dataset(
            dataset_id="prediction_applicability_domain",
            dataset_kind="chart_with_table",
            chart_type="bar",
            title="Applicability domain distribution",
            records=domain_rows,
            x_field="applicability_domain",
            y_fields=["count"],
            source_artifact=source,
        ))
    return datasets


def _optimization_datasets(payload: dict[str, Any]) -> list[VisualizationDataset]:
    from engine.optimization.service import build_optimization_visualizations

    datasets: list[VisualizationDataset] = []
    for item in build_optimization_visualizations(payload):
        data = dict(item)
        data.pop("record_type", None)
        data["columns"] = [
            VisualizationColumn(**column) for column in data.get("columns", [])
        ]
        datasets.append(VisualizationDataset(**data))
    return datasets


def _quality_field_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = list(payload.get("targets", []))
    rows.extend(payload.get("features", []))
    return rows


def _source(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "dataset_id", "version", "artifact_dir", "file_path", "data_hash",
        "model_id", "target_name", "algorithm",
    )
    return {key: payload.get(key) for key in keys if payload.get(key) is not None}


def _dataset(
    *,
    dataset_id: str,
    dataset_kind: str,
    title: str,
    records: list[dict[str, Any]],
    source_artifact: dict[str, Any],
    chart_type: str | None = None,
    x_field: str | None = None,
    y_fields: list[str] | None = None,
    series_field: str | None = None,
) -> VisualizationDataset:
    columns = _columns(records)
    return VisualizationDataset(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        chart_type=chart_type,
        title=title,
        columns=columns,
        records=records,
        x_field=x_field,
        y_fields=list(y_fields or []),
        series_field=series_field,
        source_artifact=source_artifact,
    )


def _columns(records: list[dict[str, Any]]) -> list[VisualizationColumn]:
    names: list[str] = []
    for record in records:
        for name in record:
            if name not in names:
                names.append(name)
    return [VisualizationColumn(name=name, label=name) for name in names]


def _token(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    )
