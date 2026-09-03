from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from engine.contracts import (
    CleaningOperationRecord,
    CleaningConfig,
    CleaningExecutionReport,
    CleaningStep,
    DataQualityReport,
    DatasetArtifact,
    GateDecisionRecord,
    GateDecision,
    ModelingGateResult,
    Severity,
)
from engine.dataset.builder import build_dataset
from engine.dataset.config_resolver import (
    ResolvedPreprocessingConfig,
    resolve_preprocessing_config,
)
from engine.exceptions import ValidationError
from engine.governance.gate import apply_modeling_gate
from engine.governance.quality import run_quality_checks
from engine.ingestion.reader import hash_dataframe


@dataclass
class DatasetPreprocessingResult:
    resolved_config: ResolvedPreprocessingConfig
    cleaning_plan: list[CleaningStep]
    execution_report: CleaningExecutionReport
    initial_quality_report: DataQualityReport
    final_quality_report: DataQualityReport
    initial_gate: ModelingGateResult
    final_gate: ModelingGateResult
    artifact: DatasetArtifact | None
    warnings: list[str] = field(default_factory=list)
    cleaning_operations: list[CleaningOperationRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_config": self.resolved_config.to_dict(),
            "cleaning_plan": [step.to_dict() for step in self.cleaning_plan],
            "execution_report": self.execution_report.to_dict(),
            "cleaning_operation_records": [
                record.to_dict() for record in self.cleaning_operations
            ],
            "initial_quality_report": self.initial_quality_report.to_dict(),
            "final_quality_report": self.final_quality_report.to_dict(),
            "initial_gate": self.initial_gate.to_dict(),
            "final_gate": self.final_gate.to_dict(),
            "gate_decision_records": [
                _gate_record("initial_modeling_gate", self.initial_gate).to_dict(),
                _gate_record("final_modeling_gate", self.final_gate).to_dict(),
            ],
            "dataset_artifact": self.artifact.to_dict() if self.artifact else None,
            "warnings": self.warnings,
            "stage_technical_summaries": [
                {
                    "stage": "initial_quality",
                    "status": "completed",
                    "technical_summary": self.initial_quality_report.technical_summary(),
                },
                {
                    "stage": "cleaning_plan",
                    "status": "completed",
                    "technical_summary": [
                        {
                            "rule_name": step.action,
                            "severity": "planned",
                            "message": f"{step.action}: {step.reason}",
                        }
                        for step in self.cleaning_plan
                    ],
                },
                {
                    "stage": "cleaning_execution",
                    "status": "completed",
                    "technical_summary": [
                        {
                            "rule_name": record.operation,
                            "severity": record.status,
                            "message": record.summary_text,
                        }
                        for record in self.cleaning_operations
                    ],
                },
                {
                    "stage": "final_quality",
                    "status": "completed",
                    "technical_summary": self.final_quality_report.technical_summary(),
                },
            ],
        }


def run_dataset_preprocessing(
    dataframe: pd.DataFrame,
    *,
    user_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    source_uri: str = "dataframe",
    source_hash: str | None = None,
    output_dir: str = "engine/artifacts/datasets",
) -> DatasetPreprocessingResult:
    resolved = resolve_preprocessing_config(
        dataframe,
        user_config=user_config,
        metadata=metadata,
        source_path=None if source_uri == "dataframe" else source_uri,
    )
    initial_report = _quality_report(dataframe, resolved)
    initial_gate = apply_modeling_gate(initial_report)
    plan = _build_cleaning_plan(dataframe, initial_report, resolved)
    resolved.cleaning_config = replace(
        resolved.cleaning_config,
        drop_fields=_plan_fields(plan, "drop_fields"),
        impute_numeric_fields=_plan_fields(plan, "impute_numeric_median"),
        impute_categorical_fields=_plan_fields(plan, "impute_categorical_mode"),
        missing_indicator_fields=(
            _plan_fields(plan, "impute_numeric_median")
            + _plan_fields(plan, "impute_categorical_mode")
            if resolved.cleaning_config.add_missing_indicators
            else []
        ),
    )
    cleaned, execution, operations = _apply_cleaning_plan(dataframe, resolved, plan)
    final_report = _quality_report(cleaned, resolved)
    final_gate = apply_modeling_gate(final_report)

    artifact = None
    if final_gate.decision is not GateDecision.failed:
        artifact = build_dataset(
            cleaned,
            target_fields=resolved.target_fields,
            feature_fields=final_report.details["feature_fields"],
            identifier_fields=resolved.identifier_fields,
            quality_report=final_report,
            gate_result=final_gate,
            cleaning_config=resolved.cleaning_config,
            source_uri=source_uri,
            source_hash=source_hash or hash_dataframe(dataframe),
            output_dir=output_dir,
            cleaning_already_applied=True,
            precleaning_report=execution,
        )
    return DatasetPreprocessingResult(
        resolved_config=resolved,
        cleaning_plan=plan,
        execution_report=execution,
        initial_quality_report=initial_report,
        final_quality_report=final_report,
        initial_gate=initial_gate,
        final_gate=final_gate,
        artifact=artifact,
        warnings=_result_warnings(initial_report, final_gate),
        cleaning_operations=operations,
    )


def _quality_report(
    dataframe: pd.DataFrame,
    resolved: ResolvedPreprocessingConfig,
) -> DataQualityReport:
    return run_quality_checks(
        dataframe,
        target_fields=resolved.target_fields,
        feature_fields=resolved.feature_fields,
        identifier_fields=resolved.identifier_fields,
        thresholds=resolved.thresholds,
        closure_config=resolved.closure_config,
        consistency_specs=resolved.consistency_specs,
        leakage_config=resolved.leakage_config,
    )


def _build_cleaning_plan(
    dataframe: pd.DataFrame,
    report: DataQualityReport,
    resolved: ResolvedPreprocessingConfig,
) -> list[CleaningStep]:
    steps: list[CleaningStep] = []
    config = resolved.cleaning_config
    missing = report.details.get("missing", {}) or {}
    high_missing_fields: list[str] = []
    impute_numeric: list[str] = []
    impute_categorical: list[str] = []

    for field in resolved.feature_fields:
        if field not in dataframe.columns:
            continue
        ratio = float(dataframe[field].isna().mean())
        if ratio == 0:
            continue
        if config.drop_high_missing_fields and ratio > config.max_feature_missing_ratio:
            high_missing_fields.append(field)
        elif config.impute_missing_features:
            if pd.api.types.is_numeric_dtype(dataframe[field]):
                impute_numeric.append(field)
            else:
                impute_categorical.append(field)

    leakage_details = report.details.get("leakage", {}) or {}
    leakage_fields = list(leakage_details.get("explicit_fields") or []) + list(
        leakage_details.get("suspected_derived_fields") or []
    )
    forbidden_fields = [
        field for field in resolved.leakage_config.forbidden_fields
        + resolved.leakage_config.post_experiment_fields
        if field in dataframe.columns
    ]
    removal_fields = sorted(set(
        high_missing_fields + leakage_fields + forbidden_fields + config.drop_fields
    ))
    if removal_fields:
        steps.append(CleaningStep(
            action="drop_fields",
            fields=removal_fields,
            reason="remove high-missing, user-dropped, or leakage-risk fields",
            source="system_safety_and_user_config",
        ))

    if config.drop_exact_duplicates:
        steps.append(CleaningStep(
            action="drop_exact_duplicates",
            fields=[],
            reason="remove exact duplicate records",
            source="default_safe_v1",
        ))
    if config.drop_missing_target_rows:
        steps.append(CleaningStep(
            action="drop_missing_target_rows",
            fields=resolved.target_fields,
            reason="target values cannot be imputed for supervised modeling",
            source="default_safe_v1",
        ))
    if impute_numeric:
        steps.append(CleaningStep(
            action="impute_numeric_median",
            fields=impute_numeric,
            reason="feature missing ratio is within the preset imputation threshold",
            source="default_safe_v1",
        ))
    if impute_categorical:
        steps.append(CleaningStep(
            action="impute_categorical_mode",
            fields=impute_categorical,
            reason="feature missing ratio is within the preset imputation threshold",
            source="default_safe_v1",
        ))
    return steps


def _apply_cleaning_plan(
    dataframe: pd.DataFrame,
    resolved: ResolvedPreprocessingConfig,
    plan: list[CleaningStep],
) -> tuple[pd.DataFrame, CleaningExecutionReport, list[CleaningOperationRecord]]:
    output = dataframe.copy(deep=True)
    config = resolved.cleaning_config
    actions = {step.action for step in plan}
    operations: list[CleaningOperationRecord] = []

    removed_fields: list[str] = []
    if "drop_fields" in actions:
        fields = [
            field for field in _plan_fields(plan, "drop_fields")
            if field in output.columns
        ]
        output = output.drop(columns=fields)
        removed_fields = fields
        resolved.feature_fields[:] = [
            field for field in resolved.feature_fields
            if field not in set(fields)
        ]
        if fields:
            operations.append(_operation_record(
                operation="drop_fields",
                source=_action_source(plan, "drop_fields"),
                parameters={"fields": fields},
                affected_fields=fields,
                affected_rows=[],
                input_count=len(dataframe),
                output_count=len(output),
                summary_text=(
                    f"按预处理策略删除 {len(fields)} 个字段："
                    f"{', '.join(fields)}。"
                ),
            ))

    dropped_duplicate_count = 0
    if config.drop_exact_duplicates:
        duplicate_mask = output.duplicated(keep="first")
        duplicate_rows = [item for item in output.index[duplicate_mask]]
        before = len(output)
        output = output.drop_duplicates(keep="first")
        dropped_duplicate_count = before - len(output)
        if dropped_duplicate_count:
            operations.append(_operation_record(
                operation="drop_exact_duplicates",
                source=_action_source(plan, "drop_exact_duplicates"),
                parameters={"keep": "first"},
                affected_fields=[],
                affected_rows=duplicate_rows,
                input_count=before,
                output_count=len(output),
                summary_text=(
                    f"删除 {dropped_duplicate_count} 条完全重复样本，保留第一条。"
                ),
            ))

    dropped_missing_target_count = 0
    if config.drop_missing_target_rows:
        missing_target_mask = output[resolved.target_fields].isna().any(axis=1)
        missing_target_rows = [item for item in output.index[missing_target_mask]]
        before = len(output)
        output = output.dropna(subset=resolved.target_fields)
        dropped_missing_target_count = before - len(output)
        if dropped_missing_target_count:
            operations.append(_operation_record(
                operation="drop_missing_target_rows",
                source=_action_source(plan, "drop_missing_target_rows"),
                parameters={"target_fields": resolved.target_fields},
                affected_fields=resolved.target_fields,
                affected_rows=missing_target_rows,
                input_count=before,
                output_count=len(output),
                summary_text=(
                    f"删除 {dropped_missing_target_count} 条目标缺失样本；"
                    "目标值不做插补。"
                ),
            ))

    imputed_numeric = [
        field for field in _plan_fields(plan, "impute_numeric_median")
        if field in output.columns
    ]
    imputed_categorical = [
        field for field in _plan_fields(plan, "impute_categorical_mode")
        if field in output.columns
    ]
    imputed_fields = imputed_numeric + imputed_categorical
    indicator_fields = imputed_fields if config.add_missing_indicators else []
    added_fields: list[str] = []
    imputed_value_count = 0
    for field in indicator_fields:
        if field not in output.columns:
            continue
        indicator = f"{field}_was_missing"
        output[indicator] = dataframe[field].isna().reindex(output.index, fill_value=False)
        added_fields.append(indicator)
        resolved.feature_fields.append(indicator)
    if added_fields:
        operations.append(_operation_record(
            operation="add_missing_indicators",
            source="default_safe_v1",
            parameters={"fields": indicator_fields},
            affected_fields=added_fields,
            affected_rows=[],
            input_count=len(output),
            output_count=len(output),
            summary_text=f"为 {len(added_fields)} 个插补字段添加缺失指示列。",
        ))
    for field in imputed_numeric:
        if field not in output.columns:
            continue
        missing_rows = [item for item in output.index[output[field].isna()]]
        count = int(output[field].isna().sum())
        output[field] = output[field].fillna(output[field].median())
        imputed_value_count += count
        if count:
            operations.append(_operation_record(
                operation="impute_numeric_median",
                source=_action_source(plan, "impute_numeric_median"),
                parameters={"field": field, "strategy": "median"},
                affected_fields=[field],
                affected_rows=missing_rows,
                input_count=len(output),
                output_count=len(output),
                summary_text=f"数值字段 {field} 以中位数插补 {count} 个缺失值。",
            ))
    for field in imputed_categorical:
        if field not in output.columns:
            continue
        missing_rows = [item for item in output.index[output[field].isna()]]
        count = int(output[field].isna().sum())
        mode = output[field].mode(dropna=True)
        value = mode.iloc[0] if not mode.empty else "__MISSING__"
        output[field] = output[field].fillna(value)
        imputed_value_count += count
        if count:
            operations.append(_operation_record(
                operation="impute_categorical_mode",
                source=_action_source(plan, "impute_categorical_mode"),
                parameters={"field": field, "strategy": "mode"},
                affected_fields=[field],
                affected_rows=missing_rows,
                input_count=len(output),
                output_count=len(output),
                summary_text=f"类别字段 {field} 以众数插补 {count} 个缺失值。",
            ))

    winsorized_fields: list[str] = []
    if config.outlier_strategy == "winsorize":
        for field in resolved.feature_fields + resolved.target_fields:
            if field not in output.columns or not pd.api.types.is_numeric_dtype(output[field]):
                continue
            q1 = output[field].quantile(0.25)
            q3 = output[field].quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            output[field] = output[field].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            winsorized_fields.append(field)
            operations.append(_operation_record(
                operation="winsorize_numeric_outliers",
                source="user_config",
                parameters={"field": field, "method": "IQR", "multiplier": 1.5},
                affected_fields=[field],
                affected_rows=[],
                input_count=len(output),
                output_count=len(output),
                summary_text=f"字段 {field} 的 IQR 离群值已按用户配置截尾。",
            ))

    output = output.reset_index(drop=True)
    execution = CleaningExecutionReport(
        input_row_count=len(dataframe),
        output_row_count=len(output),
        input_column_count=len(dataframe.columns),
        output_column_count=len(output.columns),
        dropped_duplicate_count=dropped_duplicate_count,
        dropped_missing_target_count=dropped_missing_target_count,
        removed_fields=removed_fields,
        added_fields=added_fields,
        imputed_fields=imputed_fields,
        imputed_value_count=imputed_value_count,
        winsorized_fields=winsorized_fields,
    )
    return output, execution, operations


def _operation_record(
    *,
    operation: str,
    source: str,
    parameters: dict[str, Any],
    affected_fields: list[str],
    affected_rows: list[Any],
    input_count: int,
    output_count: int,
    summary_text: str,
) -> CleaningOperationRecord:
    return CleaningOperationRecord(
        stage="data_preprocessing.cleaning",
        operation=operation,
        source=source,
        status="executed",
        parameters=parameters,
        affected_fields=affected_fields,
        affected_row_count=len(affected_rows),
        affected_row_indices=affected_rows,
        input_count=input_count,
        output_count=output_count,
        summary_text=summary_text,
    )


def _action_source(plan: list[CleaningStep], action: str) -> str:
    for step in plan:
        if step.action == action:
            return step.source
    return "default_safe_v1"


def _gate_record(stage: str, gate: ModelingGateResult) -> GateDecisionRecord:
    if gate.decision is GateDecision.failed:
        message = "Modeling Gate 为 FAIL，未生成正式 DatasetArtifact。"
    elif gate.decision is GateDecision.conditional:
        message = "Modeling Gate 为 CONDITIONAL_PASS，可生成实验数据集，但存在需关注项。"
    else:
        message = "Modeling Gate 为 PASS，预处理结果可用于正式建模。"
    return GateDecisionRecord(
        stage=stage,
        decision=gate.decision,
        summary_text=message,
        blocking_items=gate.blocking_items,
        warning_items=gate.warning_items,
    )


def _plan_fields(plan: list[CleaningStep], action: str) -> list[str]:
    fields: list[str] = []
    for step in plan:
        if step.action == action:
            fields.extend(step.fields)
    return list(dict.fromkeys(fields))


def _result_warnings(
    report: DataQualityReport,
    final_gate: ModelingGateResult,
) -> list[str]:
    warnings = [
        finding.reason for finding in report.findings
        if finding.severity is Severity.warning
    ]
    if final_gate.decision is GateDecision.failed:
        warnings.append("final modeling gate is FAIL; no formal dataset artifact was built")
    return warnings
