from __future__ import annotations
from typing import Any

import pandas as pd
import numpy as np

from engine.contracts import (
    ClosureConfig,
    DataQualityReport,
    LeakageConfig,
    PreprocessingRuleRecord,
    QualityFinding,
    QualityThresholdConfig,
    Severity,
    TestConsistencySpec,
)


def run_quality_checks(
    dataframe: pd.DataFrame,
    *,
    target_fields: list[str],
    feature_fields: list[str] | None = None,
    identifier_fields: list[str] | None = None,
    thresholds: QualityThresholdConfig | None = None,
    closure_config: ClosureConfig | None = None,
    consistency_specs: list[TestConsistencySpec] | None = None,
    leakage_config: LeakageConfig | None = None,
) -> DataQualityReport:
    config = thresholds or QualityThresholdConfig()
    targets = list(target_fields)
    if not targets:
        raise ValueError("target_fields must not be empty")
    missing_columns = set(targets) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"target fields missing from DataFrame: {sorted(missing_columns)}")

    identifiers = list(identifier_fields or [])
    features = feature_fields or [
        column for column in dataframe.columns if column not in set(targets) | set(identifiers)
    ]
    feature_columns = set(dataframe.columns)
    missing_features = set(features) - feature_columns
    if missing_features:
        raise ValueError(f"feature fields missing from DataFrame: {sorted(missing_features)}")

    findings: list[QualityFinding] = []
    details: dict[str, Any] = {
        "target_fields": targets,
        "feature_fields": features,
        "identifier_fields": identifiers,
    }
    closure_config = closure_config or ClosureConfig(identifier_fields=identifiers)
    leakage_config = leakage_config or LeakageConfig()

    findings.extend(_check_sample_count(dataframe, targets, features, config))
    findings.extend(_check_missing(dataframe, targets, features, config))
    findings.extend(_check_duplicates(dataframe, targets, features, identifiers, config))
    findings.extend(_check_outliers(dataframe, targets, features, config, details))
    findings.extend(_check_sample_closure(dataframe, closure_config, details))
    findings.extend(_check_test_consistency(dataframe, consistency_specs or [], details))
    findings.extend(_check_leakage(
        dataframe,
        targets,
        features,
        leakage_config,
        details,
    ))

    details["schema"] = {
        "status": "pass",
        "columns": list(dataframe.columns),
        "target_fields": targets,
        "feature_fields": features,
        "identifier_fields": identifiers,
    }
    details["sample_count"] = {
        "row_count": len(dataframe),
        "feature_count": len(features),
        "target_count": len(targets),
        "sample_feature_ratio": len(dataframe) / len(features) if features else None,
        "thresholds": {
            "min_total_samples": config.min_total_samples,
            "min_feature_count": config.min_feature_count,
            "min_samples_per_target": config.min_samples_per_target,
            "min_sample_feature_ratio": config.min_sample_feature_ratio,
        },
    }
    details["missing_values"] = _missing_details(dataframe, targets, features, config)
    details["duplicates"] = _duplicate_details(
        dataframe,
        targets,
        features,
        identifiers,
        config,
    )

    ratio = len(dataframe) / len(features) if features else float("inf")
    rule_records = _build_rule_records(
        dataframe=dataframe,
        targets=targets,
        features=features,
        identifiers=identifiers,
        config=config,
        closure_config=closure_config,
        consistency_specs=consistency_specs or [],
        leakage_config=leakage_config,
        findings=findings,
        details=details,
    )
    return DataQualityReport(
        row_count=len(dataframe),
        feature_count=len(features),
        target_count=len(targets),
        sample_feature_ratio=ratio,
        findings=findings,
        details=details,
        rule_records=rule_records,
    )


def _missing_details(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    config: QualityThresholdConfig,
) -> dict[str, Any]:
    target_reports: list[dict[str, Any]] = []
    for field in targets:
        mask = dataframe[field].isna()
        count = int(mask.sum())
        ratio = count / len(dataframe) if len(dataframe) else 1.0
        target_reports.append({
            "field": field,
            "role": "target",
            "missing_count": count,
            "ratio": ratio,
            "threshold": config.max_target_missing_ratio,
            "row_indices": [item for item in dataframe.index[mask]],
            "recommendation": "drop_rows",
        })

    feature_reports: list[dict[str, Any]] = []
    for field in features:
        mask = dataframe[field].isna()
        count = int(mask.sum())
        ratio = count / len(dataframe) if len(dataframe) else 1.0
        recommendation = (
            "drop_field" if ratio > config.max_feature_missing_ratio else "impute"
        )
        feature_reports.append({
            "field": field,
            "role": "feature",
            "missing_count": count,
            "ratio": ratio,
            "threshold": config.max_feature_missing_ratio,
            "row_indices": [item for item in dataframe.index[mask]],
            "recommendation": recommendation,
        })
    return {
        "status": "warning_or_error_detected" if any(
            item["ratio"] > item["threshold"] for item in target_reports + feature_reports
        ) else "pass",
        "targets": target_reports,
        "features": feature_reports,
    }


def _duplicate_details(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    identifiers: list[str],
    config: QualityThresholdConfig,
) -> dict[str, Any]:
    exact_mask = dataframe.duplicated(keep="first")
    exact_count = int(exact_mask.sum())
    group_fields = identifiers or features
    conflict_reports: list[dict[str, Any]] = []
    conflict_count = 0
    if group_fields:
        grouped = dataframe.groupby(group_fields, dropna=False)
        for target in targets:
            counts = grouped[target].nunique(dropna=False)
            conflict_keys = counts[counts > 1].index
            rows: list[Any] = []
            for key in conflict_keys:
                if isinstance(key, tuple):
                    mask = pd.Series(True, index=dataframe.index)
                    for field, value in zip(group_fields, key):
                        mask &= dataframe[field].eq(value)
                else:
                    mask = dataframe[group_fields[0]].eq(key)
                rows.extend(dataframe.index[mask])
            count = len(rows)
            conflict_count += count
            conflict_reports.append({
                "target_field": target,
                "conflicting_row_count": count,
                "row_indices": list(dict.fromkeys(rows)),
            })
    return {
        "exact_duplicate_count": exact_count,
        "exact_duplicate_ratio": exact_count / len(dataframe) if len(dataframe) else 0.0,
        "exact_duplicate_row_indices": [item for item in dataframe.index[exact_mask]],
        "threshold": config.max_duplicate_ratio,
        "target_conflict_row_count": conflict_count,
        "target_conflicts": conflict_reports,
    }


def _build_rule_records(
    *,
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    identifiers: list[str],
    config: QualityThresholdConfig,
    closure_config: ClosureConfig,
    consistency_specs: list[TestConsistencySpec],
    leakage_config: LeakageConfig,
    findings: list[QualityFinding],
    details: dict[str, Any],
) -> list[PreprocessingRuleRecord]:
    specifications = [
        {
            "rule_name": "schema_validation",
            "definition": {
                "required_targets": targets,
                "required_features": features,
                "optional_identifiers": identifiers,
            },
            "checks": set(),
            "scope": targets + features + identifiers,
            "details_key": "schema",
        },
        {
            "rule_name": "sample_and_dimension_gate",
            "definition": {
                "min_total_samples": config.min_total_samples,
                "min_feature_count": config.min_feature_count,
                "min_samples_per_target": config.min_samples_per_target,
            },
            "checks": {"sample_count", "feature_count", "target_sample_count"},
            "scope": targets + features,
            "details_key": "sample_count",
        },
        {
            "rule_name": "missing_value_detection",
            "definition": {
                "target_threshold": config.max_target_missing_ratio,
                "feature_threshold": config.max_feature_missing_ratio,
                "target_recommendation": "drop_rows",
                "feature_recommendation": "drop_field_or_impute",
            },
            "checks": {"target_missing", "feature_missing"},
            "scope": targets + features,
            "details_key": "missing_values",
        },
        {
            "rule_name": "duplicate_and_conflict_detection",
            "definition": {
                "duplicate_threshold": config.max_duplicate_ratio,
                "duplicate_strategy": "keep_first",
                "target_conflict_strategy": "fail",
            },
            "checks": {"exact_duplicate", "target_conflict"},
            "scope": targets + features + identifiers,
            "details_key": "duplicates",
        },
        {
            "rule_name": "iqr_outlier_detection",
            "definition": {
                "method": "IQR",
                "multiplier": details.get("outliers", {}).get("iqr_multiplier", 1.5),
                "lower_formula": "Q1 - 1.5 * IQR",
                "upper_formula": "Q3 + 1.5 * IQR",
                "recommendation": "delete_rows",
                "executed": False,
            },
            "checks": {"feature_outlier", "global_outlier"},
            "scope": targets + features,
            "details_key": "outliers",
        },
        {
            "rule_name": "sample_closure_validation",
            "definition": {
                "identifier_fields": closure_config.identifier_fields,
                "required_fields": closure_config.required_fields,
                "min_closure_ratio": closure_config.min_closure_ratio,
                "max_ambiguous_identifier_count": closure_config.max_ambiguous_identifier_count,
            },
            "checks": {"sample_closure", "ambiguous_sample_identifier"},
            "scope": closure_config.identifier_fields + closure_config.required_fields,
            "details_key": "sample_closure",
        },
        {
            "rule_name": "test_consistency_validation",
            "definition": {
                "specs": [vars(spec) for spec in consistency_specs],
                "strategy": "fail_on_mismatch",
            },
            "checks": {"test_consistency", "test_condition_missing"},
            "scope": sorted({
                field
                for spec in consistency_specs
                for field in [
                    spec.target_field,
                    spec.test_field,
                    spec.unit_field,
                    spec.method_field,
                    *spec.required_condition_fields,
                ]
                if field is not None
            }),
            "details_key": "test_consistency",
        },
        {
            "rule_name": "target_leakage_detection",
            "definition": {
                "post_experiment_fields": leakage_config.post_experiment_fields,
                "forbidden_fields": leakage_config.forbidden_fields,
                "correlation_threshold": leakage_config.target_derivation_correlation,
                "strategy": "remove_fields_or_fail",
            },
            "checks": {"explicit_leakage", "target_derived_feature"},
            "scope": features,
            "details_key": "leakage",
        },
    ]

    records: list[PreprocessingRuleRecord] = []
    for spec in specifications:
        related = [finding for finding in findings if finding.check in spec["checks"]]
        if any(finding.severity is Severity.error for finding in related):
            status = "error"
        elif any(finding.severity is Severity.warning for finding in related):
            status = "warning"
        elif spec["details_key"] == "test_consistency" and not consistency_specs:
            status = "not_configured"
        else:
            status = "pass"
        summary = _rule_summary(spec["rule_name"], status, related, details.get(spec["details_key"], {}))
        records.append(PreprocessingRuleRecord(
            stage="data_preprocessing",
            rule_name=spec["rule_name"],
            rule_version="1",
            rule_definition=spec["definition"],
            input_scope=spec["scope"],
            status=status,
            summary_text=summary,
            findings=related,
            details=details.get(spec["details_key"], {}),
        ))
    return records


def _rule_summary(
    rule_name: str,
    status: str,
    findings: list[QualityFinding],
    details: dict[str, Any],
) -> str:
    if rule_name == "iqr_outlier_detection":
        count = details.get("total_outlier_count", 0)
        if count:
            return (
                f"IQR 离群点检测发现 {count} 个离群值，建议删除相关样本；"
                "本轮仅输出记录并触发 warning，未执行删除。"
            )
        return "IQR 离群点检测未发现离群值。"
    if status == "not_configured":
        return "该检查未配置，本轮未执行。"
    if status == "pass":
        return f"{rule_name}: 通过。"
    if findings:
        reasons = "；".join(dict.fromkeys(finding.reason for finding in findings))
        return f"{rule_name}: {reasons}。"
    return f"{rule_name}: 通过。"


def _check_sample_count(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    config: QualityThresholdConfig,
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    if len(dataframe) < config.min_total_samples:
        findings.append(QualityFinding(
            check="sample_count",
            severity=Severity.error,
            affected_fields=[],
            metric_value=float(len(dataframe)),
            threshold=float(config.min_total_samples),
            reason="total sample count is below the configured minimum",
            suggestion="add more samples or change the modeling scope",
        ))
    if len(features) < config.min_feature_count:
        findings.append(QualityFinding(
            check="feature_count",
            severity=Severity.error,
            affected_fields=features,
            metric_value=float(len(features)),
            threshold=float(config.min_feature_count),
            reason="feature count is below the configured minimum",
            suggestion="add usable predictive fields",
        ))

    for target in targets:
        valid = int(dataframe[target].notna().sum())
        if valid < config.min_samples_per_target:
            findings.append(QualityFinding(
                check="target_sample_count",
                severity=Severity.error,
                affected_fields=[target],
                metric_value=float(valid),
                threshold=float(config.min_samples_per_target),
                reason=f"valid sample count for {target} is below the minimum",
                suggestion="add target measurements or exclude this target",
            ))
    return findings


def _check_sample_closure(
    dataframe: pd.DataFrame,
    config: ClosureConfig,
    details: dict[str, Any],
) -> list[QualityFinding]:
    identifiers = list(config.identifier_fields)
    required = list(config.required_fields)
    if not identifiers and not required:
        details["sample_closure"] = {"status": "not_configured"}
        return []

    configured_fields = identifiers + required
    missing = [field for field in configured_fields if field not in dataframe.columns]
    if missing:
        details["sample_closure"] = {
            "status": "missing_fields",
            "missing_fields": missing,
        }
        return [QualityFinding(
            check="sample_closure",
            severity=Severity.error,
            affected_fields=missing,
            reason="configured closure fields are missing",
            suggestion="provide the field or disable closure checking",
        )]

    closure_mask = pd.Series(True, index=dataframe.index)
    for field in configured_fields:
        closure_mask &= dataframe[field].notna()
    closed_count = int(closure_mask.sum())
    closure_ratio = closed_count / len(dataframe) if len(dataframe) else 0.0

    ambiguous_count = 0
    if identifiers:
        duplicated = dataframe.duplicated(subset=identifiers, keep=False)
        ambiguous_count = int(duplicated.sum())

    details["sample_closure"] = {
        "status": "ok" if closure_ratio >= config.min_closure_ratio else "insufficient",
        "closed_count": closed_count,
        "total_count": len(dataframe),
        "closure_ratio": closure_ratio,
        "ambiguous_identifier_count": ambiguous_count,
        "identifier_fields": identifiers,
        "required_fields": required,
    }

    findings: list[QualityFinding] = []
    if closure_ratio < config.min_closure_ratio:
        findings.append(QualityFinding(
            check="sample_closure",
            severity=Severity.error,
            affected_fields=configured_fields,
            metric_value=closure_ratio,
            threshold=config.min_closure_ratio,
            reason="sample closure ratio is below the configured minimum",
            suggestion="resolve missing identifiers or required relation fields",
        ))
    if ambiguous_count > config.max_ambiguous_identifier_count:
        findings.append(QualityFinding(
            check="ambiguous_sample_identifier",
            severity=Severity.error,
            affected_fields=identifiers,
            metric_value=float(ambiguous_count),
            threshold=float(config.max_ambiguous_identifier_count),
            reason="identifier values map to multiple records without a configured distinction",
            suggestion="add timestamp, test condition, or version fields to the identifier",
        ))
    return findings


def _check_test_consistency(
    dataframe: pd.DataFrame,
    specs: list[TestConsistencySpec],
    details: dict[str, Any],
) -> list[QualityFinding]:
    if not specs:
        details["test_consistency"] = {"status": "not_configured"}
        return []

    findings: list[QualityFinding] = []
    summary: list[dict[str, Any]] = []
    for spec in specs:
        if spec.target_field not in dataframe.columns:
            findings.append(QualityFinding(
                check="test_consistency",
                severity=Severity.error,
                affected_fields=[spec.target_field],
                reason="configured target field is missing",
                suggestion="correct the consistency configuration",
            ))
            continue

        checks: list[tuple[str, str | None, str | None]] = [
            ("test", spec.test_field, spec.expected_test),
            ("unit", spec.unit_field, spec.expected_unit),
            ("method", spec.method_field, spec.expected_method),
        ]
        mismatches: dict[str, int] = {}
        for label, field, expected in checks:
            if field is None or expected is None:
                continue
            if field not in dataframe.columns:
                mismatches[label] = len(dataframe)
                findings.append(QualityFinding(
                    check="test_consistency",
                    severity=Severity.error,
                    affected_fields=[field],
                    metric_value=float(len(dataframe)),
                    threshold=0,
                    reason=f"{label} field is missing for target {spec.target_field}",
                    suggestion="provide the field or remove the consistency check",
                ))
                continue
            actual = dataframe[field].astype("string").str.strip()
            expected_value = str(expected).strip()
            count = int((actual != expected_value).sum())
            mismatches[label] = count
            if count:
                findings.append(QualityFinding(
                    check="test_consistency",
                    severity=Severity.error,
                    affected_fields=[field],
                    metric_value=float(count),
                    threshold=0,
                    reason=f"{label} mismatch for target {spec.target_field}",
                    suggestion=f"split by {label}, convert units, or correct the source data",
                ))

        missing_conditions = 0
        condition_fields: list[str] = []
        for field in spec.required_condition_fields:
            condition_fields.append(field)
            if field not in dataframe.columns:
                missing_conditions += len(dataframe)
            else:
                missing_conditions += int(dataframe[field].isna().sum())
        if missing_conditions:
            findings.append(QualityFinding(
                check="test_condition_missing",
                severity=Severity.error,
                affected_fields=condition_fields,
                metric_value=float(missing_conditions),
                threshold=0,
                reason=f"required test conditions are missing for target {spec.target_field}",
                suggestion="collect conditions or exclude these rows from formal comparison",
            ))

        summary.append({
            "target_field": spec.target_field,
            "mismatches": mismatches,
            "missing_condition_count": missing_conditions,
        })

    details["test_consistency"] = {
        "status": "ok" if not findings else "inconsistent",
        "specs": summary,
    }
    return findings


def _check_leakage(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    config: LeakageConfig,
    details: dict[str, Any],
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    explicit_fields = [
        field for field in config.post_experiment_fields + config.forbidden_fields
        if field in features
    ]
    if explicit_fields:
        findings.append(QualityFinding(
            check="explicit_leakage",
            severity=Severity.error,
            affected_fields=explicit_fields,
            metric_value=float(len(explicit_fields)),
            threshold=0,
            reason="configured post-experiment or forbidden fields are present as features",
            suggestion="remove the fields and record them in cleaning lineage",
        ))

    derived_fields: list[str] = []
    correlations: dict[str, float] = {}
    for target in targets:
        if not pd.api.types.is_numeric_dtype(dataframe[target]):
            continue
        for feature in features:
            if not pd.api.types.is_numeric_dtype(dataframe[feature]):
                continue
            if pd.api.types.is_bool_dtype(dataframe[feature]):
                continue
            valid = dataframe[[feature, target]].dropna()
            if len(valid) < 3:
                continue
            feature_std = float(dataframe[feature].std(skipna=True))
            target_std = float(dataframe[target].std(skipna=True))
            if dataframe[feature].nunique(dropna=True) <= 1:
                continue
            if not np.isfinite(feature_std) or feature_std == 0:
                continue
            if not np.isfinite(target_std) or target_std == 0:
                continue
            correlation = abs(float(dataframe[feature].corr(dataframe[target])))
            correlations[f"{feature}->{target}"] = correlation
            if correlation >= config.target_derivation_correlation:
                derived_fields.append(feature)

    derived_fields = sorted(set(derived_fields))
    if derived_fields:
        findings.append(QualityFinding(
            check="target_derived_feature",
            severity=Severity.error,
            affected_fields=derived_fields,
            metric_value=float(len(derived_fields)),
            threshold=0,
            reason="feature is almost perfectly correlated with a target and may be target-derived",
            suggestion="confirm business availability before prediction; otherwise remove it",
        ))

    details["leakage"] = {
        "status": "ok" if not findings else "leakage_detected",
        "explicit_fields": explicit_fields,
        "suspected_derived_fields": derived_fields,
        "max_abs_correlations": dict(
            sorted(correlations.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
    }
    return findings


def _check_missing(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    config: QualityThresholdConfig,
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for target in targets:
        count = int(dataframe[target].isna().sum())
        ratio = count / len(dataframe) if len(dataframe) else 1.0
        severity = Severity.error if ratio > config.max_target_missing_ratio else Severity.info
        findings.append(QualityFinding(
            check="target_missing",
            severity=severity,
            affected_fields=[target],
            metric_value=ratio,
            threshold=config.max_target_missing_ratio,
            reason=f"{count} missing target values in {target}",
            suggestion="exclude missing rows for this target",
        ))

    for feature in features:
        count = int(dataframe[feature].isna().sum())
        ratio = count / len(dataframe) if len(dataframe) else 1.0
        severity = Severity.error if ratio > config.max_feature_missing_ratio else Severity.info
        findings.append(QualityFinding(
            check="feature_missing",
            severity=severity,
            affected_fields=[feature],
            metric_value=ratio,
            threshold=config.max_feature_missing_ratio,
            reason=f"{count} missing feature values in {feature}",
            suggestion="impute, drop the field, or collect more measurements",
        ))
    return findings


def _check_duplicates(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    identifiers: list[str],
    config: QualityThresholdConfig,
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    exact_count = int(dataframe.duplicated(keep="first").sum())
    ratio = exact_count / len(dataframe) if len(dataframe) else 0.0
    severity = Severity.error if ratio > config.max_duplicate_ratio else Severity.info
    findings.append(QualityFinding(
        check="exact_duplicate",
        severity=severity,
        affected_fields=[],
        metric_value=ratio,
        threshold=config.max_duplicate_ratio,
        reason=f"{exact_count} exact duplicate rows",
        suggestion="deduplicate and record the rule in dataset lineage",
    ))

    group_fields = identifiers or features
    conflict_count = 0
    conflict_fields: list[str] = []
    if group_fields:
        grouped = dataframe.groupby(group_fields, dropna=False)
        for target in targets:
            conflicting = grouped[target].nunique(dropna=False)
            conflicting = conflicting[conflicting > 1]
            count = int(conflicting.sum())
            if count:
                conflict_count += count
                conflict_fields.append(target)
    if conflict_count > config.max_target_conflict_count:
        findings.append(QualityFinding(
            check="target_conflict",
            severity=Severity.error,
            affected_fields=conflict_fields,
            metric_value=float(conflict_count),
            threshold=float(config.max_target_conflict_count),
            reason="same input keys map to multiple target values",
            suggestion="separate test conditions, aggregate, or resolve conflicts manually",
        ))
    return findings


def _check_outliers(
    dataframe: pd.DataFrame,
    targets: list[str],
    features: list[str],
    config: QualityThresholdConfig,
    details: dict[str, Any],
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    numeric_fields = [
        field for field in features + targets
        if pd.api.types.is_numeric_dtype(dataframe[field])
        and not pd.api.types.is_bool_dtype(dataframe[field])
    ]
    total_outliers = 0
    total_values = 0
    field_reports: list[dict[str, Any]] = []
    k = 1.5

    for field in numeric_fields:
        values = dataframe[field].dropna()
        if values.empty:
            field_reports.append({
                "field": field,
                "role": "target" if field in targets else "feature",
                "outlier_count": 0,
                "evaluated_count": 0,
                "ratio": 0.0,
                "row_indices": [],
            })
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        if iqr == 0:
            mask = values.ne(q1)
        else:
            mask = (values < lower_bound) | (values > upper_bound)
        count = int(mask.sum())
        ratio = count / len(values)
        aligned_mask = mask.reindex(dataframe.index, fill_value=False)
        row_indices = [item for item in dataframe.index[aligned_mask]]
        total_outliers += count
        total_values += len(values)
        field_reports.append({
            "field": field,
            "role": "target" if field in targets else "feature",
            "outlier_count": count,
            "evaluated_count": len(values),
            "ratio": ratio,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "row_indices": row_indices,
            "exceeds_threshold": ratio > config.max_single_feature_outlier_ratio,
        })
        if count:
            findings.append(QualityFinding(
                check="feature_outlier",
                severity=Severity.warning,
                affected_fields=[field],
                affected_rows=row_indices,
                metric_value=ratio,
                threshold=config.max_single_feature_outlier_ratio,
                reason=f"IQR detected {count} outliers in {field}",
                suggestion="建议删除相关样本；本轮仅提示，未执行删除",
                suggested_action="delete_rows",
                executed=False,
            ))

    global_ratio = total_outliers / total_values if total_values else 0.0
    if global_ratio > config.max_global_outlier_ratio:
        findings.append(QualityFinding(
            check="global_outlier",
            severity=Severity.warning,
            affected_fields=numeric_fields,
            metric_value=global_ratio,
            threshold=config.max_global_outlier_ratio,
            reason="global numeric outlier ratio exceeds the threshold",
            suggestion="建议删除相关样本并检查数据单位与测量误差；本轮仅提示，未执行删除",
            suggested_action="delete_rows",
            executed=False,
        ))
    details["outliers"] = {
        "status": "warning_detected" if total_outliers else "ok",
        "detection_method": "iqr",
        "iqr_multiplier": k,
        "suggested_action": "delete_rows",
        "executed": False,
        "total_outlier_count": total_outliers,
        "total_evaluated_count": total_values,
        "global_outlier_ratio": global_ratio,
        "fields": field_reports,
    }
    return findings
