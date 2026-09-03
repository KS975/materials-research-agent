from __future__ import annotations

from engine.contracts import (
    DataQualityReport,
    GateDecision,
    ModelingGateResult,
    Severity,
)


def apply_modeling_gate(report: DataQualityReport) -> ModelingGateResult:
    blocking = [
        item.check for item in report.findings
        if item.severity is Severity.error
    ]
    warnings = [
        item.check for item in report.findings
        if item.severity is Severity.warning
    ]
    ratio = report.sample_feature_ratio
    if ratio < 1:
        tier = 1
    elif ratio < 3:
        tier = 2
    else:
        tier = 3

    reasons = [
        f"{item.check}: {item.reason}" for item in report.findings
        if item.severity is not Severity.info
    ]
    if blocking:
        decision = GateDecision.failed
    elif warnings:
        decision = GateDecision.conditional
    else:
        decision = GateDecision.passed

    return ModelingGateResult(
        decision=decision,
        reasons=reasons,
        recommended_tier=tier,
        blocking_items=blocking,
        warning_items=warnings,
    )
