from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from engine.contracts import VisualizationColumn, VisualizationDataset
from engine.exceptions import ValidationError
from engine.optimization.bo import recommend_next_experiments
from engine.optimization.contracts import CandidateResult, OptimizationRequest, OptimizationResult
from engine.optimization.models import evaluate_model_quality, load_optimization_models
from engine.optimization.ranking import rank_and_select
from engine.optimization.search_space import build_search_space
from engine.optimization.strategy_selector import select_strategy
from engine.optimization.strategies import run_strategy


def optimize_formula(
    request: OptimizationRequest,
    *,
    output_dir: str | Path | None = "engine/artifacts/optimizations",
) -> OptimizationResult:
    request.validate()
    model_set = load_optimization_models(
        registry_path=request.model_registry_path,
        objectives=request.objectives,
        model_selection=request.model_selection,
    )
    space = build_search_space(request.variables, model_set)
    quality_warnings = evaluate_model_quality(
        model_set, request.model_quality_gate
    )
    if any(item["mode"] == "block" for item in quality_warnings):
        raise ValidationError("model quality gate blocked optimization")

    strategy, strategy_reason, strategy_inputs = select_strategy(
        request, space=space, model_set=model_set
    )
    run = run_strategy(
        strategy, request=request, space=space, model_set=model_set
    )
    selected, exploratory, diagnostic, ranking_warnings = rank_and_select(
        run.candidates,
        objectives=request.objectives,
        soft_constraints=request.soft_constraints,
        space=space,
        history=request.historical_candidates,
        top_n=request.top_n,
    )
    warnings = quality_warnings + ranking_warnings
    if len(selected) < request.top_n:
        warnings.append({
            "code": "INSUFFICIENT_TRUSTED_CANDIDATES",
            "message": "fewer trusted candidates were returned than requested",
            "requested": request.top_n,
            "returned": len(selected),
        })
    _apply_quality_trust(selected, quality_warnings)

    status = run.status
    if not selected and not exploratory:
        status = "EMPTY"
    diagnostics = _build_diagnostics(
        request=request,
        model_set=model_set,
        candidates=run.candidates,
        selected=selected,
        strategy=strategy,
        strategy_reason=strategy_reason,
        strategy_inputs=strategy_inputs,
        strategy_diagnostics=run.diagnostics,
        stop_reason=run.stop_reason,
        completed_evaluations=run.completed_evaluations,
        search_dimension=len(space.free_variables),
    )
    result = OptimizationResult(
        request_id=request.request_id or _request_id(),
        status=status,
        selected_candidates=selected,
        exploratory_candidates=exploratory,
        diagnostic_candidates=diagnostic,
        rejected_summary=_rejected_summary(run.candidates),
        diagnostics=diagnostics,
        warnings=warnings,
    )
    if output_dir is not None:
        _save_artifact(result, request, output_dir)
    return result


def optimize_next_experiments(
    request: OptimizationRequest,
    *,
    output_dir: str | Path | None = "engine/artifacts/optimizations",
) -> OptimizationResult:
    request.mode = "recommend_next_experiments"
    request.validate()
    model_set = load_optimization_models(
        registry_path=request.model_registry_path,
        objectives=request.objectives,
        model_selection=request.model_selection,
    )
    space = build_search_space(request.variables, model_set)
    quality_warnings = evaluate_model_quality(
        model_set, request.model_quality_gate
    )
    if any(item["mode"] == "block" for item in quality_warnings):
        raise ValidationError("model quality gate blocked optimization")
    selected, exploratory, diagnostic, diagnostics, ranking_warnings = (
        recommend_next_experiments(
            request=request, space=space, model_set=model_set
        )
    )
    warnings = quality_warnings + ranking_warnings
    _apply_quality_trust(selected, quality_warnings)
    diagnostics.update({
        "model_refs": model_set.model_refs(),
        "search_dimension": len(space.free_variables),
        "selected_count": len(selected),
        "strategy_thresholds": request.strategy_thresholds.to_dict(),
        "stop_reason": None,
    })
    result = OptimizationResult(
        request_id=request.request_id or _request_id(),
        status="COMPLETE" if selected else "EMPTY",
        selected_candidates=selected,
        exploratory_candidates=exploratory,
        diagnostic_candidates=diagnostic,
        rejected_summary={},
        diagnostics=diagnostics,
        warnings=warnings,
    )
    if output_dir is not None:
        _save_artifact(result, request, output_dir)
    return result


def _apply_quality_trust(
    candidates: list[CandidateResult],
    quality_warnings: list[dict[str, Any]],
) -> None:
    if quality_warnings:
        for candidate in candidates:
            if candidate.trust_level == "HIGH":
                candidate.trust_level = "MEDIUM"


def _build_diagnostics(
    *,
    request: OptimizationRequest,
    model_set: Any,
    candidates: list[CandidateResult],
    selected: list[CandidateResult],
    strategy: str,
    strategy_reason: str,
    strategy_inputs: dict[str, Any],
    strategy_diagnostics: dict[str, Any],
    stop_reason: str | None,
    completed_evaluations: int,
    search_dimension: int,
) -> dict[str, Any]:
    return {
        "selected_strategy": strategy,
        "strategy_reason": strategy_reason,
        "strategy_inputs": strategy_inputs,
        "search_dimension": search_dimension,
        "generated_count": max(len(candidates), completed_evaluations),
        "repaired_count": sum(
            any(
                item["kind"] != "target_threshold"
                for item in candidate.hard_constraint_report
            )
            for candidate in candidates
        ),
        "hard_feasible_count": sum(
            candidate.trust_level != "REJECTED" for candidate in candidates
        ),
        "model_evaluated_count": len(candidates),
        "in_domain_count": sum(
            candidate.applicability_domain == "IN_DOMAIN"
            for candidate in candidates
        ),
        "edge_count": sum(
            candidate.applicability_domain == "EDGE" for candidate in candidates
        ),
        "out_of_domain_count": sum(
            candidate.applicability_domain == "OUT_OF_DOMAIN"
            for candidate in candidates
        ),
        "selected_count": len(selected),
        "algorithm_parameters": strategy_diagnostics.get(
            "algorithm_parameters", {}
        ),
        "removed_variables": strategy_diagnostics.get("removed_variables", []),
        "strategy_details": strategy_diagnostics,
        "elapsed_ms": strategy_diagnostics.get("elapsed_ms", 0),
        "stop_reason": stop_reason,
        "completed_evaluations": completed_evaluations,
        "strategy_thresholds": request.strategy_thresholds.to_dict(),
        "model_refs": model_set.model_refs(),
    }


def _rejected_summary(candidates: list[CandidateResult]) -> dict[str, Any]:
    rejected = [item for item in candidates if item.trust_level == "REJECTED"]
    reasons: dict[str, int] = {}
    for candidate in rejected:
        failed = [
            item["name"] for item in candidate.hard_constraint_report
            if not item["satisfied"]
        ]
        reason = ",".join(failed) if failed else "variable_bounds"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"count": len(rejected), "reasons": reasons}


def _save_artifact(
    result: OptimizationResult,
    request: OptimizationRequest,
    output_dir: str | Path,
) -> None:
    root = Path(output_dir)
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", result.request_id)
    artifact_dir = root / f"optimization_{token}"
    suffix = 2
    while artifact_dir.exists():
        artifact_dir = root / f"optimization_{token}_{suffix}"
        suffix += 1
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "optimization_result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "request_snapshot.json").write_text(
        json.dumps(request.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_rows = _candidate_rows(result.selected_candidates)
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(
            artifact_dir / "selected_candidates.csv", index=False
        )
    (artifact_dir / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "warnings.json").write_text(
        json.dumps(result.warnings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    visualizations = build_optimization_visualizations(result)
    (artifact_dir / "visualization_datasets.json").write_text(
        json.dumps(visualizations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.artifact_ids.update({
        "artifact_dir": str(artifact_dir),
        "optimization_result": str(artifact_dir / "optimization_result.json"),
        "request_snapshot": str(artifact_dir / "request_snapshot.json"),
        "visualization_datasets": str(
            artifact_dir / "visualization_datasets.json"
        ),
        "request_hash": _request_hash(request),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": request.random_seed,
    })
    (artifact_dir / "optimization_result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _candidate_rows(candidates: list[CandidateResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "trust_level": candidate.trust_level,
            "applicability_domain": candidate.applicability_domain,
            "pareto_rank": candidate.pareto_rank,
            "soft_constraint_score": candidate.soft_constraint_score,
        }
        for key, value in candidate.values.items():
            row[f"value:{key}"] = value
        for key, value in candidate.predicted_values.items():
            row[f"predicted:{key}"] = value
        for key, value in candidate.objective_errors.items():
            row[f"error:{key}"] = value
        rows.append(row)
    return rows


def build_optimization_visualizations(
    result: OptimizationResult | dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        result = OptimizationResult(
            request_id=str(result.get("request_id", "")),
            status=str(result.get("status", "")),
            selected_candidates=[
                CandidateResult(**item)
                for item in result.get("selected_candidates", [])
            ],
            exploratory_candidates=[
                CandidateResult(**item)
                for item in result.get("exploratory_candidates", [])
            ],
            diagnostic_candidates=[
                CandidateResult(**item)
                for item in result.get("diagnostic_candidates", [])
            ],
            rejected_summary=dict(result.get("rejected_summary", {})),
            diagnostics=dict(result.get("diagnostics", {})),
            warnings=list(result.get("warnings", [])),
            artifact_ids=dict(result.get("artifact_ids", {})),
        )
    datasets: list[VisualizationDataset] = []
    candidate_rows = _candidate_rows(result.selected_candidates)
    if candidate_rows:
        columns = [
            VisualizationColumn(name=key, label=key)
            for key in candidate_rows[0]
        ]
        datasets.append(VisualizationDataset(
            dataset_id="selected_candidates",
            dataset_kind="table",
            title="Recommended candidates",
            columns=columns,
            records=candidate_rows,
            description="Selected optimization candidates with model traceability",
        ))
    constraint_records = _constraint_records(result.selected_candidates)
    if constraint_records:
        datasets.append(VisualizationDataset(
            dataset_id="constraint_report",
            dataset_kind="table",
            title="Constraint satisfaction",
            columns=[
                VisualizationColumn(name="candidate_id", label="Candidate"),
                VisualizationColumn(name="constraint", label="Constraint"),
                VisualizationColumn(name="satisfied", label="Satisfied"),
                VisualizationColumn(name="violation", label="Violation"),
            ],
            records=constraint_records,
        ))
    domain_counts = {
        domain: sum(
            candidate.applicability_domain == domain
            for candidate in result.selected_candidates
        )
        for domain in ["IN_DOMAIN", "EDGE", "OUT_OF_DOMAIN"]
    }
    datasets.append(VisualizationDataset(
        dataset_id="applicability_domain_distribution",
        dataset_kind="chart_with_table",
        title="Applicability domain distribution",
        chart_type="bar",
        x_field="domain",
        y_fields=["count"],
        columns=[
            VisualizationColumn(name="domain", label="Domain"),
            VisualizationColumn(name="count", label="Count", data_type="integer"),
        ],
        records=[
            {"domain": key, "count": value}
            for key, value in domain_counts.items()
        ],
    ))
    model_refs = result.diagnostics.get("model_refs", [])
    target_names = [item.get("target_name") for item in model_refs]
    if len(target_names) >= 2 and result.selected_candidates:
        first, second = target_names[:2]
        records = [
            {
                "candidate_id": item.candidate_id,
                f"objective:{first}": item.objective_values[first],
                f"objective:{second}": item.objective_values[second],
                "pareto_rank": item.pareto_rank,
            }
            for item in result.selected_candidates
            if first in item.objective_values and second in item.objective_values
        ]
        datasets.append(VisualizationDataset(
            dataset_id="pareto_front",
            dataset_kind="chart_with_table",
            title="Pareto front",
            chart_type="scatter",
            x_field=f"objective:{first}",
            y_fields=[f"objective:{second}"],
            series_field="pareto_rank",
            columns=[
                VisualizationColumn(name="candidate_id", label="Candidate"),
                VisualizationColumn(name=f"objective:{first}", label=str(first)),
                VisualizationColumn(name=f"objective:{second}", label=str(second)),
                VisualizationColumn(
                    name="pareto_rank", label="Pareto rank", data_type="integer"
                ),
            ],
            records=records,
        ))
    convergence = result.diagnostics.get("strategy_details", {}).get(
        "convergence_history", []
    )
    if convergence:
        records = [
            {"generation": index + 1, "mean_objective": float(value)}
            for index, value in enumerate(convergence)
        ]
        datasets.append(VisualizationDataset(
            dataset_id="convergence_history",
            dataset_kind="chart_with_table",
            title="Convergence history",
            chart_type="line",
            x_field="generation",
            y_fields=["mean_objective"],
            columns=[
                VisualizationColumn(
                    name="generation", label="Generation", data_type="integer"
                ),
                VisualizationColumn(
                    name="mean_objective", label="Mean objective error"
                ),
            ],
            records=records,
        ))
    acquisition_records = _acquisition_records(result.selected_candidates)
    if acquisition_records:
        datasets.append(VisualizationDataset(
            dataset_id="bo_acquisition",
            dataset_kind="table",
            title="BO acquisition",
            columns=[
                VisualizationColumn(name="candidate_id", label="Experiment candidate"),
                VisualizationColumn(name="acquisition_name", label="Acquisition"),
                VisualizationColumn(name="acquisition_value", label="Acquisition value"),
                VisualizationColumn(name="acquisition_mean", label="GP mean"),
                VisualizationColumn(name="acquisition_std", label="GP std"),
                VisualizationColumn(
                    name="distance_to_nearest_history", label="Distance to history"
                ),
            ],
            records=acquisition_records,
        ))
    return [item.to_dict() for item in datasets]


def _constraint_records(
    candidates: list[CandidateResult],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        for report in candidate.hard_constraint_report:
            records.append({
                "candidate_id": candidate.candidate_id,
                "constraint": report["name"],
                "satisfied": report["satisfied"],
                "violation": report["violation"],
            })
    return records


def _acquisition_records(
    candidates: list[CandidateResult],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.candidate_id,
            "acquisition_name": item.acquisition_name,
            "acquisition_value": item.acquisition_value,
            "acquisition_mean": item.acquisition_mean,
            "acquisition_std": item.acquisition_std,
            "distance_to_nearest_history": item.distance_to_nearest_history,
        }
        for item in candidates
        if item.acquisition_name is not None
    ]


def _request_id() -> str:
    return uuid.uuid4().hex


def _request_hash(request: OptimizationRequest) -> str:
    payload = json.dumps(
        request.to_dict(), ensure_ascii=False, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
