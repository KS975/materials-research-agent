from __future__ import annotations

from collections import Counter
from typing import Any


DEFAULT_SWITCH_THRESHOLD = {
    "max_exception_rate": 0.0,
    "max_target_hit_rate_drop": 0.05,
    "max_hard_feasible_rate_drop": 0.05,
    "max_in_domain_rate_drop": 0.05,
    "max_pareto_diversity_drop": 0.20,
    "max_duration_multiplier": 3.0,
}


def _payload(report: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(report or {})
    return dict(data.get("result") or data)


def _candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("selected_candidates", "design_cards", "next_experiments"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _target_hit_count(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> int:
    counts = dict(payload.get("counts") or {})
    if isinstance(counts.get("qualified_all_targets"), int):
        return int(counts["qualified_all_targets"])
    explicit = sum(
        1
        for item in candidates
        if item.get("all_targets_satisfied") is True
        or item.get("qualified_all_targets") is True
    )
    if explicit:
        return explicit
    return sum(
        1
        for item in candidates
        if isinstance(item.get("objective_errors"), dict)
        and item["objective_errors"]
        and all(abs(float(value)) <= 1e-6 for value in item["objective_errors"].values())
    )


def _domain_distribution(candidates: list[dict[str, Any]]) -> dict[str, int]:
    values=[]
    for item in candidates:
        domain=item.get("applicability_domain")
        if isinstance(domain,dict):domain=domain.get("status")
        values.append(str(domain or "UNKNOWN").upper())
    return dict(sorted(Counter(values).items()))


def _pareto_diversity(candidates: list[dict[str, Any]]) -> int:
    signatures=set()
    for item in candidates:
        values=item.get("values") or item.get("features") or item.get("predictions")
        if isinstance(values,dict):
            signatures.add(tuple(sorted((str(k),str(v)) for k,v in values.items())))
        else:
            signatures.add(str(item.get("candidate_id") or item.get("recommendation_rank") or len(signatures)))
    return len(signatures)


def summarize_optimization_route(report: dict[str, Any] | None) -> dict[str, Any]:
    payload=_payload(report)
    candidates=_candidates(payload)
    counts=dict(payload.get("counts") or {})
    diagnostics=dict(payload.get("diagnostics") or {})
    generation=dict(payload.get("generation") or {})
    generated=int(
        diagnostics.get("generated_count")
        or generation.get("generated_count")
        or counts.get("generated")
        or counts.get("generated_hard_valid")
        or len(candidates)
        or 0
    )
    hard_feasible=int(
        diagnostics.get("hard_feasible_count")
        or counts.get("generated_hard_valid")
        or len(candidates)
    )
    domains=_domain_distribution(candidates)
    in_domain=domains.get("IN_DOMAIN",0)
    return {
        "status":str(payload.get("status") or report.get("status") or "UNKNOWN"),
        "selected_count":len(candidates),
        "generated_count":generated,
        "hard_feasible_count":hard_feasible,
        "hard_feasible_rate":_rate(hard_feasible,generated),
        "target_hit_count":_target_hit_count(payload,candidates),
        "target_hit_rate":_rate(_target_hit_count(payload,candidates),len(candidates)),
        "pareto_diversity":_pareto_diversity(candidates),
        "applicability_domain_distribution":domains,
        "in_domain_rate":_rate(in_domain,len(candidates)),
        "duration_ms":int(diagnostics.get("elapsed_ms") or payload.get("elapsed_ms") or 0),
        "exception_rate":1.0 if str(payload.get("status") or "").upper() in {"ERROR","FAILED"} else 0.0,
    }


def compare_optimization_routes(
    legacy_report: dict[str, Any] | None,
    engine_report: dict[str, Any] | None,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits={**DEFAULT_SWITCH_THRESHOLD,**dict(thresholds or {})}
    legacy=summarize_optimization_route(legacy_report)
    engine=summarize_optimization_route(engine_report)
    checks={
        "no_engine_exception":engine["exception_rate"]<=limits["max_exception_rate"],
        "target_hit_rate":engine["target_hit_rate"]>=legacy["target_hit_rate"]-limits["max_target_hit_rate_drop"],
        "hard_feasible_rate":engine["hard_feasible_rate"]>=legacy["hard_feasible_rate"]-limits["max_hard_feasible_rate_drop"],
        "in_domain_rate":engine["in_domain_rate"]>=legacy["in_domain_rate"]-limits["max_in_domain_rate_drop"],
        "pareto_diversity":engine["pareto_diversity"]>=legacy["pareto_diversity"]*(1-limits["max_pareto_diversity_drop"]),
        "duration":engine["duration_ms"]<=max(
            legacy["duration_ms"]*limits["max_duration_multiplier"],
            legacy["duration_ms"]+5000,
        ),
    }
    return {
        "schema_version":1,
        "mode":"shadow",
        "primary_route":"legacy",
        "candidate_route":"engine",
        "rollback_route":"legacy",
        "decision":"READY" if all(checks.values()) else "NOT_READY",
        "checks":checks,
        "thresholds":limits,
        "metrics":{"legacy":legacy,"engine":engine},
    }


def shadow_failure(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version":1,
        "mode":"shadow",
        "primary_route":"legacy",
        "candidate_route":"engine",
        "rollback_route":"legacy",
        "decision":"NOT_READY",
        "checks":{},
        "error":{"type":type(error).__name__,"message":str(error)[:1000]},
    }
