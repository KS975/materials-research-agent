from runtime.optimization_shadow import compare_optimization_routes, shadow_failure


def _legacy():
    return {
        "status": "SUCCESS",
        "generation": {"generated_count": 100},
        "counts": {
            "generated_hard_valid": 80,
            "qualified_all_targets": 4,
            "pareto_front": 5,
        },
        "design_cards": [
            {"candidate_id": "a", "values": {"x": 1}, "applicability_domain": {"status": "IN_DOMAIN"}},
            {"candidate_id": "b", "values": {"x": 2}, "applicability_domain": {"status": "IN_DOMAIN"}},
            {"candidate_id": "c", "values": {"x": 3}, "applicability_domain": {"status": "IN_DOMAIN"}},
            {"candidate_id": "d", "values": {"x": 4}, "applicability_domain": {"status": "EDGE"}},
        ],
    }


def _engine():
    return {
        "status": "OK",
        "result": {
            "status": "COMPLETE",
            "selected_candidates": [
                {"candidate_id": "e", "values": {"x": 5}, "objective_errors": {"impact": 0}, "applicability_domain": "IN_DOMAIN"},
                {"candidate_id": "f", "values": {"x": 6}, "objective_errors": {"impact": 0}, "applicability_domain": "IN_DOMAIN"},
                {"candidate_id": "g", "values": {"x": 7}, "objective_errors": {"impact": 0}, "applicability_domain": "IN_DOMAIN"},
                {"candidate_id": "h", "values": {"x": 8}, "objective_errors": {"impact": 0}, "applicability_domain": "IN_DOMAIN"},
            ],
            "diagnostics": {
                "generated_count": 100,
                "hard_feasible_count": 82,
                "elapsed_ms": 2000,
            },
        },
    }


def test_shadow_comparison_reports_metrics_and_ready_decision():
    comparison = compare_optimization_routes(_legacy(), _engine())
    assert comparison["decision"] == "READY"
    assert comparison["rollback_route"] == "legacy"
    assert comparison["metrics"]["legacy"]["target_hit_rate"] == 1.0
    assert comparison["metrics"]["engine"]["in_domain_rate"] == 1.0


def test_shadow_failure_is_fail_closed():
    comparison = shadow_failure(RuntimeError("legacy fixture unavailable"))
    assert comparison["decision"] == "NOT_READY"
    assert comparison["rollback_route"] == "legacy"
