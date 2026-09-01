from optimization.search_space import load_search_space


def _space():
    return load_search_space(
        {
            "stage": "V0.1.4-T14_search_space",
            "project_id": 1,
            "name": "unit",
            "variables": [
                {
                    "name": "formula::A",
                    "kind": "continuous",
                    "min": 0,
                    "max": 100,
                    "step": 1,
                },
                {
                    "name": "formula::B",
                    "kind": "continuous",
                    "min": 0,
                    "max": 100,
                    "step": 1,
                },
                {
                    "name": "process::mode",
                    "kind": "categorical",
                    "choices": ["X", "Y"],
                },
                {
                    "name": "process::rpm",
                    "kind": "integer",
                    "min": 100,
                    "max": 300,
                    "step": 10,
                },
            ],
            "constraints": [
                {
                    "id": "sum100",
                    "type": "weighted_sum",
                    "severity": "HARD",
                    "terms": [
                        {"variable": "formula::A"},
                        {"variable": "formula::B"},
                    ],
                    "operator": "==",
                    "value": 100,
                    "tolerance": 0.1,
                },
                {
                    "id": "a_soft",
                    "type": "scalar",
                    "severity": "SOFT",
                    "variable": "formula::A",
                    "operator": "<=",
                    "value": 60,
                    "weight": 2,
                },
                {
                    "id": "forbid",
                    "type": "forbidden_combination",
                    "severity": "HARD",
                    "clauses": [
                        {
                            "variable": "process::mode",
                            "operator": "==",
                            "value": "Y",
                        },
                        {
                            "variable": "process::rpm",
                            "operator": ">",
                            "value": 250,
                        },
                    ],
                },
            ],
        }
    )


def test_valid_candidate():
    report = _space().validate_candidate(
        {
            "features": {
                "formula::A": 50,
                "formula::B": 50,
                "process::mode": "X",
                "process::rpm": 200,
            }
        }
    )
    assert report["status"] == "VALID"
    assert report["hard_valid"] is True
    assert report["soft_penalty"] == 0


def test_soft_violation_does_not_invalidate_candidate():
    report = _space().validate_candidate(
        {
            "features": {
                "formula::A": 70,
                "formula::B": 30,
                "process::mode": "X",
                "process::rpm": 200,
            }
        }
    )
    assert report["status"] == "VALID_WITH_SOFT_PENALTY"
    assert report["hard_valid"] is True
    assert report["soft_penalty"] > 0
    assert len(report["soft_violations"]) == 1


def test_hard_weighted_sum_violation_invalidates_candidate():
    report = _space().validate_candidate(
        {
            "features": {
                "formula::A": 40,
                "formula::B": 40,
                "process::mode": "X",
                "process::rpm": 200,
            }
        }
    )
    assert report["status"] == "INVALID"
    assert report["hard_valid"] is False
    assert report["hard_violations"][0]["constraint_id"] == "sum100"


def test_forbidden_combination_invalidates_candidate():
    report = _space().validate_candidate(
        {
            "features": {
                "formula::A": 50,
                "formula::B": 50,
                "process::mode": "Y",
                "process::rpm": 280,
            }
        }
    )
    assert report["status"] == "INVALID"
    ids = {x["constraint_id"] for x in report["hard_violations"]}
    assert "forbid" in ids


def test_variable_bounds_and_step_are_enforced():
    report = _space().validate_candidate(
        {
            "features": {
                "formula::A": 50,
                "formula::B": 50,
                "process::mode": "X",
                "process::rpm": 205,
            }
        }
    )
    assert report["status"] == "INVALID"
    assert report["variable_errors"][0]["type"] == "off_step"
