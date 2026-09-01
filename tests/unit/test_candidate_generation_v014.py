from optimization.candidate_generation import CandidateGenerator
from optimization.search_space import load_search_space


def _space():
    return load_search_space(
        {
            "stage": "V0.1.4-T14_search_space",
            "project_id": 1,
            "name": "generator_unit",
            "variables": [
                {
                    "name": "formula::A",
                    "kind": "continuous",
                    "min": 20,
                    "max": 60,
                    "step": 1,
                },
                {
                    "name": "formula::B",
                    "kind": "continuous",
                    "min": 40,
                    "max": 80,
                    "step": 1,
                },
                {
                    "name": "process::rpm",
                    "kind": "integer",
                    "min": 100,
                    "max": 300,
                    "step": 10,
                },
                {
                    "name": "process::mode",
                    "kind": "categorical",
                    "choices": ["X", "Y"],
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
                    "tolerance": 0.01,
                },
                {
                    "id": "soft_a",
                    "type": "scalar",
                    "severity": "SOFT",
                    "variable": "formula::A",
                    "operator": "<=",
                    "value": 50,
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


def test_generator_returns_requested_unique_hard_valid_candidates():
    space = _space()
    result = CandidateGenerator(space, random_state=42).generate(
        candidate_count=25,
        max_attempts=2000,
    )

    assert result["generation_complete"] is True
    assert result["generated_count"] == 25

    keys = set()
    for candidate in result["candidates"]:
        report = space.validate_candidate(
            {"features": candidate["features"]}
        )
        assert report["hard_valid"] is True
        key = tuple(sorted(candidate["features"].items()))
        assert key not in keys
        keys.add(key)


def test_generator_repairs_weighted_sum_equality():
    space = _space()
    result = CandidateGenerator(space, random_state=7).generate(
        candidate_count=10,
        max_attempts=1000,
    )

    for candidate in result["candidates"]:
        features = candidate["features"]
        assert abs(
            features["formula::A"]
            + features["formula::B"]
            - 100
        ) <= 0.01
