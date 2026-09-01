from optimization.pareto import (
    ObjectiveSpec,
    diverse_select,
    dominates,
    non_dominated_sort,
    pareto_front_indices,
    threshold_pass,
)


OBJECTIVES = [
    ObjectiveSpec("impact", "maximize"),
    ObjectiveSpec("mfr", "maximize"),
]


def test_dominance():
    assert dominates(
        {"impact": 10, "mfr": 8},
        {"impact": 9, "mfr": 8},
        OBJECTIVES,
    )
    assert not dominates(
        {"impact": 10, "mfr": 7},
        {"impact": 9, "mfr": 8},
        OBJECTIVES,
    )


def test_pareto_front_keeps_tradeoff_points():
    rows = [
        {"impact": 10, "mfr": 5},
        {"impact": 8, "mfr": 8},
        {"impact": 5, "mfr": 10},
        {"impact": 6, "mfr": 6},
    ]
    front = pareto_front_indices(rows, OBJECTIVES)
    assert set(front) == {0, 1, 2}


def test_non_dominated_sort():
    rows = [
        {"impact": 10, "mfr": 10},
        {"impact": 9, "mfr": 9},
        {"impact": 8, "mfr": 8},
    ]
    assert non_dominated_sort(rows, OBJECTIVES) == [1, 2, 3]


def test_threshold_pass():
    obj = ObjectiveSpec(
        "impact",
        "maximize",
        threshold_operator=">=",
        threshold_value=10,
    )
    assert threshold_pass(10, obj)
    assert not threshold_pass(9.9, obj)


def test_diverse_select_returns_unique_candidates():
    candidates = [
        {
            "candidate_id": "A",
            "adjusted_utility": 1.0,
            "features": {"x": 0.0, "y": 0.0},
        },
        {
            "candidate_id": "B",
            "adjusted_utility": 0.95,
            "features": {"x": 0.1, "y": 0.1},
        },
        {
            "candidate_id": "C",
            "adjusted_utility": 0.90,
            "features": {"x": 10.0, "y": 10.0},
        },
    ]

    selected = diverse_select(
        candidates,
        count=2,
        feature_columns=["x", "y"],
        diversity_weight=0.8,
    )

    ids = [item["candidate_id"] for item in selected]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert "A" in ids
    assert "C" in ids
