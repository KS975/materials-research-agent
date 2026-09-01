from optimization.search_space import load_search_space
from runtime.v014_ui import _build_experiment_conditions


def test_bo_conditions_preserve_features_units_and_groups():
    space = load_search_space({
        "stage": "V0.1.4-T14_search_space",
        "project_id": 9018,
        "name": "test",
        "variables": [
            {
                "name": "formula::ABS",
                "kind": "continuous",
                "min": 20,
                "max": 40,
                "unit": "%",
            },
            {
                "name": "process::加工温度",
                "kind": "integer",
                "min": 220,
                "max": 260,
                "unit": "℃",
            },
        ],
        "constraints": [],
    })

    rows = _build_experiment_conditions(
        space,
        {
            "formula::ABS": 33.0,
            "process::加工温度": 246,
        },
    )

    assert rows == [
        {
            "name": "formula::ABS",
            "group": "配方",
            "label": "ABS",
            "value": 33.0,
            "unit": "%",
            "kind": "continuous",
        },
        {
            "name": "process::加工温度",
            "group": "工艺",
            "label": "加工温度",
            "value": 246,
            "unit": "℃",
            "kind": "integer",
        },
    ]


def test_unknown_future_feature_remains_visible():
    space = load_search_space({
        "stage": "V0.1.4-T14_search_space",
        "project_id": 9018,
        "name": "test",
        "variables": [
            {
                "name": "formula::ABS",
                "kind": "continuous",
                "min": 20,
                "max": 40,
            },
        ],
        "constraints": [],
    })

    rows = _build_experiment_conditions(
        space,
        {
            "formula::ABS": 33.0,
            "process::未来参数": 123,
        },
    )

    assert len(rows) == 2
    assert rows[1]["label"] == "未来参数"
    assert rows[1]["group"] == "工艺"
    assert rows[1]["value"] == 123
