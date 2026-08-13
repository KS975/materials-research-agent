from agent.tools import MaterialsTools


def test_diff_distinguishes_same_and_changed():
    left = [
        {"raw_key": "R3-1", "name": "A", "value": "10", "unit": "%"},
        {"raw_key": "R3-2", "name": "B", "value": "20", "unit": "%"},
    ]
    right = [
        {"raw_key": "R3-1", "name": "A", "value": "10", "unit": "%"},
        {"raw_key": "R3-2", "name": "B", "value": "25", "unit": "%"},
    ]
    diff = MaterialsTools._diff_fields(left, right)
    assert [x["field"] for x in diff["same"]] == ["A"]
    assert [x["field"] for x in diff["changed"]] == ["B"]
