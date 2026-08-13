from skills.analysis import AnalysisSkill


class FakeRegistry:
    def execute(self, name, **kwargs):
        assert name == "compare_samples"
        return {
            "status": "ok",
            "left_sample": {"id": 3811, "name": "trial_6"},
            "right_sample": {"id": 3809, "name": "trial_4"},
            "formula_diff": {
                "changed": [{"field": "ABS", "left": "18", "right": "33", "unit": "%"}],
                "same": [],
            },
            "process_diff": {
                "changed": [{"field": "挤出温度", "left": "25", "right": "42", "unit": "℃"}],
                "same": [],
            },
            "performance_diff": {
                "changed": [
                    {
                        "field": "冲击强度",
                        "left": "24",
                        "right": "54",
                        "unit": "kJ/m²",
                        "left_present": True,
                        "right_present": True,
                    }
                ],
                "same": [],
            },
            "service_performance_diff": {"changed": [], "same": []},
            "test_conditions": {
                "left": {},
                "right": {},
                "status": "missing_both",
                "same": None,
                "comparable": False,
            },
            "evidence": [{"source": "eln_sample", "record_id": 3811}],
            "warnings": [],
        }


def test_t03_builds_fact_hypothesis_gap_structure():
    skill = AnalysisSkill(FakeRegistry())
    result = skill.execute(
        "compare_samples",
        {
            "left_identifier": "3811",
            "right_identifier": "3809",
            "target_metric": "冲击强度",
            "direction": "低",
        },
        ctx=object(),
    )

    assert result["status"] == "ok"
    assert result["facts"]["target_performance"]["left"] == "24"
    assert result["facts"]["target_performance"]["right"] == "54"
    assert result["facts"]["numeric_difference"]["left_minus_right"] == "-30"
    assert result["facts"]["numeric_difference"]["relative_to_right_percent"] == "-55.56"
    assert result["hypotheses"]
    assert result["evidence_gaps"]
    assert "不能" in result["conclusion_limit"]
