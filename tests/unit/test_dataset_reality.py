from data.dataset.reality import DatasetRealityAnalyzer


def _columns():
    return {
        1001: {"id": 1001, "name": "保温时间", "unit": "min"},
        2001: {
            "id": 2001,
            "name": "冲击强度",
            "unit": "kJ/m²",
            "belonging_column": "performance",
        },
    }


def _materials():
    return {
        501: {"id": 501, "name": "ABS", "unit": "%"},
        502: {"id": 502, "name": "PC", "unit": "%"},
    }


def test_reality_counts_sample_level_closure_and_target():
    rows = [
        {
            "id": 1,
            "name": "a",
            "recipes": {"R3-501": 20, "R3-502": 80},
            "craft_param": {"S1001": 60},
            "performances": {"P2001": 24},
            "service_performances": {},
            "conditions": {"standard": "A"},
        },
        {
            "id": 2,
            "name": "b",
            "recipes": {"R3-501": 30, "R3-502": 70},
            "craft_param": {"S1001": 70},
            "performances": {"P2001": 54},
            "service_performances": {},
            "conditions": {},
        },
    ]

    result = DatasetRealityAnalyzer().analyze(
        project_id=115,
        company_id="company-a",
        rows=rows,
        material_definitions=_materials(),
        column_definitions=_columns(),
        target_metric="冲击强度",
    )

    summary = result.report["summary"]
    assert summary["total_samples"] == 2
    assert summary["formula_present"] == 2
    assert summary["process_present"] == 2
    assert summary["target_present"] == 2
    assert summary["core_closed_formula_process_target"] == 2
    assert summary["strict_closed_with_conditions"] == 1
    assert result.report["target"]["numeric_count"] == 2
    assert result.report["target"]["min"] == 24
    assert result.report["target"]["max"] == 54


def test_reality_does_not_treat_missing_conditions_as_comparable():
    rows = [
        {
            "id": 1,
            "name": "a",
            "recipes": {"R3-501": 20},
            "craft_param": {"S1001": 60},
            "performances": {"P2001": 24},
            "service_performances": {},
            "conditions": {},
        },
        {
            "id": 2,
            "name": "b",
            "recipes": {"R3-501": 30},
            "craft_param": {"S1001": 70},
            "performances": {"P2001": 54},
            "service_performances": {},
            "conditions": {},
        },
    ]

    result = DatasetRealityAnalyzer().analyze(
        project_id=115,
        company_id="company-a",
        rows=rows,
        material_definitions=_materials(),
        column_definitions=_columns(),
        target_metric="冲击强度",
    )

    conditions = result.report["test_conditions"]
    assert conditions["present_count"] == 0
    assert conditions["missing_count"] == 2
    assert conditions["unique_nonempty_signatures"] == 0
    assert any("测试条件缺失" in warning for warning in result.report["warnings"])


def test_reality_reports_unresolved_dynamic_fields_and_missing_target():
    rows = [
        {
            "id": 1,
            "name": "a",
            "recipes": {"R3-999999": 20},
            "craft_param": {"S999999": 60},
            "performances": {},
            "service_performances": {},
            "conditions": {},
        }
    ]

    result = DatasetRealityAnalyzer().analyze(
        project_id=115,
        company_id="company-a",
        rows=rows,
        material_definitions={},
        column_definitions={},
        target_metric="冲击强度",
    )

    assert result.report["summary"]["target_present"] == 0
    assert result.report["unresolved_dynamic_fields"]["R3-999999"] == 1
    assert result.report["unresolved_dynamic_fields"]["S999999"] == 1
    assert any("目标性能" in warning for warning in result.report["warnings"])


def test_reality_detects_duplicate_name_and_duplicate_feature_rows():
    row = {
        "recipes": {"R3-501": 20},
        "craft_param": {"S1001": 60},
        "performances": {"P2001": 24},
        "service_performances": {},
        "conditions": {"standard": "A"},
    }
    rows = [
        {"id": 1, "name": "same-name", **row},
        {"id": 2, "name": "same-name", **row},
    ]

    result = DatasetRealityAnalyzer().analyze(
        project_id=115,
        company_id="company-a",
        rows=rows,
        material_definitions=_materials(),
        column_definitions=_columns(),
        target_metric="冲击强度",
    )

    dup = result.report["duplicates"]
    assert len(dup["duplicate_sample_name_groups"]) == 1
    assert len(dup["duplicate_formula_process_target_groups"]) == 1
