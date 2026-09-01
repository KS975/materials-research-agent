from pathlib import Path


def test_bo_conditions_are_visible_not_hidden_in_empty_details():
    source = Path("frontend/src/App.jsx").read_text(encoding="utf-8")
    assert "normalizedBOConditions" in source
    assert "推荐配方 / 工艺条件" in source
    assert "experiment_conditions" in source
    assert "条件数据缺失" in source
    assert "查看原始 features JSON" in source
    assert "<details><summary>下一轮实验条件</summary>" not in source


def test_bo_condition_styles_exist():
    source = Path("frontend/src/styles.css").read_text(encoding="utf-8")
    assert ".boConditionGrid" in source
    assert ".boConditionItem" in source
    assert ".boConditionMissing" in source
