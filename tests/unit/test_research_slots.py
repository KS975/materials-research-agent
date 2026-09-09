from __future__ import annotations

from agent.research_slots import (
    extract_competitor_name,
    extract_identifier,
    extract_phenomenon,
    extract_substitution_materials,
    normalize_research_slots,
    parse_target_filters,
)


def test_material_usage_slots_have_backend_fallback_extraction():
    args = normalize_research_slots(
        message="查询 P507+煤油 的原料使用效果，并结合历史资料。",
        args={},
    )
    assert args["material_name"] == "P507+煤油"

    args = normalize_research_slots(
        message="使用水的样品性能怎么样？结合历史资料判断。",
        args={},
    )
    assert args["material_name"] == "水"


def test_direct_substitution_phrase_extracts_original_and_replacement():
    assert extract_substitution_materials(
        "查找 P507+煤油 替代 水的历史记录，并结合历史资料。"
    ) == ("P507+煤油", "水")


def test_reversed_substitution_phrase_removes_non_material_suffixes():
    args = normalize_research_slots(
        message="有没有用水替换 P507+煤油 的配方记录？结合历史案例说明。",
        args={},
    )
    assert args["original_material"] == "P507+煤油"
    assert args["replacement_material"] == "水"


def test_competitor_name_is_extracted_only_for_explicit_product_identifier():
    assert extract_competitor_name("与竞品 ABS 的性能差距是多少？") == "ABS"
    assert extract_competitor_name("历史上哪些路线最接近竞品 PC/ABS？") == "PC/ABS"
    assert extract_competitor_name("查竞品对标的内部样品和资料。") is None


def test_failure_phenomenon_has_backend_fallback_extraction():
    assert extract_phenomenon("查粘接失效类似案例，并结合内部资料。") == "粘接失效"
    assert extract_phenomenon("查找开裂异常案例，并结合历史资料。") == "开裂"


def test_explicit_reference_identifier_has_backend_fallback_extraction():
    assert extract_identifier("找与 EXP-128 工艺最像的样品，并结合历史资料。") == "EXP-128"


def test_target_filters_fallback_ignores_threshold_as_sample_id():
    filters = parse_target_filters(
        "新项目冷启动：目标密度差大于 150，请结合历史资料给第一轮方案。"
    )
    assert filters == [
        {
            "section": "performance",
            "field": "密度差",
            "operator": "gt",
            "value": 150,
        }
    ]


def test_filters_are_normalized_before_scenario_classification():
    args = normalize_research_slots(
        message="查找持液量在 0.05 到 0.1 之间的样品，并结合历史资料。",
        args={},
    )
    assert args["filters"] == [
        {
            "section": "performance",
            "field": "持液量",
            "operator": "between",
            "values": [0.05, 0.1],
        }
    ]
