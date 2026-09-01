import pytest

from optimization.candidate_generation import CandidateGenerator
from optimization.inverse_design import (
    InverseDesignError,
    parse_inverse_design_request,
    parse_inverse_design_text,
)
from optimization.search_space import load_search_space


def test_parse_structured_inverse_design_request():
    request = parse_inverse_design_request(
        {
            "stage": "V0.1.4-T17_inverse_design_request",
            "project_id": 9016,
            "request_name": "unit",
            "objectives": [
                {
                    "metric": "冲击强度",
                    "direction": "maximize",
                    "threshold": {
                        "operator": ">=",
                        "value": 43,
                    },
                },
                {
                    "metric": "MFR",
                    "direction": "maximize",
                    "threshold": {
                        "operator": ">=",
                        "value": 8.5,
                    },
                },
            ],
            "recommendation_count": 5,
        }
    )

    assert request.project_id == 9016
    assert request.recommendation_count == 5
    assert [x.metric for x in request.objectives] == [
        "冲击强度",
        "MFR",
    ]


def test_parse_simple_chinese_request_text():
    request = parse_inverse_design_text(
        "冲击强度 >= 43、MFR >= 8.5，推荐5组方案",
        project_id=9016,
    )

    assert request.source == "text"
    assert request.recommendation_count == 5
    assert request.objectives[0].metric == "冲击强度"
    assert request.objectives[0].threshold_value == 43
    assert request.objectives[1].metric == "MFR"
    assert request.objectives[1].threshold_value == 8.5


@pytest.mark.parametrize(
    "message",
    [
        "冲击强度 >= 43、MFR >= 8.5，推荐5组方案",
        "给我推荐五组冲击强度>=43，MFR>=8.5的方案",
        "给我推荐五组里冲击强度>=43，MFR>=8.5的方案",
        "给我推荐五组project 9016里冲击强度>=43，MFR>=8.5的方案",
    ],
)
def test_schema_bound_parser_handles_inverse_design_prompt_prefixes(message):
    request = parse_inverse_design_text(
        message,
        project_id=9016,
        allowed_metrics=["冲击强度", "MFR"],
    )

    assert [x.metric for x in request.objectives] == ["冲击强度", "MFR"]
    assert request.recommendation_count == 5


def test_schema_bound_parser_prefers_longest_metric_name():
    request = parse_inverse_design_text(
        "悬臂梁冲击强度 >= 20，推荐3组方案",
        project_id=9016,
        allowed_metrics=["冲击强度", "悬臂梁冲击强度"],
    )

    assert request.objectives[0].metric == "悬臂梁冲击强度"
    assert request.recommendation_count == 3


def test_schema_bound_parser_rejects_unknown_metric_before_path_resolution():
    with pytest.raises(InverseDesignError, match="可用指标：冲击强度、MFR"):
        parse_inverse_design_text(
            "成本 <= 20，推荐5组方案",
            project_id=9016,
            allowed_metrics=["冲击强度", "MFR"],
        )


def test_parser_understands_non_default_chinese_recommendation_count():
    request = parse_inverse_design_text(
        "冲击强度 >= 43，给我推荐三组方案",
        project_id=9016,
        allowed_metrics=["冲击强度"],
    )

    assert request.recommendation_count == 3


def test_rejects_threshold_direction_mismatch():
    with pytest.raises(InverseDesignError):
        parse_inverse_design_request(
            {
                "stage": "V0.1.4-T17_inverse_design_request",
                "project_id": 1,
                "objectives": [
                    {
                        "metric": "impact",
                        "direction": "minimize",
                        "threshold": {
                            "operator": ">=",
                            "value": 10,
                        },
                    }
                ],
            }
        )


def test_candidate_generator_supports_stage_specific_prefix():
    space = load_search_space(
        {
            "stage": "V0.1.4-T14_search_space",
            "project_id": 1,
            "name": "prefix_test",
            "variables": [
                {
                    "name": "x",
                    "kind": "integer",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }
            ],
            "constraints": [],
        }
    )

    result = CandidateGenerator(
        space,
        random_state=42,
        id_prefix="V014_T17",
    ).generate(
        candidate_count=3,
        max_attempts=100,
    )

    assert result["generation_complete"] is True
    assert all(
        row["candidate_id"].startswith("V014_T17_")
        for row in result["candidates"]
    )
