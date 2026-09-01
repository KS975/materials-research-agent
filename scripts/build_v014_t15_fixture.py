from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V0.1.4-T15 candidate-generation search space."
    )
    parser.add_argument(
        "--output-dir",
        default=".runtime/v014/fixtures/t15",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Deliberately overlaps but slightly exceeds the T10 training domain.
    # This lets T15 verify that generated candidates are split into
    # IN_DOMAIN / BORDERLINE / OUT_OF_DOMAIN instead of blindly trusted.
    search_space = {
        "stage": "V0.1.4-T14_search_space",
        "project_id": 9010,
        "name": "t15_candidate_generation_space",
        "metadata": {
            "purpose": "T15 工程验收 fixture，不是材料科学推荐",
            "target_metric": "冲击强度",
        },
        "variables": [
            {
                "name": "formula::ABS",
                "kind": "continuous",
                "min": 20.0,
                "max": 42.0,
                "step": 0.5,
                "unit": "%",
                "role": "formula",
            },
            {
                "name": "formula::PC",
                "kind": "continuous",
                "min": 45.0,
                "max": 72.0,
                "step": 0.5,
                "unit": "%",
                "role": "formula",
            },
            {
                "name": "formula::增韧剂",
                "kind": "continuous",
                "min": 5.0,
                "max": 20.0,
                "step": 0.5,
                "unit": "%",
                "role": "formula",
            },
            {
                "name": "process::加工温度",
                "kind": "continuous",
                "min": 218.0,
                "max": 265.0,
                "step": 1.0,
                "unit": "℃",
                "role": "process",
            },
            {
                "name": "process::螺杆转速",
                "kind": "integer",
                "min": 170,
                "max": 330,
                "step": 10,
                "unit": "rpm",
                "role": "process",
            },
            {
                "name": "process::催化剂",
                "kind": "categorical",
                "choices": ["NONE", "A", "B"],
                "role": "process",
            },
        ],
        "constraints": [
            {
                "id": "formula_sum_100",
                "type": "weighted_sum",
                "severity": "HARD",
                "terms": [
                    {"variable": "formula::ABS"},
                    {"variable": "formula::PC"},
                    {"variable": "formula::增韧剂"},
                ],
                "operator": "==",
                "value": 100.0,
                "tolerance": 0.5,
                "message": "主配方总和必须约等于 100%",
            },
            {
                "id": "toughener_recommended_max",
                "type": "scalar",
                "severity": "SOFT",
                "variable": "formula::增韧剂",
                "operator": "<=",
                "value": 15.0,
                "weight": 2.0,
                "message": "增韧剂高于推荐上限 15%，允许生成但增加软惩罚",
            },
            {
                "id": "forbid_hot_catalyst_b",
                "type": "forbidden_combination",
                "severity": "HARD",
                "clauses": [
                    {
                        "variable": "process::加工温度",
                        "operator": ">",
                        "value": 250.0,
                    },
                    {
                        "variable": "process::催化剂",
                        "operator": "==",
                        "value": "B",
                    },
                ],
                "message": "加工温度 > 250℃ 时禁止使用催化剂 B",
            },
        ],
    }

    path = out / "search_space.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    print("V0.1.4-T15 FIXTURE BUILDER")
    print(f"search_space_json: {path}")
    print()
    print(
        "NOTE: 该搜索空间故意略微超出 T10 训练域，"
        "用于验证 Applicability Domain 过滤。"
    )
    print()
    print("V0.1.4-T15 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
