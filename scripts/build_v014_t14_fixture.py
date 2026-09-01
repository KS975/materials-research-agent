from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V0.1.4 T14 search-space fixtures."
    )
    parser.add_argument(
        "--output-dir",
        default=".runtime/v014/fixtures/t14",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    search_space = {
        "stage": "V0.1.4-T14_search_space",
        "project_id": 9014,
        "name": "t14_impact_design_space",
        "metadata": {
            "purpose": "工程验收 fixture，不是材料科学推荐",
            "target_metric": "冲击强度",
        },
        "variables": [
            {
                "name": "formula::ABS",
                "kind": "continuous",
                "min": 20.0,
                "max": 40.0,
                "step": 0.5,
                "unit": "%",
                "role": "formula",
            },
            {
                "name": "formula::PC",
                "kind": "continuous",
                "min": 45.0,
                "max": 70.0,
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
                "min": 220.0,
                "max": 260.0,
                "step": 1.0,
                "unit": "℃",
                "role": "process",
            },
            {
                "name": "process::螺杆转速",
                "kind": "integer",
                "min": 180,
                "max": 320,
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
                    {"variable": "formula::ABS", "weight": 1.0},
                    {"variable": "formula::PC", "weight": 1.0},
                    {"variable": "formula::增韧剂", "weight": 1.0},
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
                "message": "增韧剂高于推荐上限 15%，允许搜索但应惩罚",
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

    valid = {
        "sample_name": "T14_valid",
        "features": {
            "formula::ABS": 30.0,
            "formula::PC": 58.0,
            "formula::增韧剂": 12.0,
            "process::加工温度": 240.0,
            "process::螺杆转速": 250,
            "process::催化剂": "A",
        },
    }

    soft = {
        "sample_name": "T14_soft_violation",
        "features": {
            "formula::ABS": 30.0,
            "formula::PC": 54.0,
            "formula::增韧剂": 16.0,
            "process::加工温度": 240.0,
            "process::螺杆转速": 250,
            "process::催化剂": "A",
        },
    }

    hard = {
        "sample_name": "T14_hard_violation",
        "features": {
            "formula::ABS": 30.0,
            "formula::PC": 58.0,
            "formula::增韧剂": 12.0,
            "process::加工温度": 255.0,
            "process::螺杆转速": 250,
            "process::催化剂": "B",
        },
    }

    write_json(out / "search_space.json", search_space)
    write_json(out / "candidate_valid.json", valid)
    write_json(out / "candidate_soft.json", soft)
    write_json(out / "candidate_hard.json", hard)

    print("V0.1.4-T14 FIXTURE BUILDER")
    print(f"output_dir: {out}")
    print("expected:")
    print("- candidate_valid.json -> VALID")
    print("- candidate_soft.json -> VALID_WITH_SOFT_PENALTY")
    print("- candidate_hard.json -> INVALID")
    print()
    print("V0.1.4-T14 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
