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
        description="Build V0.1.4-T17 inverse-design acceptance requests."
    )
    parser.add_argument(
        "--output-dir",
        default=".runtime/v014/fixtures/t17",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feasible = {
        "stage": "V0.1.4-T17_inverse_design_request",
        "project_id": 9016,
        "request_name": "impact_mfr_feasible",
        "objectives": [
            {
                "metric": "冲击强度",
                "direction": "maximize",
                "threshold": {
                    "operator": ">=",
                    "value": 43.0,
                },
                "weight": 1.0,
            },
            {
                "metric": "MFR",
                "direction": "maximize",
                "threshold": {
                    "operator": ">=",
                    "value": 8.5,
                },
                "weight": 1.0,
            },
        ],
        "recommendation_count": 5,
        "candidate_count": 600,
        "max_attempts": 80000,
        "random_state": 42,
        "soft_penalty_weight": 0.20,
        "diversity_weight": 0.35,
    }

    impossible = {
        "stage": "V0.1.4-T17_inverse_design_request",
        "project_id": 9016,
        "request_name": "impact_mfr_impossible",
        "objectives": [
            {
                "metric": "冲击强度",
                "direction": "maximize",
                "threshold": {
                    "operator": ">=",
                    "value": 80.0,
                },
                "weight": 1.0,
            },
            {
                "metric": "MFR",
                "direction": "maximize",
                "threshold": {
                    "operator": ">=",
                    "value": 25.0,
                },
                "weight": 1.0,
            },
        ],
        "recommendation_count": 5,
        "candidate_count": 500,
        "max_attempts": 80000,
        "random_state": 42,
        "soft_penalty_weight": 0.20,
        "diversity_weight": 0.35,
    }

    feasible_path = out / "request_feasible.json"
    impossible_path = out / "request_impossible.json"
    text_path = out / "request_text.txt"

    write_json(feasible_path, feasible)
    write_json(impossible_path, impossible)
    text_path.write_text(
        "冲击强度 >= 43、MFR >= 8.5，推荐5组方案",
        encoding="utf-8",
    )

    print("V0.1.4-T17 FIXTURE BUILDER")
    print(f"feasible_request: {feasible_path}")
    print(f"impossible_request: {impossible_path}")
    print(f"request_text: {text_path}")
    print()
    print("EXPECTED")
    print("- feasible request -> SUCCESS + 5 designs")
    print("- impossible request -> NO_FEASIBLE_DESIGN + 0 fabricated designs")
    print()
    print("V0.1.4-T17 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
