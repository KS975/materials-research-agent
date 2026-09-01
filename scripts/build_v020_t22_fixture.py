from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V0.2-T22 dataset versioning fixture")
    parser.add_argument("--output-dir", default=".runtime/v020/fixtures/t22")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    columns = [
        "candidate_id", "project_id", "test_condition_signature",
        "source_campaign", "source_round",
        "formula::ABS", "formula::PC", "formula::增韧剂",
        "process::加工温度", "process::螺杆转速", "冲击强度",
    ]
    base_csv = out / "dataset_v001.csv"
    with base_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(1, 36):
            writer.writerow({
                "candidate_id": f"BASE_{i:03d}",
                "project_id": "9022",
                "test_condition_signature": "T22_STANDARD_23C",
                "source_campaign": "BASE_IMPORT",
                "source_round": "BASE",
                "formula::ABS": f"{30 + (i % 5):.1f}",
                "formula::PC": f"{57 - (i % 5):.1f}",
                "formula::增韧剂": "13",
                "process::加工温度": str(240 + (i % 7)),
                "process::螺杆转速": str(240 + (i % 6) * 10),
                "冲击强度": f"{45 + i * 0.18:.3f}",
            })

    campaign = {
        "campaign_id": "V020_T22_DEMO",
        "project_id": 9022,
        "name": "冲击强度数据集版本更新演示",
        "target_metrics": ["冲击强度"],
        "metadata": {"fixture": True, "purpose": "V0.2-T22 engineering acceptance"},
    }
    round_plan = {
        "planned_experiment_count": 5,
        "dataset_version": "dataset_v001",
        "model_versions": {"冲击强度": "model_v001"},
        "search_space_snapshot": {"version": "search_space_v001"},
        "constraints_snapshot": {"version": "constraints_v001"},
        "optimizer_config": {"engine": "GaussianProcess", "acquisition": "EI"},
        "source": "V0.1.4-T18",
    }
    planned = []
    for i in range(1, 6):
        planned.append({
            "candidate_id": f"V020_T22_EXP_{i:02d}",
            "required_metrics": ["冲击强度"],
            "expected_test_condition_signature": "T22_STANDARD_23C",
            "units": {"冲击强度": "kJ/m²"},
            "features": {
                "formula::ABS": 30.0 + i,
                "formula::PC": 57.0 - i,
                "formula::增韧剂": 13.0,
                "process::加工温度": 240 + i,
                "process::螺杆转速": 240 + i * 10,
            },
            "prediction_snapshot": {"冲击强度": {"value": 49.0 + i}},
        })
    results = [
        {"candidate_id":"V020_T22_EXP_01","status":"COMPLETED","test_condition_signature":"T22_STANDARD_23C","measurements":{"冲击强度":48.7},"units":{"冲击强度":"kJ/m²"}},
        {"candidate_id":"V020_T22_EXP_02","status":"FAILED","test_condition_signature":"T22_STANDARD_23C","measurements":{},"units":{},"failure_reason":"specimen broke"},
        {"candidate_id":"V020_T22_EXP_03","status":"INVALID","test_condition_signature":"T22_STANDARD_23C","measurements":{},"units":{},"failure_reason":"instrument invalid"},
        {"candidate_id":"V020_T22_EXP_04","status":"NOT_TESTED","test_condition_signature":"T22_STANDARD_23C","measurements":{},"units":{}},
        {"candidate_id":"V020_T22_EXP_05","status":"COMPLETED","test_condition_signature":"T22_STANDARD_23C","measurements":{"冲击强度":51.1},"units":{"冲击强度":"kJ/m²"}},
    ]
    write_json(out/"campaign_create.json", campaign)
    write_json(out/"round_plan.json", round_plan)
    write_json(out/"planned_experiments.json", planned)
    write_json(out/"results.json", results)

    print("V0.2-T22 FIXTURE BUILDER")
    print(f"base_dataset_csv: {base_csv}")
    print("base_rows: 35")
    print("training_eligible_new_rows: 2")
    print("excluded_nontraining_rows: 3")
    print("expected_child_rows: 37")
    print()
    print("V0.2-T22 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
