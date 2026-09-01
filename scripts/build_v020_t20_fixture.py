from __future__ import annotations
import argparse, json
from pathlib import Path

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".runtime/v020/fixtures/t20")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    campaign = {
        "campaign_id": "V020_T20_DEMO",
        "project_id": 9018,
        "name": "冲击强度实验结果回流演示",
        "target_metrics": ["冲击强度"],
        "metadata": {"fixture": True, "purpose": "V0.2-T20 engineering acceptance"},
    }
    round_plan = {
        "planned_experiment_count": 5,
        "dataset_version": "dataset_v001",
        "model_versions": {"冲击强度": "model_v001"},
        "search_space_snapshot": {"version": "search_space_v001"},
        "constraints_snapshot": {"version": "constraints_v001"},
        "optimizer_config": {"engine": "GaussianProcess", "acquisition": "EI", "batch_strategy": "kriging_believer"},
        "source": "V0.1.4-T18",
    }

    predictions = [50.332931,49.622512,49.749811,49.852930,49.585865]
    planned = []
    for i, pred in enumerate(predictions, start=1):
        planned.append({
            "candidate_id": f"V020_T20_EXP_{i:02d}",
            "required_metrics": ["冲击强度"],
            "expected_test_condition_signature": "T20_STANDARD_23C",
            "units": {"冲击强度": "kJ/m²"},
            "features": {
                "formula::ABS": 30.0+i,
                "formula::PC": 57.0-i,
                "formula::增韧剂": 13.0,
                "process::加工温度": 240+i,
                "process::螺杆转速": 240+i*10,
            },
            "prediction_snapshot": {
                "冲击强度": {"value": pred, "source": "V0.1.4-T18_GP_posterior_mean"}
            },
        })

    results = {
        "result_01_completed.json": {
            "candidate_id":"V020_T20_EXP_01","status":"COMPLETED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{"冲击强度":48.7},"units":{"冲击强度":"kJ/m²"},
            "notes":"normal completed experiment",
        },
        "result_02_failed.json": {
            "candidate_id":"V020_T20_EXP_02","status":"FAILED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{},"units":{},"failure_reason":"试样在制备阶段破损",
        },
        "result_03_invalid.json": {
            "candidate_id":"V020_T20_EXP_03","status":"INVALID",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{},"units":{},"failure_reason":"测试仪器校准异常",
        },
        "result_04_not_tested.json": {
            "candidate_id":"V020_T20_EXP_04","status":"NOT_TESTED",
            "test_condition_signature":"T20_STANDARD_23C","measurements":{},"units":{},
            "notes":"本轮预算不足，明确标记未测试",
        },
        "result_05_completed.json": {
            "candidate_id":"V020_T20_EXP_05","status":"COMPLETED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{"冲击强度":51.1},"units":{"冲击强度":"kJ/m²"},
        },
        "invalid_unknown_candidate.json": {
            "candidate_id":"NOT_IN_THIS_ROUND","status":"COMPLETED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{"冲击强度":50.0},"units":{"冲击强度":"kJ/m²"},
        },
        "invalid_non_numeric.json": {
            "candidate_id":"V020_T20_EXP_05","status":"COMPLETED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{"冲击强度":"high"},"units":{"冲击强度":"kJ/m²"},
        },
        "invalid_condition_mismatch.json": {
            "candidate_id":"V020_T20_EXP_05","status":"COMPLETED",
            "test_condition_signature":"ASTM_DIFFERENT_25C",
            "measurements":{"冲击强度":50.0},"units":{"冲击强度":"kJ/m²"},
        },
        "invalid_unit_mismatch.json": {
            "candidate_id":"V020_T20_EXP_05","status":"COMPLETED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{"冲击强度":50.0},"units":{"冲击强度":"J/m"},
        },
        "conflicting_duplicate_01.json": {
            "candidate_id":"V020_T20_EXP_01","status":"COMPLETED",
            "test_condition_signature":"T20_STANDARD_23C",
            "measurements":{"冲击强度":49.9},"units":{"冲击强度":"kJ/m²"},
            "notes":"different value should conflict",
        },
    }

    write_json(out/"campaign_create.json", campaign)
    write_json(out/"round_plan.json", round_plan)
    write_json(out/"planned_experiments.json", planned)
    for name,payload in results.items(): write_json(out/name,payload)

    print("V0.2-T20 FIXTURE BUILDER")
    print(f"campaign_create: {out/'campaign_create.json'}")
    print(f"round_plan: {out/'round_plan.json'}")
    print(f"planned_experiments: {out/'planned_experiments.json'}")
    print()
    print("V0.2-T20 FIXTURE BUILD PASS")

if __name__ == "__main__": main()
