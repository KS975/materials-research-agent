from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def make_rows(n: int, seed: int, prefix: str, project_id: int):
    rng = np.random.default_rng(seed)
    abs_values = rng.uniform(20, 45, n)
    tough_values = rng.uniform(8, 22, n)
    pc_values = 100.0 - abs_values - tough_values
    temp_values = rng.uniform(220, 280, n)
    speed_values = rng.uniform(180, 360, n)
    noise = rng.normal(0, 0.3, n)
    impact_values = (
        20
        + 0.35 * abs_values
        + 0.8 * tough_values
        - 0.02 * (temp_values - 250) ** 2
        + 0.015 * speed_values
        + 2.0 * np.sin(abs_values / 5.0)
        + 1.5 * np.cos(tough_values / 3.0)
        + noise
    )
    rows = []
    for i in range(n):
        abs_v = float(abs_values[i])
        tough = float(tough_values[i])
        pc = float(pc_values[i])
        temp = float(temp_values[i])
        speed = float(speed_values[i])
        impact = float(impact_values[i])
        rows.append({
            "candidate_id": f"{prefix}_{i+1:03d}",
            "project_id": str(project_id),
            "test_condition_signature": "T23_STANDARD_23C",
            "source_campaign": "BASE_IMPORT" if prefix == "BASE" else "V020_T23_DEMO",
            "source_round": "BASE" if prefix == "BASE" else "V020_T23_DEMO-R001",
            "formula::ABS": f"{abs_v:.8f}",
            "formula::PC": f"{pc:.8f}",
            "formula::增韧剂": f"{tough:.8f}",
            "process::加工温度": f"{temp:.8f}",
            "process::螺杆转速": f"{speed:.8f}",
            "冲击强度": f"{impact:.8f}",
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".runtime/v020/fixtures/t23")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    project_id = 9023
    columns = [
        "candidate_id","project_id","test_condition_signature",
        "source_campaign","source_round",
        "formula::ABS","formula::PC","formula::增韧剂",
        "process::加工温度","process::螺杆转速","冲击强度",
    ]

    base_rows = make_rows(35, 1, "BASE", project_id)
    new_rows = make_rows(20, 2, "NEW", project_id)

    base_csv = out/"dataset_v001.csv"
    with base_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(base_rows)

    campaign = {
        "campaign_id":"V020_T23_DEMO","project_id":project_id,
        "name":"冲击强度模型重训与晋级演示",
        "target_metrics":["冲击强度"],
        "metadata":{"fixture":True},
    }
    plan = {
        "planned_experiment_count":20,
        "dataset_version":"dataset_v001",
        "model_versions":{"冲击强度":"model_v001"},
        "search_space_snapshot":{"version":"search_space_v001"},
        "constraints_snapshot":{"version":"constraints_v001"},
        "optimizer_config":{"engine":"GaussianProcess","acquisition":"EI"},
    }

    planned, results = [], []
    for row in new_rows:
        cid = row["candidate_id"]
        planned.append({
            "candidate_id":cid,
            "required_metrics":["冲击强度"],
            "expected_test_condition_signature":"T23_STANDARD_23C",
            "units":{"冲击强度":"kJ/m²"},
            "features":{
                "formula::ABS":float(row["formula::ABS"]),
                "formula::PC":float(row["formula::PC"]),
                "formula::增韧剂":float(row["formula::增韧剂"]),
                "process::加工温度":float(row["process::加工温度"]),
                "process::螺杆转速":float(row["process::螺杆转速"]),
            },
            "prediction_snapshot":{"冲击强度":{"value":float(row["冲击强度"])-0.7}},
        })
        results.append({
            "candidate_id":cid,"status":"COMPLETED",
            "test_condition_signature":"T23_STANDARD_23C",
            "measurements":{"冲击强度":float(row["冲击强度"])},
            "units":{"冲击强度":"kJ/m²"},
        })

    gate = {
        "stage":"V0.1.3-B_modeling_gate",
        "project_id":project_id,
        "target_metric":"冲击强度",
        "decision":"PASS",
        "training_allowed":True,
        "official_model_allowed":True,
    }

    write_json(out/"campaign_create.json", campaign)
    write_json(out/"round_plan.json", plan)
    write_json(out/"planned_experiments.json", planned)
    write_json(out/"results.json", results)
    write_json(out/"gate_pass.json", gate)

    print("V0.2-T23 FIXTURE BUILDER")
    print(f"base_dataset_csv: {base_csv}")
    print("base_rows: 35")
    print("new_training_rows: 20")
    print("expected_child_rows: 55")
    print("model_family: ExtraTreesRegressor")
    print()
    print("V0.2-T23 FIXTURE BUILD PASS")

if __name__ == "__main__":
    main()
