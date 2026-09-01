from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def response(a, tough, temp, speed):
    return (
        22.0 + 0.32 * a + 0.88 * tough
        - 0.009 * (temp - 248.0) ** 2
        + 0.012 * speed
        + 1.8 * np.sin(a / 6.5)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".runtime/v020/fixtures/t25")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(25)
    project_id = 9025
    campaign_id = "V020_T25_DEMO"
    fields = [
        "candidate_id","project_id","test_condition_signature",
        "source_campaign","source_round",
        "formula::ABS","formula::PC","formula::增韧剂",
        "process::加工温度","process::螺杆转速","冲击强度",
    ]

    base_rows=[]
    for i in range(35):
        a=float(rng.uniform(22,42)); tough=float(rng.uniform(10,20)); pc=100-a-tough
        temp=float(rng.uniform(225,275)); speed=float(rng.uniform(190,350))
        y=float(response(a,tough,temp,speed)+rng.normal(0,0.6))
        base_rows.append({
            "candidate_id":f"BASE_{i+1:03d}","project_id":project_id,
            "test_condition_signature":"T25_STANDARD_23C",
            "source_campaign":"BASE_IMPORT","source_round":"BASE",
            "formula::ABS":a,"formula::PC":pc,"formula::增韧剂":tough,
            "process::加工温度":temp,"process::螺杆转速":speed,"冲击强度":y,
        })
    with (out/"dataset_v001.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(base_rows)

    campaign={
        "campaign_id":campaign_id,"project_id":project_id,
        "name":"T25 checkpoint/resume 演示","target_metrics":["冲击强度"],
        "metadata":{"fixture":True},
    }
    plan={
        "planned_experiment_count":5,"dataset_version":"dataset_v001",
        "model_versions":{"冲击强度":"model_v001"},
        "search_space_snapshot":{"version":"search_space_v001"},
        "constraints_snapshot":{"version":"constraints_v001"},
        "optimizer_config":{"engine":"GaussianProcess","acquisition":"EI"},
        "source":"V0.1.4-T18",
    }
    feats=[
        (32.0,17.5,247.0,300.0),(34.0,18.0,249.0,315.0),
        (30.0,16.5,246.0,290.0),(36.0,18.5,251.0,325.0),
        (33.0,19.0,248.0,335.0),
    ]
    planned=[]; results=[]
    for i,(a,tough,temp,speed) in enumerate(feats,start=1):
        pc=100-a-tough; actual=float(response(a,tough,temp,speed))
        cid=f"V020_T25_R1_{i:02d}"
        feature={
            "formula::ABS":a,"formula::PC":pc,"formula::增韧剂":tough,
            "process::加工温度":temp,"process::螺杆转速":speed,
        }
        planned.append({
            "candidate_id":cid,"required_metrics":["冲击强度"],
            "expected_test_condition_signature":"T25_STANDARD_23C",
            "units":{"冲击强度":"kJ/m²"},"features":feature,
            "prediction_snapshot":{"冲击强度":{"value":actual-0.8,"posterior_std":1.1}},
        })
        results.append({
            "candidate_id":cid,"status":"COMPLETED",
            "test_condition_signature":"T25_STANDARD_23C",
            "measurements":{"冲击强度":actual},"units":{"冲击强度":"kJ/m²"},
        })

    pool=[]
    for i in range(450):
        a=float(rng.uniform(23,41)); tough=float(rng.uniform(10.5,19.5)); pc=100-a-tough
        temp=float(rng.uniform(228,272)); speed=float(rng.uniform(195,345))
        pool.append({
            "candidate_id":f"V020_T25_POOL_{i+1:04d}",
            "hard_valid":"false" if i in {90,190} else "true",
            "soft_penalty":"0.12" if tough>19 else "0",
            "formula::ABS":a,"formula::PC":pc,"formula::增韧剂":tough,
            "process::加工温度":temp,"process::螺杆转速":speed,
        })
    pfields=[
        "candidate_id","hard_valid","soft_penalty","formula::ABS","formula::PC",
        "formula::增韧剂","process::加工温度","process::螺杆转速",
    ]
    with (out/"candidate_pool.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=pfields); w.writeheader(); w.writerows(pool)

    gate={"decision":"PASS","training_allowed":True,"official_model_allowed":True}
    write_json(out/"campaign_create.json",campaign)
    write_json(out/"round1_plan.json",plan)
    write_json(out/"round1_planned_experiments.json",planned)
    write_json(out/"round1_results.json",results)
    write_json(out/"gate_pass.json",gate)

    print("V0.2-T25 FIXTURE BUILDER")
    print(f"base_dataset_csv: {out/'dataset_v001.csv'}")
    print("base_rows: 35")
    print("round1_results: 5")
    print("candidate_pool_rows: 450")
    print("simulated_partial_results_before_restart: 2")
    print("simulated_pause_after_step: DATASET_UPDATED")
    print()
    print("V0.2-T25 FIXTURE BUILD PASS")

if __name__ == "__main__": main()
