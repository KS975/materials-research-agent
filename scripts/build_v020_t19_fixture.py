from __future__ import annotations
import argparse, json
from pathlib import Path

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir", default=".runtime/v020/fixtures/t19"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    campaign={"campaign_id":"V020_T19_DEMO","project_id":9018,"name":"冲击强度闭环优化演示","target_metrics":["冲击强度"],"metadata":{"purpose":"V0.2-T19 engineering acceptance","fixture":True}}
    round1={"planned_experiment_count":5,"dataset_version":"dataset_v001","model_versions":{"冲击强度":"model_v001"},"search_space_snapshot":{"version":"search_space_v001","source":"V0.1.4-T18 fixture"},"constraints_snapshot":{"version":"constraints_v001","hard_constraints":["formula_sum_100"],"soft_constraints":["toughener_preferred_max"]},"optimizer_config":{"engine":"GaussianProcess","acquisition":"EI","batch_strategy":"kriging_believer","batch_size":5},"source":"V0.1.4-T18","notes":"Round 1 uses frozen V0.1.4 BO design."}
    round2={"planned_experiment_count":5,"dataset_version":"dataset_v002","model_versions":{"冲击强度":"model_v002"},"search_space_snapshot":{"version":"search_space_v001","source":"same validated search space"},"constraints_snapshot":{"version":"constraints_v001","hard_constraints":["formula_sum_100"],"soft_constraints":["toughener_preferred_max"]},"optimizer_config":{"engine":"GaussianProcess","acquisition":"EI","batch_strategy":"kriging_believer","batch_size":5},"source":"planned T24 closed-loop update","notes":"Round 2 can only be created after Round 1 COMPLETED."}
    write_json(out/"campaign_create.json",campaign); write_json(out/"round1_plan.json",round1); write_json(out/"round2_plan.json",round2)
    print("V0.2-T19 FIXTURE BUILDER")
    print(f"campaign_create: {out/'campaign_create.json'}")
    print(f"round1_plan: {out/'round1_plan.json'}")
    print(f"round2_plan: {out/'round2_plan.json'}")
    print("\nEXPECTED")
    print("- create Campaign")
    print("- create Round 1 -> PLANNED")
    print("- Round 1 未完成时创建 Round 2 -> BLOCKED")
    print("- Round 1: PLANNED -> RUNNING -> COMPLETED")
    print("- create Round 2 -> PLANNED")
    print("\nV0.2-T19 FIXTURE BUILD PASS")

if __name__=="__main__": main()
