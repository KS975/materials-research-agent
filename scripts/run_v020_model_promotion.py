from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from experiments import (
    CampaignStore, DatasetVersionStore, ExperimentalResultService,
    ModelPromotionService,
)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture-dir",default=".runtime/v020/fixtures/t23")
    parser.add_argument("--runtime-root",default=".runtime")
    parser.add_argument("--reset",action="store_true")
    args=parser.parse_args()

    fixture=Path(args.fixture_dir)
    cdoc=load_json(fixture/"campaign_create.json")
    plan=load_json(fixture/"round_plan.json")
    planned=load_json(fixture/"planned_experiments.json")
    results=load_json(fixture/"results.json")
    gate=load_json(fixture/"gate_pass.json")
    pid=cdoc["project_id"]; cid=cdoc["campaign_id"]

    campaigns=CampaignStore(args.runtime_root)
    result_service=ExperimentalResultService(args.runtime_root)
    datasets=DatasetVersionStore(args.runtime_root)
    promotion=ModelPromotionService(args.runtime_root)

    if args.reset:
        for path in [
            campaigns.campaign_dir(cid),
            datasets.project_dir(pid),
            Path(args.runtime_root)/"v020"/"models"/f"project_{pid}",
            Path(args.runtime_root)/"v020"/"model_promotion"/f"project_{pid}",
        ]:
            if path.exists(): shutil.rmtree(path)

    print("V0.2-T23 MODEL RETRAINING + PROMOTION")
    print(f"project_id: {pid}")
    print("target_metric: 冲击强度")
    print()

    base=datasets.register_base_csv(
        project_id=pid,dataset_version="dataset_v001",
        source_csv=fixture/"dataset_v001.csv",metadata={"fixture":True},
    )

    campaigns.create(
        campaign_id=cid,project_id=pid,name=cdoc["name"],
        target_metrics=cdoc["target_metrics"],metadata=cdoc.get("metadata"),
    )
    r=campaigns.add_round(cid,plan=plan)
    result_service.register_planned_experiments(
        cid,round_id=r["round_id"],experiments=planned
    )
    campaigns.transition_round(cid,round_id=r["round_id"],new_status="RUNNING")
    for payload in results:
        result_service.ingest(cid,round_id=r["round_id"],payload=payload)
    campaigns.transition_round(cid,round_id=r["round_id"],new_status="COMPLETED")
    child=datasets.update_from_round(
        campaign_store=campaigns,campaign_id=cid,round_id=r["round_id"],
        new_dataset_version="dataset_v002",
    )["manifest"]

    print("DATASETS")
    print(f"dataset_v001_rows: {base['row_count']}")
    print(f"dataset_v002_rows: {child['row_count']}")
    print(f"added_rows: {child['added_row_count']}")
    print()

    report=promotion.compare_and_register(
        project_id=pid,target_metric="冲击强度",
        parent_dataset_version="dataset_v001",
        child_dataset_version="dataset_v002",
        incumbent_model_version="model_v001",
        challenger_model_version="model_v002",
        model_family="ExtraTreesRegressor",
        gate=gate,folds=5,random_state=42,
    )

    inc=report["incumbent"]; ch=report["challenger"]
    print("COMMON HOLDOUT")
    print(f"rows: {report['dataset']['common_holdout_rows']}")
    print()
    print("INCUMBENT")
    print("model_version: model_v001")
    print(f"dataset_version: {inc['dataset_version']}")
    print(f"holdout_R2: {inc['holdout']['r2']:.6f}")
    print(f"holdout_MAE: {inc['holdout']['mae']:.6f}")
    print(f"holdout_RMSE: {inc['holdout']['rmse']:.6f}")
    print(f"CV_R2_mean: {inc['cv']['summary']['r2']['mean']:.6f}")
    print()
    print("CHALLENGER")
    print("model_version: model_v002")
    print(f"dataset_version: {ch['dataset_version']}")
    print(f"holdout_R2: {ch['holdout']['r2']:.6f}")
    print(f"holdout_MAE: {ch['holdout']['mae']:.6f}")
    print(f"holdout_RMSE: {ch['holdout']['rmse']:.6f}")
    print(f"CV_R2_mean: {ch['cv']['summary']['r2']['mean']:.6f}")
    print(f"CV_R2_std: {ch['cv']['summary']['r2']['std']:.6f}")
    print()

    print("PROMOTION DECISION")
    print(f"decision: {report['decision']}")
    if "deltas" in report:
        print(f"RMSE_improvement_fraction: {report['deltas']['rmse_improvement_fraction']:.6f}")
        print(f"MAE_improvement_fraction: {report['deltas']['mae_improvement_fraction']:.6f}")
        print(f"R2_delta: {report['deltas']['r2_delta']:.6f}")
    print()
    print("MODEL GOVERNANCE")
    print(f"active_model_version: {report['registry']['active_model_version_after_decision']}")
    print(f"challenger_status: {report['registry']['challenger_status_after_decision']}")
    print("automatic_activation: false")
    print("human_approval_required: true")
    print()
    print("OUTPUT")
    print(f"report_json: {report['report_json']}")
    print()

    if report["decision"] != "PROMOTE":
        raise SystemExit(f"ERROR: fixture expected PROMOTE, got {report['decision']}")
    if report["registry"]["active_model_version_after_decision"] != "model_v001":
        raise SystemExit("ERROR: PROMOTE recommendation must not auto-activate challenger")
    if report["registry"]["challenger_status_after_decision"] != "CANDIDATE":
        raise SystemExit("ERROR: challenger should remain CANDIDATE before human approval")

    print("NOTE: PROMOTE 是系统建议；V0.2 不自动替换正式模型。")
    print("V0.2-T23 MODEL RETRAINING + PROMOTION PASS")

if __name__=="__main__":
    main()
