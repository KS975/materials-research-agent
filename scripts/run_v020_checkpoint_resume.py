from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments import (
    CampaignStore,
    CheckpointStore,
    DatasetVersionStore,
    ExperimentalResultService,
    ResumableClosedLoopWorkflow,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture-dir",default=".runtime/v020/fixtures/t25")
    parser.add_argument("--runtime-root",default=".runtime")
    parser.add_argument("--reset",action="store_true")
    args=parser.parse_args()
    fixture=Path(args.fixture_dir)
    cdoc=load_json(fixture/"campaign_create.json")
    plan=load_json(fixture/"round1_plan.json")
    planned=load_json(fixture/"round1_planned_experiments.json")
    results=load_json(fixture/"round1_results.json")
    gate=load_json(fixture/"gate_pass.json")
    cid=cdoc["campaign_id"]; pid=cdoc["project_id"]

    campaigns=CampaignStore(args.runtime_root)
    result_service=ExperimentalResultService(args.runtime_root)
    datasets=DatasetVersionStore(args.runtime_root)
    checkpoints=CheckpointStore(args.runtime_root)

    if args.reset:
        for path in [
            campaigns.campaign_dir(cid), datasets.project_dir(pid),
            Path(args.runtime_root)/"v020"/"checkpoints"/cid,
            Path(args.runtime_root)/"v020"/"models"/f"project_{pid}",
            Path(args.runtime_root)/"v020"/"model_promotion"/f"project_{pid}",
            Path(args.runtime_root)/"v020"/"closed_loop_bo"/cid,
        ]:
            if path.exists(): shutil.rmtree(path)

    print("V0.2-T25 CHECKPOINT + RESUME")
    print(f"campaign_id: {cid}")
    print(f"project_id: {pid}")
    print()

    datasets.register_base_csv(
        project_id=pid,dataset_version="dataset_v001",
        source_csv=fixture/"dataset_v001.csv",metadata={"fixture":True},
    )
    campaigns.create(
        campaign_id=cid,project_id=pid,name=cdoc["name"],
        target_metrics=cdoc["target_metrics"],metadata=cdoc.get("metadata"),
    )
    r1=campaigns.add_round(cid,plan=plan)
    result_service.register_planned_experiments(
        cid,round_id=r1["round_id"],experiments=planned,
    )
    campaigns.transition_round(cid,round_id=r1["round_id"],new_status="RUNNING")

    workflow_id=f"{r1['round_id']}__to_next_round"
    context={
        "campaign_id":cid,"source_round_id":r1["round_id"],
        "parent_dataset_version":"dataset_v001","child_dataset_version":"dataset_v002",
        "candidate_pool_sha256":__import__('hashlib').sha256((fixture/"candidate_pool.csv").read_bytes()).hexdigest(),
        "target_metric":"冲击强度","target_unit":"kJ/m²","gate":gate,
        "incumbent_model_version":"model_v001","challenger_model_version":"model_v002",
        "model_family":"ExtraTreesRegressor","batch_size":5,"acquisition":"EI",
        "direction":"maximize","random_state":42,
    }
    checkpoints.start_or_resume(campaign_id=cid,workflow_id=workflow_id,context=context)

    # Two real results arrive, then process restarts.
    first_replays=[]
    for payload in results[:2]:
        result_service.ingest(cid,round_id=r1["round_id"],payload=payload)
    summary=result_service.summary(cid,round_id=r1["round_id"])
    checkpoints.record_progress(
        campaign_id=cid,workflow_id=workflow_id,
        progress={"round_terminal_results":summary["progress"]["terminal"],"round_pending":summary["progress"]["pending"]},
    )
    print("PARTIAL RESULT CHECKPOINT")
    print(f"terminal_results_before_restart: {summary['progress']['terminal']}")
    print(f"pending_before_restart: {summary['progress']['pending']}")
    print()

    # Simulated process restart: new service objects.
    result_service=ExperimentalResultService(args.runtime_root)
    for payload in results[:2]:
        replay=result_service.ingest(cid,round_id=r1["round_id"],payload=payload)
        first_replays.append(replay["idempotent_replay"])
    for payload in results[2:]:
        result_service.ingest(cid,round_id=r1["round_id"],payload=payload)
    campaigns=CampaignStore(args.runtime_root)
    campaigns.transition_round(cid,round_id=r1["round_id"],new_status="COMPLETED")
    print("RESULT INGESTION RESUME")
    print(f"replayed_first_two_idempotent: {str(all(first_replays)).lower()}")
    print("round_status_after_resume: COMPLETED")
    print()

    # Run workflow only through DATASET_UPDATED, then simulate another crash.
    workflow=ResumableClosedLoopWorkflow(args.runtime_root)
    paused=workflow.resume(
        campaign_id=cid,source_round_id=r1["round_id"],
        parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",
        candidate_pool_csv=fixture/"candidate_pool.csv",target_metric="冲击强度",
        target_unit="kJ/m²",gate=gate,batch_size=5,random_state=42,
        pause_after_step="DATASET_UPDATED",
    )
    print("CHECKPOINT BEFORE MODEL STEP")
    print(f"status: {paused['status']}")
    print(f"last_completed_step: {paused['last_completed_step']}")
    print(f"dataset_v002_rows: {datasets.load_manifest(pid,'dataset_v002')['row_count']}")
    print()

    # Restart again. It must not append the dataset twice; it resumes T23/T24.
    workflow=ResumableClosedLoopWorkflow(args.runtime_root)
    completed=workflow.resume(
        campaign_id=cid,source_round_id=r1["round_id"],
        parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",
        candidate_pool_csv=fixture/"candidate_pool.csv",target_metric="冲击强度",
        target_unit="kJ/m²",gate=gate,batch_size=5,random_state=42,
    )
    campaign=CampaignStore(args.runtime_root).load(cid)
    r2=campaign["rounds"][-1]
    cp=completed["checkpoint"]
    print("RESUME AFTER RESTART")
    print(f"workflow_status: {completed['status']}")
    print(f"resume_count: {cp['resume_count']}")
    print(f"model_decision: {completed['model_decision']}")
    print(f"next_round_id: {completed['next_round_id']}")
    print(f"round_count: {len(campaign['rounds'])}")
    print(f"round2_status: {r2['status']}")
    print(f"round2_planned_experiments: {len(r2['experiments'])}")
    print()

    # Fully completed replay must not create Round 3.
    replay=ResumableClosedLoopWorkflow(args.runtime_root).resume(
        campaign_id=cid,source_round_id=r1["round_id"],
        parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",
        candidate_pool_csv=fixture/"candidate_pool.csv",target_metric="冲击强度",
        target_unit="kJ/m²",gate=gate,batch_size=5,random_state=42,
    )
    campaign2=CampaignStore(args.runtime_root).load(cid)
    print("COMPLETED WORKFLOW REPLAY")
    print(f"idempotent_replay: {str(replay['idempotent_replay']).lower()}")
    print(f"same_next_round: {str(replay['next_round_id']==completed['next_round_id']).lower()}")
    print(f"round_count_after_replay: {len(campaign2['rounds'])}")
    print()

    print("CHECKPOINT")
    print(f"completed_steps: {json.dumps(replay['checkpoint']['completed_steps'],ensure_ascii=False)}")
    print(f"checkpoint_json: {replay['checkpoint_json']}")
    print()

    if len(campaign2["rounds"]) != 2:
        raise SystemExit("ERROR: resume/replay created unexpected extra Round")
    if r2["status"] != "PLANNED" or len(r2["experiments"]) != 5:
        raise SystemExit("ERROR: recovered Round 2 invalid")
    if replay["checkpoint"]["status"] != "COMPLETED":
        raise SystemExit("ERROR: checkpoint did not reach COMPLETED")
    if datasets.load_manifest(pid,"dataset_v002")["row_count"] != 40:
        raise SystemExit("ERROR: dataset was duplicated during resume")

    print("V0.2-T25 CHECKPOINT + RESUME PASS")

if __name__=="__main__": main()
