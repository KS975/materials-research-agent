import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import (
    CampaignStore,
    CheckpointConflictError,
    CheckpointStore,
    ClosedLoopBOService,
    DatasetVersionStore,
    ExperimentalResultService,
    ResumableClosedLoopWorkflow,
)


def base_csv(path: Path, n=25):
    rng=np.random.default_rng(11)
    fields=["candidate_id","project_id","test_condition_signature","source_campaign","source_round","formula::A","formula::B","process::T","impact"]
    rows=[]
    for i in range(n):
        a=float(rng.uniform(20,40)); b=100-a; t=float(rng.uniform(225,275)); y=10+0.4*a-0.008*(t-250)**2+rng.normal(0,0.2)
        rows.append({"candidate_id":f"B{i}","project_id":1,"test_condition_signature":"COND","source_campaign":"BASE","source_round":"BASE","formula::A":a,"formula::B":b,"process::T":t,"impact":y})
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def pool_csv(path: Path, n=120):
    rng=np.random.default_rng(12)
    fields=["candidate_id","hard_valid","soft_penalty","formula::A","formula::B","process::T"]
    rows=[]
    for i in range(n):
        a=float(rng.uniform(22,38)); b=100-a; t=float(rng.uniform(228,272))
        rows.append({"candidate_id":f"P{i}","hard_valid":"true","soft_penalty":0,"formula::A":a,"formula::B":b,"process::T":t})
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def gate():
    return {"decision":"PASS","training_allowed":True,"official_model_allowed":True}


def setup(tmp_path, complete=True):
    runtime=tmp_path/"runtime"; bc=tmp_path/"base.csv"; pc=tmp_path/"pool.csv"
    base_csv(bc); pool_csv(pc)
    campaigns=CampaignStore(runtime); results=ExperimentalResultService(runtime); datasets=DatasetVersionStore(runtime)
    datasets.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=bc)
    campaigns.create(campaign_id="C1",project_id=1,name="demo",target_metrics=["impact"])
    plan={"planned_experiment_count":5,"dataset_version":"dataset_v001","model_versions":{"impact":"model_v001"},"search_space_snapshot":{"v":1},"constraints_snapshot":{"v":1},"optimizer_config":{"acquisition":"EI"},"source":"old"}
    r=campaigns.add_round("C1",plan=plan)
    exps=[]
    for i in range(5):
        a=28+i; t=245+i; cid=f"R1_{i}"
        exps.append({"candidate_id":cid,"required_metrics":["impact"],"expected_test_condition_signature":"COND","units":{"impact":"u"},"features":{"formula::A":a,"formula::B":100-a,"process::T":t},"prediction_snapshot":{"impact":{"value":20+i}}})
    results.register_planned_experiments("C1",round_id=r["round_id"],experiments=exps)
    campaigns.transition_round("C1",round_id=r["round_id"],new_status="RUNNING")
    if complete:
        for i,e in enumerate(exps):
            results.ingest("C1",round_id=r["round_id"],payload={"candidate_id":e["candidate_id"],"status":"COMPLETED","test_condition_signature":"COND","measurements":{"impact":21.0+i},"units":{"impact":"u"}})
        campaigns.transition_round("C1",round_id=r["round_id"],new_status="COMPLETED")
    return runtime,pc,r["round_id"],campaigns,results,datasets,exps


def workflow_context(pool):
    import hashlib
    return {"campaign_id":"C1","source_round_id":"C1-R001","parent_dataset_version":"dataset_v001","child_dataset_version":"dataset_v002","candidate_pool_sha256":hashlib.sha256(Path(pool).read_bytes()).hexdigest(),"target_metric":"impact","target_unit":"u","gate":gate(),"incumbent_model_version":"model_v001","challenger_model_version":"model_v002","model_family":"ExtraTreesRegressor","batch_size":2,"acquisition":"EI","direction":"maximize","random_state":42}


def test_checkpoint_context_fingerprint_blocks_changed_request(tmp_path):
    store=CheckpointStore(tmp_path); ctx={"x":1}
    store.start_or_resume(campaign_id="C1",workflow_id="W1",context=ctx)
    with pytest.raises(CheckpointConflictError):
        store.start_or_resume(campaign_id="C1",workflow_id="W1",context={"x":2})


def test_checkpoint_steps_are_monotonic_and_idempotent(tmp_path):
    s=CheckpointStore(tmp_path); s.start_or_resume(campaign_id="C1",workflow_id="W1",context={"x":1})
    s.mark_step(campaign_id="C1",workflow_id="W1",step="ROUND_COMPLETED")
    again=s.mark_step(campaign_id="C1",workflow_id="W1",step="ROUND_COMPLETED")
    assert again["completed_steps"]==["ROUND_COMPLETED"]
    with pytest.raises(CheckpointConflictError):
        s.mark_step(campaign_id="C1",workflow_id="W1",step="MODEL_DECISION_RECORDED")


def test_partial_results_survive_restart_and_replay_is_idempotent(tmp_path):
    runtime,pool,rid,campaigns,results,_,exps=setup(tmp_path,complete=False)
    payload={"candidate_id":exps[0]["candidate_id"],"status":"COMPLETED","test_condition_signature":"COND","measurements":{"impact":21.0},"units":{"impact":"u"}}
    results.ingest("C1",round_id=rid,payload=payload)
    restarted=ExperimentalResultService(runtime)
    replay=restarted.ingest("C1",round_id=rid,payload=payload)
    assert replay["idempotent_replay"] is True
    assert restarted.summary("C1",round_id=rid)["progress"]["terminal"]==1


def test_resume_after_dataset_update_does_not_duplicate_rows(tmp_path):
    runtime,pool,rid,_,_,datasets,_=setup(tmp_path,complete=True)
    w=ResumableClosedLoopWorkflow(runtime)
    paused=w.resume(campaign_id="C1",source_round_id=rid,parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2,pause_after_step="DATASET_UPDATED")
    rows_before=datasets.load_manifest(1,"dataset_v002")["row_count"]
    completed=ResumableClosedLoopWorkflow(runtime).resume(campaign_id="C1",source_round_id=rid,parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2)
    rows_after=datasets.load_manifest(1,"dataset_v002")["row_count"]
    assert paused["last_completed_step"]=="DATASET_UPDATED"
    assert rows_before==rows_after==30
    assert completed["status"]=="COMPLETED"


def test_t24_recovers_precreated_empty_next_round(tmp_path):
    runtime,pool,rid,campaigns,_,datasets,_=setup(tmp_path,complete=True)
    datasets.update_from_round(campaign_store=campaigns,campaign_id="C1",round_id=rid,new_dataset_version="dataset_v002")
    source=campaigns.load("C1")["rounds"][0]
    plan={"planned_experiment_count":2,"dataset_version":"dataset_v002","model_versions":{"impact":"model_v001"},"search_space_snapshot":source["plan"]["search_space_snapshot"],"constraints_snapshot":source["plan"]["constraints_snapshot"],"optimizer_config":{"engine":"GaussianProcess","acquisition":"EI"},"source":"V0.2-T24_closed_loop_BO"}
    pre=campaigns.add_round("C1",plan=plan)
    assert "experiments" not in pre
    report=ClosedLoopBOService(runtime).generate_next_round(campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    campaign=campaigns.load("C1")
    assert len(campaign["rounds"])==2
    assert report["next_round_id"]==pre["round_id"]
    assert report["recovery"]["reused_existing_next_round"] is True
    assert report["recovery"]["recovered_missing_experiment_registration"] is True
    assert len(campaign["rounds"][1]["experiments"])==2


def test_t24_recovers_when_final_report_was_lost(tmp_path):
    runtime,pool,rid,campaigns,_,datasets,_=setup(tmp_path,complete=True)
    datasets.update_from_round(campaign_store=campaigns,campaign_id="C1",round_id=rid,new_dataset_version="dataset_v002")
    service=ClosedLoopBOService(runtime)
    first=service.generate_next_round(campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    Path(first["report_json"]).unlink()
    recovered=service.generate_next_round(campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    assert len(campaigns.load("C1")["rounds"])==2
    assert recovered["next_round_id"]==first["next_round_id"]
    assert recovered["recovery"]["recovered_existing_experiment_registration"] is True


def test_completed_workflow_replay_does_not_create_round3(tmp_path):
    runtime,pool,rid,campaigns,_,_,_=setup(tmp_path,complete=True)
    kwargs=dict(campaign_id="C1",source_round_id=rid,parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2)
    first=ResumableClosedLoopWorkflow(runtime).resume(**kwargs)
    second=ResumableClosedLoopWorkflow(runtime).resume(**kwargs)
    assert first["next_round_id"]==second["next_round_id"]
    assert second["idempotent_replay"] is True
    assert len(campaigns.load("C1")["rounds"])==2


def test_checkpoint_completed_contains_all_steps(tmp_path):
    runtime,pool,rid,_,_,_,_=setup(tmp_path,complete=True)
    result=ResumableClosedLoopWorkflow(runtime).resume(campaign_id="C1",source_round_id=rid,parent_dataset_version="dataset_v001",child_dataset_version="dataset_v002",candidate_pool_csv=pool,target_metric="impact",target_unit="u",gate=gate(),batch_size=2)
    assert result["checkpoint"]["completed_steps"]==[
        "ROUND_COMPLETED","DATASET_UPDATED","MODEL_DECISION_RECORDED","NEXT_ROUND_CREATED","WORKFLOW_COMPLETED"
    ]
    assert result["checkpoint"]["status"]=="COMPLETED"
