import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import (
    CampaignStore,
    ClosedLoopBOConflictError,
    ClosedLoopBOService,
    DatasetVersionStore,
    ExperimentalResultService,
)
from optimization.bayesian_optimization import feature_key_from_vector


def make_base_csv(path: Path, project_id=1, n=20):
    rng=np.random.default_rng(1)
    fields=[
        "candidate_id","project_id","test_condition_signature",
        "source_campaign","source_round",
        "formula::A","formula::B","process::T","impact",
    ]
    rows=[]
    for i in range(n):
        a=float(rng.uniform(20,40)); b=100-a; t=float(rng.uniform(220,280))
        y=10+0.4*a-0.01*(t-250)**2
        rows.append({
            "candidate_id":f"B{i}","project_id":project_id,
            "test_condition_signature":"COND","source_campaign":"BASE",
            "source_round":"BASE","formula::A":a,"formula::B":b,
            "process::T":t,"impact":y,
        })
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def make_pool(path: Path, base_rows, count=100):
    rng=np.random.default_rng(2)
    fields=["candidate_id","hard_valid","soft_penalty","formula::A","formula::B","process::T"]
    rows=[]
    # exact observed duplicate
    r=base_rows[0]
    rows.append({"candidate_id":"DUP","hard_valid":"true","soft_penalty":0,
                 "formula::A":r["formula::A"],"formula::B":r["formula::B"],"process::T":r["process::T"]})
    for i in range(count-1):
        a=float(rng.uniform(22,38)); b=100-a; t=float(rng.uniform(225,275))
        rows.append({"candidate_id":f"P{i}","hard_valid":"true","soft_penalty":0,
                     "formula::A":a,"formula::B":b,"process::T":t})
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def setup_completed_round(tmp_path):
    runtime=tmp_path/"runtime"; base_csv=tmp_path/"base.csv"; pool=tmp_path/"pool.csv"
    base_rows=make_base_csv(base_csv)
    make_pool(pool,base_rows)
    campaigns=CampaignStore(runtime); results=ExperimentalResultService(runtime); datasets=DatasetVersionStore(runtime)
    datasets.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=base_csv)
    campaigns.create(campaign_id="C1",project_id=1,name="demo",target_metrics=["impact"])
    plan={"planned_experiment_count":2,"dataset_version":"dataset_v001","model_versions":{"impact":"m1"},
          "search_space_snapshot":{"v":1},"constraints_snapshot":{"v":1},"optimizer_config":{"acquisition":"EI"}}
    r=campaigns.add_round("C1",plan=plan)
    exp=[]
    for i,(a,t,val) in enumerate([(30.0,250.0,25.0),(32.0,252.0,26.0)],start=1):
        exp.append({"candidate_id":f"R1_{i}","required_metrics":["impact"],"expected_test_condition_signature":"COND",
                    "units":{"impact":"u"},"features":{"formula::A":a,"formula::B":100-a,"process::T":t},
                    "prediction_snapshot":{"impact":{"value":val-1}}})
    results.register_planned_experiments("C1",round_id=r["round_id"],experiments=exp)
    campaigns.transition_round("C1",round_id=r["round_id"],new_status="RUNNING")
    for i,e in enumerate(exp,start=1):
        results.ingest("C1",round_id=r["round_id"],payload={"candidate_id":e["candidate_id"],"status":"COMPLETED",
            "test_condition_signature":"COND","measurements":{"impact":25.0+i},"units":{"impact":"u"}})
    campaigns.transition_round("C1",round_id=r["round_id"],new_status="COMPLETED")
    datasets.update_from_round(campaign_store=campaigns,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    return runtime,pool,r["round_id"],campaigns,datasets


def gate(ok=True):
    return {"decision":"PASS" if ok else "FAIL","training_allowed":ok,"official_model_allowed":ok}


def test_gate_blocks_closed_loop(tmp_path):
    runtime,pool,rid,_,_=setup_completed_round(tmp_path)
    with pytest.raises(ClosedLoopBOConflictError):
        ClosedLoopBOService(runtime).generate_next_round(
            campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
            target_metric="impact",target_unit="u",gate=gate(False),batch_size=2)


def test_source_round_must_be_completed(tmp_path):
    runtime=tmp_path/"runtime"; base_csv=tmp_path/"base.csv"; pool=tmp_path/"pool.csv"
    base_rows=make_base_csv(base_csv); make_pool(pool,base_rows)
    campaigns=CampaignStore(runtime); datasets=DatasetVersionStore(runtime)
    datasets.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=base_csv)
    campaigns.create(campaign_id="C1",project_id=1,name="demo",target_metrics=["impact"])
    r=campaigns.add_round("C1",plan={"planned_experiment_count":2,"dataset_version":"dataset_v001","model_versions":{"impact":"m1"},
        "search_space_snapshot":{"v":1},"constraints_snapshot":{"v":1},"optimizer_config":{}})
    with pytest.raises(ClosedLoopBOConflictError):
        ClosedLoopBOService(runtime).generate_next_round(
            campaign_id="C1",source_round_id=r["round_id"],latest_dataset_version="dataset_v001",candidate_pool_csv=pool,
            target_metric="impact",target_unit="u",gate=gate(),batch_size=2)


def test_closed_loop_creates_planned_next_round(tmp_path):
    runtime,pool,rid,campaigns,_=setup_completed_round(tmp_path)
    report=ClosedLoopBOService(runtime).generate_next_round(
        campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
        target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    campaign=campaigns.load("C1"); r2=campaign["rounds"][-1]
    assert report["next_round_id"]==r2["round_id"]
    assert r2["status"]=="PLANNED"
    assert r2["plan"]["dataset_version"]=="dataset_v002"
    assert len(r2["experiments"])==2


def test_observed_feature_duplicate_is_filtered(tmp_path):
    runtime,pool,rid,_,datasets=setup_completed_round(tmp_path)
    report=ClosedLoopBOService(runtime).generate_next_round(
        campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
        target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    assert report["candidate_flow"]["already_observed_feature_filtered"] >= 1
    fields,rows=datasets.load_rows(1,"dataset_v002")
    fcols=[c for c in fields if c.startswith("formula::") or c.startswith("process::")]
    observed={feature_key_from_vector([float(r[c]) for c in fcols]) for r in rows}
    for item in report["next_experiments"]:
        assert feature_key_from_vector([item["features"][c] for c in fcols]) not in observed


def test_best_so_far_uses_feedback_dataset(tmp_path):
    runtime,pool,rid,_,_=setup_completed_round(tmp_path)
    report=ClosedLoopBOService(runtime).generate_next_round(
        campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
        target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    b=report["best_so_far"]
    assert b["current_dataset_best"] >= b["previous_dataset_best"]


def test_idempotent_replay_does_not_create_third_round(tmp_path):
    runtime,pool,rid,campaigns,_=setup_completed_round(tmp_path)
    service=ClosedLoopBOService(runtime)
    first=service.generate_next_round(
        campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
        target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    second=service.generate_next_round(
        campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
        target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert len(campaigns.load("C1")["rounds"])==2


def test_next_round_predictions_are_not_measurements(tmp_path):
    runtime,pool,rid,campaigns,_=setup_completed_round(tmp_path)
    ClosedLoopBOService(runtime).generate_next_round(
        campaign_id="C1",source_round_id=rid,latest_dataset_version="dataset_v002",candidate_pool_csv=pool,
        target_metric="impact",target_unit="u",gate=gate(),batch_size=2,min_batch_distance=0.0)
    r2=campaigns.load("C1")["rounds"][-1]
    for exp in r2["experiments"]:
        assert exp["status"]=="PLANNED"
        assert exp["result"] is None
        assert "posterior_std" in exp["prediction_snapshot"]["impact"]
