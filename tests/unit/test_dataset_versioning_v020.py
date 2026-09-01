import csv
import json
from pathlib import Path

import pytest

from experiments import (
    CampaignStore,
    DatasetIntegrityError,
    DatasetVersionConflictError,
    DatasetVersionStore,
    ExperimentalResultService,
)


def write_base(path: Path, rows=None):
    columns = [
        "candidate_id","project_id","test_condition_signature","source_campaign","source_round",
        "x","impact",
    ]
    rows = rows or [
        {"candidate_id":"BASE_1","project_id":"1","test_condition_signature":"ISO","source_campaign":"BASE","source_round":"BASE","x":"1","impact":"10"},
        {"candidate_id":"BASE_2","project_id":"1","test_condition_signature":"ISO","source_campaign":"BASE","source_round":"BASE","x":"2","impact":"11"},
    ]
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)


def plan(count=2, dataset="dataset_v001"):
    return {
        "planned_experiment_count":count,
        "dataset_version":dataset,
        "model_versions":{"impact":"model_v001"},
        "search_space_snapshot":{"version":"ss1"},
        "constraints_snapshot":{"version":"c1"},
        "optimizer_config":{"acquisition":"EI"},
    }


def planned(cid,x):
    return {
        "candidate_id":cid,
        "required_metrics":["impact"],
        "expected_test_condition_signature":"ISO",
        "units":{"impact":"kJ/m2"},
        "features":{"x":x},
        "prediction_snapshot":{"impact":{"value":12.0}},
    }


def result(cid,status,value=None):
    p={"candidate_id":cid,"status":status,"test_condition_signature":"ISO","measurements":{},"units":{}}
    if status=="COMPLETED":
        p["measurements"]={"impact":value}; p["units"]={"impact":"kJ/m2"}
    elif status in {"FAILED","INVALID"}:
        p["failure_reason"]="fixture reason"
    return p


def setup_campaign(tmp_path, statuses=("COMPLETED","FAILED"), candidate_ids=("E1","E2"), values=(12.5,None), complete_round=True, parent_dataset="dataset_v001"):
    cstore=CampaignStore(tmp_path); rservice=ExperimentalResultService(tmp_path); dstore=DatasetVersionStore(tmp_path)
    base=tmp_path/"base.csv"; write_base(base)
    dstore.register_base_csv(project_id=1,dataset_version=parent_dataset,source_csv=base)
    cstore.create(campaign_id="C1",project_id=1,name="demo",target_metrics=["impact"])
    r=cstore.add_round("C1",plan=plan(len(statuses),parent_dataset))
    exps=[planned(cid,i+3) for i,cid in enumerate(candidate_ids)]
    rservice.register_planned_experiments("C1",round_id=r["round_id"],experiments=exps)
    cstore.transition_round("C1",round_id=r["round_id"],new_status="RUNNING")
    for cid,status,value in zip(candidate_ids,statuses,values):
        rservice.ingest("C1",round_id=r["round_id"],payload=result(cid,status,value))
    if complete_round:
        cstore.transition_round("C1",round_id=r["round_id"],new_status="COMPLETED")
    return cstore,rservice,dstore,r


def test_register_base_version_and_verify(tmp_path):
    d=DatasetVersionStore(tmp_path); p=tmp_path/"base.csv"; write_base(p)
    m=d.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=p)
    assert m["row_count"]==2 and m["parent_dataset_version"] is None
    assert d.verify(1,"dataset_v001")["verified"] is True


def test_update_adds_only_training_eligible_rows(tmp_path):
    c,_,d,r=setup_campaign(tmp_path)
    out=d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    m=out["manifest"]
    assert m["row_count_before"]==2 and m["row_count_after"]==3
    assert m["added_candidate_ids"]==["E1"]
    assert m["excluded_nontraining"]["FAILED"]==1


def test_parent_version_is_immutable(tmp_path):
    c,_,d,r=setup_campaign(tmp_path)
    before=d.verify(1,"dataset_v001")["sha256"]
    d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    after=d.verify(1,"dataset_v001")["sha256"]
    assert before==after


def test_replay_same_round_and_version_is_idempotent(tmp_path):
    c,_,d,r=setup_campaign(tmp_path)
    d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    replay=d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    assert replay["idempotent_replay"] is True


def test_same_round_cannot_create_second_child_version(tmp_path):
    c,_,d,r=setup_campaign(tmp_path)
    d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    with pytest.raises(DatasetVersionConflictError):
        d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v003")


def test_round_must_be_completed_before_update(tmp_path):
    c,_,d,r=setup_campaign(tmp_path,complete_round=False)
    with pytest.raises(DatasetVersionConflictError):
        d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")


def test_existing_version_cannot_be_overwritten(tmp_path):
    c,_,d,r=setup_campaign(tmp_path)
    base2=tmp_path/"other.csv"; write_base(base2)
    d.register_base_csv(project_id=1,dataset_version="dataset_v002",source_csv=base2)
    with pytest.raises(DatasetVersionConflictError):
        d.update_from_round(campaign_store=c,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")


def test_exact_duplicate_candidate_is_not_appended(tmp_path):
    # Parent contains E1 with exactly the row T22 would build.
    base_rows=[
        {"candidate_id":"BASE_1","project_id":"1","test_condition_signature":"ISO","source_campaign":"BASE","source_round":"BASE","x":"1","impact":"10"},
        {"candidate_id":"E1","project_id":"1","test_condition_signature":"ISO","source_campaign":"C1","source_round":"C1-R001","x":"3","impact":"12.5"},
    ]
    cstore=CampaignStore(tmp_path); rs=ExperimentalResultService(tmp_path); d=DatasetVersionStore(tmp_path)
    base=tmp_path/"base.csv"; write_base(base,base_rows)
    d.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=base)
    cstore.create(campaign_id="C1",project_id=1,name="demo",target_metrics=["impact"])
    r=cstore.add_round("C1",plan=plan(2))
    rs.register_planned_experiments("C1",round_id=r["round_id"],experiments=[planned("E1",3),planned("E2",4)])
    cstore.transition_round("C1",round_id=r["round_id"],new_status="RUNNING")
    rs.ingest("C1",round_id=r["round_id"],payload=result("E1","COMPLETED",12.5))
    rs.ingest("C1",round_id=r["round_id"],payload=result("E2","COMPLETED",13.0))
    cstore.transition_round("C1",round_id=r["round_id"],new_status="COMPLETED")
    out=d.update_from_round(campaign_store=cstore,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")
    assert out["manifest"]["duplicate_skipped_candidate_ids"]==["E1"]
    assert out["manifest"]["added_candidate_ids"]==["E2"]


def test_conflicting_duplicate_candidate_is_rejected(tmp_path):
    base_rows=[
        {"candidate_id":"BASE_1","project_id":"1","test_condition_signature":"ISO","source_campaign":"BASE","source_round":"BASE","x":"1","impact":"10"},
        {"candidate_id":"E1","project_id":"1","test_condition_signature":"ISO","source_campaign":"OLD","source_round":"OLD","x":"99","impact":"999"},
    ]
    cstore=CampaignStore(tmp_path); rs=ExperimentalResultService(tmp_path); d=DatasetVersionStore(tmp_path)
    base=tmp_path/"base.csv"; write_base(base,base_rows)
    d.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=base)
    cstore.create(campaign_id="C1",project_id=1,name="demo",target_metrics=["impact"])
    r=cstore.add_round("C1",plan=plan(1))
    rs.register_planned_experiments("C1",round_id=r["round_id"],experiments=[planned("E1",3)])
    cstore.transition_round("C1",round_id=r["round_id"],new_status="RUNNING")
    rs.ingest("C1",round_id=r["round_id"],payload=result("E1","COMPLETED",12.5))
    cstore.transition_round("C1",round_id=r["round_id"],new_status="COMPLETED")
    with pytest.raises(DatasetVersionConflictError):
        d.update_from_round(campaign_store=cstore,campaign_id="C1",round_id=r["round_id"],new_dataset_version="dataset_v002")


def test_sha256_tampering_is_detected(tmp_path):
    d=DatasetVersionStore(tmp_path); p=tmp_path/"base.csv"; write_base(p)
    d.register_base_csv(project_id=1,dataset_version="dataset_v001",source_csv=p)
    with d.dataset_path(1,"dataset_v001").open("a",encoding="utf-8") as f:
        f.write("tampered")
    with pytest.raises(DatasetIntegrityError):
        d.verify(1,"dataset_v001")
