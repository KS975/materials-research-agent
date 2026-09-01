import pytest
from experiments import CampaignConflictError, CampaignStore, CampaignValidationError

def valid_plan(dataset_version="dataset_v001"):
    return {"planned_experiment_count":5,"dataset_version":dataset_version,"model_versions":{"impact":"model_v001"},"search_space_snapshot":{"version":"ss_v001"},"constraints_snapshot":{"version":"c_v001"},"optimizer_config":{"acquisition":"EI"}}

def create(store):
    return store.create(campaign_id="C001",project_id=1,name="demo",target_metrics=["impact"])

def test_create_campaign_and_reload(tmp_path):
    s=CampaignStore(tmp_path); create(s); x=s.load("C001")
    assert x["status"]=="ACTIVE" and x["current_round_no"]==0 and x["events"][0]["event_type"]=="CAMPAIGN_CREATED"

def test_duplicate_campaign_is_blocked(tmp_path):
    s=CampaignStore(tmp_path); create(s)
    with pytest.raises(CampaignConflictError): create(s)

def test_round_stores_required_snapshots(tmp_path):
    s=CampaignStore(tmp_path); create(s); r=s.add_round("C001",plan=valid_plan())
    assert r["status"]=="PLANNED"
    assert r["plan"]["dataset_version"]=="dataset_v001"
    assert r["plan"]["model_versions"]["impact"]=="model_v001"
    assert r["plan"]["search_space_snapshot"]["version"]=="ss_v001"
    assert r["plan"]["constraints_snapshot"]["version"]=="c_v001"
    assert r["plan"]["optimizer_config"]["acquisition"]=="EI"

def test_next_round_blocked_until_previous_completed(tmp_path):
    s=CampaignStore(tmp_path); create(s); r1=s.add_round("C001",plan=valid_plan("dataset_v001"))
    with pytest.raises(CampaignConflictError): s.add_round("C001",plan=valid_plan("dataset_v002"))
    s.transition_round("C001",round_id=r1["round_id"],new_status="RUNNING")
    s.transition_round("C001",round_id=r1["round_id"],new_status="COMPLETED")
    r2=s.add_round("C001",plan=valid_plan("dataset_v002"))
    assert r2["round_no"]==2 and r2["status"]=="PLANNED"

def test_illegal_round_transition_is_blocked(tmp_path):
    s=CampaignStore(tmp_path); create(s); r=s.add_round("C001",plan=valid_plan())
    with pytest.raises(CampaignConflictError): s.transition_round("C001",round_id=r["round_id"],new_status="COMPLETED")

def test_round_transition_is_persisted(tmp_path):
    s=CampaignStore(tmp_path); create(s); r=s.add_round("C001",plan=valid_plan())
    s.transition_round("C001",round_id=r["round_id"],new_status="RUNNING")
    x=s.load("C001")
    assert x["rounds"][0]["status"]=="RUNNING" and x["rounds"][0]["started_at"] is not None and x["events"][-1]["event_type"]=="ROUND_STATUS_CHANGED"

def test_invalid_round_plan_is_rejected(tmp_path):
    s=CampaignStore(tmp_path); create(s); p=valid_plan(); p["planned_experiment_count"]=0
    with pytest.raises(CampaignValidationError): s.add_round("C001",plan=p)
