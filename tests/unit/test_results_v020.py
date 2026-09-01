import pytest
from experiments import (
    CampaignConflictError, CampaignStore,
    ExperimentalResultConflictError, ExperimentalResultNotFoundError,
    ExperimentalResultService, ExperimentalResultValidationError,
)

def round_plan(count=2):
    return {
        "planned_experiment_count":count,"dataset_version":"dataset_v001",
        "model_versions":{"impact":"model_v001"},
        "search_space_snapshot":{"version":"ss_v001"},
        "constraints_snapshot":{"version":"c_v001"},
        "optimizer_config":{"acquisition":"EI"},
    }

def planned(cid):
    return {
        "candidate_id":cid,"required_metrics":["impact"],
        "expected_test_condition_signature":"ISO_23C",
        "units":{"impact":"kJ/m2"},"features":{"x":1.0},
        "prediction_snapshot":{"impact":{"value":50.0}},
    }

def completed(cid,value=49.0):
    return {
        "candidate_id":cid,"status":"COMPLETED",
        "test_condition_signature":"ISO_23C",
        "measurements":{"impact":value},"units":{"impact":"kJ/m2"},
    }

def setup_round(tmp_path,count=2):
    store=CampaignStore(tmp_path); service=ExperimentalResultService(tmp_path)
    store.create(campaign_id="C001",project_id=1,name="demo",target_metrics=["impact"])
    r=store.add_round("C001",plan=round_plan(count))
    service.register_planned_experiments("C001",round_id=r["round_id"],experiments=[planned(f"E{i+1}") for i in range(count)])
    store.transition_round("C001",round_id=r["round_id"],new_status="RUNNING")
    return store,service,r

def test_completed_result_is_persisted_and_training_eligible(tmp_path):
    store,service,r=setup_round(tmp_path)
    result=service.ingest("C001",round_id=r["round_id"],payload=completed("E1",48.7))
    assert result["experiment"]["status"]=="COMPLETED"
    assert result["experiment"]["result"]["measurements"]["impact"]==48.7
    assert result["experiment"]["result"]["training_eligible"] is True
    assert store.load("C001")["rounds"][0]["status"]=="PARTIALLY_COMPLETED"

def test_exact_duplicate_is_idempotent(tmp_path):
    _,service,r=setup_round(tmp_path); payload=completed("E1",48.7)
    service.ingest("C001",round_id=r["round_id"],payload=payload)
    replay=service.ingest("C001",round_id=r["round_id"],payload=payload)
    assert replay["idempotent_replay"] is True

def test_conflicting_duplicate_is_rejected(tmp_path):
    _,service,r=setup_round(tmp_path)
    service.ingest("C001",round_id=r["round_id"],payload=completed("E1",48.7))
    with pytest.raises(ExperimentalResultConflictError):
        service.ingest("C001",round_id=r["round_id"],payload=completed("E1",49.9))

def test_unknown_candidate_is_rejected(tmp_path):
    _,service,r=setup_round(tmp_path)
    with pytest.raises(ExperimentalResultNotFoundError):
        service.ingest("C001",round_id=r["round_id"],payload=completed("UNKNOWN",49.0))

def test_non_numeric_completed_measurement_is_rejected(tmp_path):
    _,service,r=setup_round(tmp_path); payload=completed("E1"); payload["measurements"]["impact"]="high"
    with pytest.raises(ExperimentalResultValidationError):
        service.ingest("C001",round_id=r["round_id"],payload=payload)

def test_condition_and_unit_mismatch_are_rejected(tmp_path):
    _,service,r=setup_round(tmp_path)
    bad=completed("E1"); bad["test_condition_signature"]="ASTM_25C"
    with pytest.raises(ExperimentalResultConflictError):
        service.ingest("C001",round_id=r["round_id"],payload=bad)
    bad=completed("E1"); bad["units"]["impact"]="J/m"
    with pytest.raises(ExperimentalResultConflictError):
        service.ingest("C001",round_id=r["round_id"],payload=bad)

def test_failed_invalid_not_tested_are_not_training_eligible(tmp_path):
    _,service,r=setup_round(tmp_path,count=3)
    payloads=[
        {"candidate_id":"E1","status":"FAILED","test_condition_signature":"ISO_23C","measurements":{},"units":{},"failure_reason":"specimen broke"},
        {"candidate_id":"E2","status":"INVALID","test_condition_signature":"ISO_23C","measurements":{},"units":{},"failure_reason":"instrument invalid"},
        {"candidate_id":"E3","status":"NOT_TESTED","test_condition_signature":"ISO_23C","measurements":{},"units":{}},
    ]
    for p in payloads: service.ingest("C001",round_id=r["round_id"],payload=p)
    s=service.summary("C001",round_id=r["round_id"])
    assert s["progress"]["training_eligible"]==0
    assert s["progress"]["failed"]==1 and s["progress"]["invalid"]==1 and s["progress"]["not_tested"]==1

def test_failed_result_cannot_fake_zero_measurement(tmp_path):
    _,service,r=setup_round(tmp_path)
    payload={"candidate_id":"E1","status":"FAILED","test_condition_signature":"ISO_23C","measurements":{"impact":0},"units":{"impact":"kJ/m2"},"failure_reason":"failed"}
    with pytest.raises(ExperimentalResultValidationError):
        service.ingest("C001",round_id=r["round_id"],payload=payload)

def test_round_cannot_complete_until_all_experiments_terminal(tmp_path):
    store,service,r=setup_round(tmp_path)
    service.ingest("C001",round_id=r["round_id"],payload=completed("E1",48.7))
    with pytest.raises(CampaignConflictError):
        store.transition_round("C001",round_id=r["round_id"],new_status="COMPLETED")
    service.ingest("C001",round_id=r["round_id"],payload=completed("E2",50.1))
    store.transition_round("C001",round_id=r["round_id"],new_status="COMPLETED")
    loaded=store.load("C001")
    assert loaded["rounds"][0]["status"]=="COMPLETED"
    assert loaded["rounds"][0]["progress"]["pending"]==0
