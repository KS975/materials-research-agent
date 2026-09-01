import json
import pytest
from experiments import (
    ModelPromotionConflictError, ModelPromotionValidationError,
    ModelRegistry, decide_promotion,
)

def gate(ok=True):
    return {
        "decision":"PASS" if ok else "FAIL",
        "training_allowed":ok,
        "official_model_allowed":ok,
    }

def cv(r2=0.8,std=0.05):
    return {"summary":{"r2":{"mean":r2,"std":std}}}

def test_promote_when_challenger_materially_improves():
    d=decide_promotion(
        gate=gate(),
        incumbent_holdout={"r2":0.70,"mae":2.0,"rmse":3.0},
        challenger_holdout={"r2":0.82,"mae":1.5,"rmse":2.2},
        challenger_cv=cv(0.78,0.08),
    )
    assert d["decision"]=="PROMOTE"

def test_keep_incumbent_when_challenger_degrades():
    d=decide_promotion(
        gate=gate(),
        incumbent_holdout={"r2":0.80,"mae":1.5,"rmse":2.0},
        challenger_holdout={"r2":0.60,"mae":2.3,"rmse":2.8},
        challenger_cv=cv(0.70,0.08),
    )
    assert d["decision"]=="KEEP_INCUMBENT"

def test_review_when_change_is_small():
    d=decide_promotion(
        gate=gate(),
        incumbent_holdout={"r2":0.80,"mae":1.50,"rmse":2.00},
        challenger_holdout={"r2":0.81,"mae":1.47,"rmse":1.96},
        challenger_cv=cv(0.78,0.08),
    )
    assert d["decision"]=="REVIEW_REQUIRED"

def test_gate_blocks_promotion():
    d=decide_promotion(
        gate=gate(False),
        incumbent_holdout={"r2":0.8,"mae":1.0,"rmse":1.2},
        challenger_holdout={"r2":0.9,"mae":0.5,"rmse":0.6},
        challenger_cv=cv(0.9,0.01),
    )
    assert d["decision"]=="BLOCKED"

def test_low_challenger_cv_keeps_incumbent():
    d=decide_promotion(
        gate=gate(),
        incumbent_holdout={"r2":0.5,"mae":3.0,"rmse":4.0},
        challenger_holdout={"r2":0.8,"mae":1.0,"rmse":2.0},
        challenger_cv=cv(0.2,0.05),
    )
    assert d["decision"]=="KEEP_INCUMBENT"

class DummyModel:
    pass

def test_registry_promotion_requires_human_approval(tmp_path):
    registry=ModelRegistry(tmp_path)
    registry.save_model(
        project_id=1,target_metric="impact",model_version="model_v001",
        model=DummyModel(),metadata={"dataset_version":"dataset_v001","created_at":"now"},
        make_active=True,
    )
    registry.save_model(
        project_id=1,target_metric="impact",model_version="model_v002",
        model=DummyModel(),metadata={"dataset_version":"dataset_v002","created_at":"now"},
        make_active=False,
    )
    before=registry.load_registry(1,"impact")
    assert before["active_model_version"]=="model_v001"
    assert before["models"]["model_v002"]["status"]=="CANDIDATE"

    after=registry.approve_promotion(
        project_id=1,target_metric="impact",
        challenger_model_version="model_v002",
        promotion_report={"decision":"PROMOTE"},
        approved_by="reviewer",
    )
    assert after["active_model_version"]=="model_v002"
    assert after["models"]["model_v001"]["status"]=="RETIRED"
    assert after["models"]["model_v002"]["status"]=="ACTIVE"

def test_registry_rejects_approval_without_promote(tmp_path):
    registry=ModelRegistry(tmp_path)
    with pytest.raises(ModelPromotionConflictError):
        registry.approve_promotion(
            project_id=1,target_metric="impact",
            challenger_model_version="model_v002",
            promotion_report={"decision":"REVIEW_REQUIRED"},
            approved_by="reviewer",
        )

def test_registry_requires_approver(tmp_path):
    registry=ModelRegistry(tmp_path)
    registry.save_model(
        project_id=1,target_metric="impact",model_version="model_v001",
        model=DummyModel(),metadata={"dataset_version":"dataset_v001","created_at":"now"},
        make_active=True,
    )
    registry.save_model(
        project_id=1,target_metric="impact",model_version="model_v002",
        model=DummyModel(),metadata={"dataset_version":"dataset_v002","created_at":"now"},
        make_active=False,
    )
    with pytest.raises(ModelPromotionValidationError):
        registry.approve_promotion(
            project_id=1,target_metric="impact",
            challenger_model_version="model_v002",
            promotion_report={"decision":"PROMOTE"},
            approved_by="",
        )
