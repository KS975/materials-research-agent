import pytest
from experiments import (
    EndToEndValidationError,
    ensure_monotonic_best,
    feature_key,
    summarize_experiment_integrity,
    summarize_model_decisions,
    verify_dataset_lineage_manifests,
)


def test_feature_key_is_deterministic():
    f={'a':1.00000000001,'b':2.0}
    assert feature_key(f,['a','b']) == feature_key(dict(f),['a','b'])


def test_feature_key_rejects_missing_feature():
    with pytest.raises(EndToEndValidationError):
        feature_key({'a':1},['a','b'])


def test_dataset_lineage_valid_chain():
    manifests=[
        {'dataset_version':'v1','parent_dataset_version':None,'sha256':'h1'},
        {'dataset_version':'v2','parent_dataset_version':'v1','parent_sha256':'h1','sha256':'h2'},
        {'dataset_version':'v3','parent_dataset_version':'v2','parent_sha256':'h2','sha256':'h3'},
    ]
    assert verify_dataset_lineage_manifests(manifests)['valid'] is True


def test_dataset_lineage_detects_broken_parent_hash():
    manifests=[
        {'dataset_version':'v1','parent_dataset_version':None,'sha256':'h1'},
        {'dataset_version':'v2','parent_dataset_version':'v1','parent_sha256':'WRONG','sha256':'h2'},
    ]
    result=verify_dataset_lineage_manifests(manifests)
    assert result['valid'] is False
    assert result['errors']


def test_monotonic_best_for_maximize_and_minimize():
    assert ensure_monotonic_best([1,1,2,3],'maximize') is True
    assert ensure_monotonic_best([5,4,4,2],'minimize') is True
    assert ensure_monotonic_best([1,3,2],'maximize') is False


def test_experiment_integrity_detects_duplicates():
    campaign={'rounds':[{'experiments':[
        {'candidate_id':'A','status':'COMPLETED','features':{'x':1.0},'result':{'training_eligible':True}},
        {'candidate_id':'A','status':'COMPLETED','features':{'x':1.0},'result':{'training_eligible':True}},
    ]}]}
    s=summarize_experiment_integrity(campaign,feature_columns=['x'])
    assert s['duplicate_candidate_id_count']==1
    assert s['duplicate_feature_point_count']==1


def test_experiment_integrity_counts_training_rows():
    campaign={'rounds':[{'experiments':[
        {'candidate_id':'A','status':'COMPLETED','features':{'x':1.0},'result':{'training_eligible':True}},
        {'candidate_id':'B','status':'FAILED','features':{'x':2.0},'result':{'training_eligible':False}},
    ]}]}
    s=summarize_experiment_integrity(campaign,feature_columns=['x'])
    assert s['terminal_count']==2
    assert s['training_eligible_count']==1
    assert s['result_missing_count']==0


def test_model_decision_summary_counts_blocked():
    s=summarize_model_decisions([
        {'decision':'PROMOTE'},{'decision':'KEEP_INCUMBENT'},{'decision':'BLOCKED'},
    ])
    assert s['count']==3
    assert s['promote_count']==1
    assert s['keep_incumbent_count']==1
    assert s['blocked_count']==1
