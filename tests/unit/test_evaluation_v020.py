import math
from experiments import build_prediction_measurement_report, PredictionEvaluationService, CampaignStore

def campaign(rows):
    return {
        'campaign_id':'C1','project_id':1,'target_metrics':['impact'],
        'rounds':[{'round_id':'C1-R001','status':'COMPLETED','experiments':rows}],
    }

def exp(cid,pred,actual,status='COMPLETED',std=None):
    snap={'value':pred}
    if std is not None: snap['posterior_std']=std
    result=None
    if status=='COMPLETED':
        result={'training_eligible':True,'measurements':{'impact':actual},'units':{'impact':'kJ/m2'},'test_condition_signature':'ISO'}
    elif status=='FAILED':
        result={'training_eligible':False,'measurements':{},'units':{},'test_condition_signature':'ISO'}
    return {'candidate_id':cid,'status':status,'prediction_snapshot':{'impact':snap},'result':result}

def test_residual_definition_actual_minus_predicted():
    r=build_prediction_measurement_report(campaign([exp('E1',50,48)]),round_id='C1-R001',metric='impact')
    row=r['rows'][0]
    assert row['residual']==-2
    assert row['absolute_error']==2

def test_aggregate_mae_rmse_bias():
    rows=[exp('E1',10,12),exp('E2',20,19),exp('E3',30,33)]
    r=build_prediction_measurement_report(campaign(rows),round_id='C1-R001',metric='impact')
    assert math.isclose(r['aggregate']['mae'],2.0)
    assert math.isclose(r['aggregate']['rmse'],math.sqrt(14/3))
    assert math.isclose(r['aggregate']['bias_actual_minus_predicted'],4/3)

def test_r2_is_computed_without_sklearn_dependency():
    rows=[exp('E1',9,10),exp('E2',21,20),exp('E3',29,30)]
    r=build_prediction_measurement_report(campaign(rows),round_id='C1-R001',metric='impact')
    assert r['aggregate']['r2'] is not None
    assert r['aggregate']['r2'] > 0.9

def test_failed_is_excluded():
    rows=[exp('E1',10,12),exp('E2',20,None,status='FAILED')]
    r=build_prediction_measurement_report(campaign(rows),round_id='C1-R001',metric='impact')
    assert r['counts']['planned_experiments']==2
    assert r['counts']['eligible_completed_experiments']==1
    assert r['counts']['evaluated']==1

def test_uncertainty_coverage_and_overconfidence():
    rows=[exp('E1',10,10.5,std=1),exp('E2',10,13,std=1),exp('E3',10,11.5,std=1)]
    r=build_prediction_measurement_report(campaign(rows),round_id='C1-R001',metric='impact')
    u=r['uncertainty']
    assert u['samples_with_std']==3
    assert math.isclose(u['coverage_1sigma'],1/3)
    assert math.isclose(u['coverage_2sigma'],2/3)
    assert u['overconfident_candidate_ids']==['E2']

def test_missing_std_is_allowed():
    r=build_prediction_measurement_report(campaign([exp('E1',10,11,std=None)]),round_id='C1-R001',metric='impact')
    assert r['rows'][0]['prediction_std'] is None
    assert r['uncertainty']['samples_with_std']==0
    assert r['uncertainty']['coverage_2sigma'] is None

def test_missing_prediction_is_excluded_not_fabricated():
    row=exp('E1',10,11); row['prediction_snapshot']={}
    r=build_prediction_measurement_report(campaign([row,exp('E2',20,21)]),round_id='C1-R001',metric='impact')
    assert r['counts']['eligible_completed_experiments']==2
    assert r['counts']['evaluated']==1
    assert r['excluded']['missing_prediction_candidate_ids']==['E1']

def test_zero_actual_relative_error_is_none():
    r=build_prediction_measurement_report(campaign([exp('E1',1,0)]),round_id='C1-R001',metric='impact')
    assert r['rows'][0]['relative_absolute_error'] is None
    assert r['aggregate']['mean_relative_absolute_error'] is None

def test_service_persists_report(tmp_path):
    store=CampaignStore(tmp_path)
    store.create(campaign_id='C1',project_id=1,name='demo',target_metrics=['impact'])
    c=store.load('C1')
    c['rounds']=[{'round_id':'C1-R001','round_no':1,'status':'COMPLETED','experiments':[exp('E1',10,11)]}]
    c['current_round_no']=1
    store.save(c)
    svc=PredictionEvaluationService(tmp_path)
    r=svc.evaluate('C1',round_id='C1-R001',metric='impact',persist=True)
    assert svc.report_path('C1',round_id='C1-R001',metric='impact').exists()
    assert 'report_json' in r
