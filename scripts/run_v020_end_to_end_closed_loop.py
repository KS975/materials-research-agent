from __future__ import annotations
import argparse,json,math,shutil
from pathlib import Path
import numpy as np

from experiments import (
    CampaignStore, DatasetVersionStore, ExperimentalResultService,
    PredictionEvaluationService, ResumableClosedLoopWorkflow,
    ModelPromotionService, EndToEndAuditService,
)


def load_json(path:Path):
    return json.loads(path.read_text(encoding='utf-8'))


def oracle(features):
    a=float(features['formula::ABS']); tough=float(features['formula::增韧剂'])
    temp=float(features['process::加工温度']); speed=float(features['process::螺杆转速'])
    return float(
        21.5 + 0.30*a + 0.98*tough
        - 0.010*(temp-249.0)**2
        + 0.013*speed
        + 2.1*np.sin(a/6.0)
        - 0.7*np.cos(tough/2.7)
    )


def synthesize_round_results(round_record, round_no):
    # Deterministic synthetic fixture measurement. Production must never use this.
    results=[]
    for i,experiment in enumerate(round_record.get('experiments') or [],start=1):
        actual=oracle(experiment['features']) + 0.04*math.sin(round_no*10+i)
        results.append({
            'candidate_id':experiment['candidate_id'],'status':'COMPLETED',
            'test_condition_signature':experiment['expected_test_condition_signature'],
            'measurements':{'冲击强度':actual},'units':{'冲击强度':'kJ/m²'},
            'notes':'T26 synthetic fixture measurement; not a real experiment',
        })
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixture-dir',default='.runtime/v020/fixtures/t26')
    parser.add_argument('--runtime-root',default='.runtime')
    parser.add_argument('--reset',action='store_true')
    args=parser.parse_args(); fixture=Path(args.fixture_dir)

    cdoc=load_json(fixture/'campaign_create.json'); plan=load_json(fixture/'round1_plan.json')
    r1_planned=load_json(fixture/'round1_planned_experiments.json'); r1_results=load_json(fixture/'round1_results.json')
    gate=load_json(fixture/'gate_pass.json')
    pid=int(cdoc['project_id']); cid=cdoc['campaign_id']; runtime=Path(args.runtime_root)

    campaigns=CampaignStore(runtime); datasets=DatasetVersionStore(runtime)
    results=ExperimentalResultService(runtime); evaluations=PredictionEvaluationService(runtime)
    promotion=ModelPromotionService(runtime)

    if args.reset:
        paths=[
            campaigns.campaign_dir(cid),datasets.project_dir(pid),
            runtime/'v020'/'models'/f'project_{pid}',runtime/'v020'/'model_promotion'/f'project_{pid}',
            runtime/'v020'/'closed_loop_bo'/cid,runtime/'v020'/'checkpoints'/cid,
            runtime/'v020'/'evaluations'/cid,runtime/'v020'/'end_to_end'/cid,
        ]
        for path in paths:
            if path.exists(): shutil.rmtree(path)

    print('V0.2-T26 END-TO-END CLOSED LOOP')
    print(f'campaign_id: {cid}')
    print(f'project_id: {pid}')
    print('target_metric: 冲击强度')
    print()

    base=datasets.register_base_csv(
        project_id=pid,dataset_version='dataset_v001',source_csv=fixture/'dataset_v001.csv',
        metadata={'fixture':True,'stage':'T26'},
    )
    campaigns.create(
        campaign_id=cid,project_id=pid,name=cdoc['name'],target_metrics=cdoc['target_metrics'],metadata=cdoc.get('metadata')
    )
    r1=campaigns.add_round(cid,plan=plan)
    results.register_planned_experiments(cid,round_id=r1['round_id'],experiments=r1_planned)
    campaigns.transition_round(cid,round_id=r1['round_id'],new_status='RUNNING')
    for payload in r1_results: results.ingest(cid,round_id=r1['round_id'],payload=payload)
    campaigns.transition_round(cid,round_id=r1['round_id'],new_status='COMPLETED')
    eval1=evaluations.evaluate(cid,round_id=r1['round_id'],metric='冲击强度')

    print('ROUND 1')
    print(f'round_id: {r1["round_id"]}')
    print('status: COMPLETED')
    print('terminal_results: 5')
    print(f'MAE: {eval1["aggregate"]["mae"]:.6f}')
    print()

    # T25 is exercised inside the full T26 flow: pause after dataset update, then resume.
    wf1=ResumableClosedLoopWorkflow(runtime)
    paused=wf1.resume(
        campaign_id=cid,source_round_id=r1['round_id'],
        parent_dataset_version='dataset_v001',child_dataset_version='dataset_v002',
        candidate_pool_csv=fixture/'candidate_pool.csv',target_metric='冲击强度',target_unit='kJ/m²',
        gate=gate,incumbent_model_version='model_v001',challenger_model_version='model_v002',
        model_family='ExtraTreesRegressor',batch_size=5,random_state=42,pause_after_step='DATASET_UPDATED',
    )
    wf1=ResumableClosedLoopWorkflow(runtime)
    trans1=wf1.resume(
        campaign_id=cid,source_round_id=r1['round_id'],
        parent_dataset_version='dataset_v001',child_dataset_version='dataset_v002',
        candidate_pool_csv=fixture/'candidate_pool.csv',target_metric='冲击强度',target_unit='kJ/m²',
        gate=gate,incumbent_model_version='model_v001',challenger_model_version='model_v002',
        model_family='ExtraTreesRegressor',batch_size=5,random_state=42,
    )
    r2=campaigns.load(cid)['rounds'][-1]
    print('ROUND 1 -> ROUND 2 TRANSITION')
    print(f'checkpoint_pause_after: {paused["last_completed_step"]}')
    print(f'checkpoint_resume_status: {trans1["status"]}')
    print(f'dataset_v002_rows: {datasets.load_manifest(pid,"dataset_v002")["row_count"]}')
    print(f'model_decision: {trans1["model_decision"]}')
    print(f'round2_id: {r2["round_id"]}')
    print('round2_status: PLANNED')
    print()

    campaigns.transition_round(cid,round_id=r2['round_id'],new_status='RUNNING')
    for payload in synthesize_round_results(campaigns.load(cid)['rounds'][-1],2):
        results.ingest(cid,round_id=r2['round_id'],payload=payload)
    campaigns.transition_round(cid,round_id=r2['round_id'],new_status='COMPLETED')
    eval2=evaluations.evaluate(cid,round_id=r2['round_id'],metric='冲击强度')
    print('ROUND 2')
    print(f'round_id: {r2["round_id"]}')
    print('status: COMPLETED')
    print('terminal_results: 5')
    print(f'MAE: {eval2["aggregate"]["mae"]:.6f}')
    print()

    trans2=ResumableClosedLoopWorkflow(runtime).resume(
        campaign_id=cid,source_round_id=r2['round_id'],
        parent_dataset_version='dataset_v002',child_dataset_version='dataset_v003',
        candidate_pool_csv=fixture/'candidate_pool.csv',target_metric='冲击强度',target_unit='kJ/m²',
        gate=gate,incumbent_model_version='model_v001',challenger_model_version='model_v003',
        model_family='ExtraTreesRegressor',batch_size=5,random_state=43,
    )
    r3=campaigns.load(cid)['rounds'][-1]
    print('ROUND 2 -> ROUND 3 TRANSITION')
    print(f'dataset_v003_rows: {datasets.load_manifest(pid,"dataset_v003")["row_count"]}')
    print(f'model_decision: {trans2["model_decision"]}')
    print(f'round3_id: {r3["round_id"]}')
    print('round3_status: PLANNED')
    print()

    campaigns.transition_round(cid,round_id=r3['round_id'],new_status='RUNNING')
    for payload in synthesize_round_results(campaigns.load(cid)['rounds'][-1],3):
        results.ingest(cid,round_id=r3['round_id'],payload=payload)
    campaigns.transition_round(cid,round_id=r3['round_id'],new_status='COMPLETED')
    eval3=evaluations.evaluate(cid,round_id=r3['round_id'],metric='冲击强度')
    final_update=datasets.update_from_round(
        campaign_store=campaigns,campaign_id=cid,round_id=r3['round_id'],new_dataset_version='dataset_v004'
    )
    final_model=promotion.compare_and_register(
        project_id=pid,target_metric='冲击强度',parent_dataset_version='dataset_v003',child_dataset_version='dataset_v004',
        incumbent_model_version='model_v001',challenger_model_version='model_v004',model_family='ExtraTreesRegressor',
        gate=gate,folds=5,random_state=44,
    )
    campaigns.complete_campaign(cid)

    print('ROUND 3 + FINALIZATION')
    print(f'round_id: {r3["round_id"]}')
    print('status: COMPLETED')
    print('terminal_results: 5')
    print(f'MAE: {eval3["aggregate"]["mae"]:.6f}')
    print(f'dataset_v004_rows: {datasets.load_manifest(pid,"dataset_v004")["row_count"]}')
    print(f'final_model_decision: {final_model["decision"]}')
    print('campaign_status: COMPLETED')
    print()

    report=EndToEndAuditService(runtime).build_report(
        campaign_id=cid,dataset_versions=['dataset_v001','dataset_v002','dataset_v003','dataset_v004'],
        target_metric='冲击强度',direction='maximize',evaluation_reports=[eval1,eval2,eval3],
        model_decisions=[
            {'decision':trans1['model_decision']},{'decision':trans2['model_decision']},{'decision':final_model['decision']},
        ],
        checkpoint_reports=[trans1,trans2],bo_reports=[trans1.get('bo_report') or {},trans2.get('bo_report') or {}],
        expected_rounds=3,expected_experiments_per_round=5,persist=True,fixture=True,
    )

    print('END-TO-END AUDIT')
    print(f'decision: {report["decision"]}')
    print(f'round_count: {report["round_count"]}')
    print(f'dataset_versions: {json.dumps(report["datasets"]["versions"],ensure_ascii=False)}')
    print(f'dataset_row_counts: {json.dumps(report["datasets"]["row_counts"])}')
    print(f'initial_best: {report["best_so_far"]["initial"]:.6f}')
    print(f'final_best: {report["best_so_far"]["final"]:.6f}')
    print(f'net_improvement: {report["best_so_far"]["net_improvement"]:.6f}')
    print(f'total_experiments: {report["experiment_integrity"]["experiment_count"]}')
    print(f'duplicate_candidate_ids: {report["experiment_integrity"]["duplicate_candidate_id_count"]}')
    print(f'duplicate_feature_points: {report["experiment_integrity"]["duplicate_feature_point_count"]}')
    print(f'model_blocked_count: {report["model_governance"]["blocked_count"]}')
    print(f'checkpoint_completed_count: {report["checkpoints"]["completed_count"]}')
    print(f'bo_ood_selected_count: {report["bayesian_optimization"]["out_of_domain_selected_count"]}')
    print()
    print('OUTPUT')
    print(f'report_json: {report["report_json"]}')
    print(f'campaign_json: {campaigns.campaign_path(cid)}')
    print()
    print('NOTE: T26 中 Round 2/3 的 measurement 来自 synthetic oracle，仅用于工程验收，不是真实材料实验结果。')

    if report['decision']!='PASS':
        failed=[k for k,v in report['checks'].items() if not v]
        raise SystemExit(f'ERROR: T26 audit failed checks: {failed}')
    if report['datasets']['row_counts'] != [35,40,45,50]:
        raise SystemExit('ERROR: dataset row-count chain mismatch')
    if report['experiment_integrity']['experiment_count'] != 15:
        raise SystemExit('ERROR: expected 15 experiments')
    if report['campaign_status'] != 'COMPLETED':
        raise SystemExit('ERROR: campaign not completed')

    print('V0.2-T26 END-TO-END CLOSED LOOP PASS')

if __name__=='__main__': main()
