from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from experiments import CampaignStore, ExperimentalResultService, PredictionEvaluationService

def load(path: Path): return json.loads(path.read_text(encoding='utf-8'))

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--fixture-dir',default='.runtime/v020/fixtures/t21')
    p.add_argument('--runtime-root',default='.runtime')
    p.add_argument('--reset',action='store_true')
    args=p.parse_args(); fixture=Path(args.fixture_dir)
    c=load(fixture/'campaign_create.json'); plan=load(fixture/'round_plan.json')
    planned=load(fixture/'planned_experiments.json'); results=load(fixture/'results.json')
    store=CampaignStore(args.runtime_root); ingest=ExperimentalResultService(args.runtime_root)
    evaluator=PredictionEvaluationService(args.runtime_root); cid=c['campaign_id']
    if args.reset:
        d=store.campaign_dir(cid)
        if d.exists(): shutil.rmtree(d)
        e=Path(args.runtime_root)/'v020'/'evaluations'/cid
        if e.exists(): shutil.rmtree(e)

    store.create(campaign_id=cid,project_id=c['project_id'],name=c['name'],target_metrics=c['target_metrics'],metadata=c.get('metadata'))
    r=store.add_round(cid,plan=plan)
    ingest.register_planned_experiments(cid,round_id=r['round_id'],experiments=planned)
    store.transition_round(cid,round_id=r['round_id'],new_status='RUNNING')
    for payload in results: ingest.ingest(cid,round_id=r['round_id'],payload=payload)
    store.transition_round(cid,round_id=r['round_id'],new_status='COMPLETED')

    report=evaluator.evaluate(cid,round_id=r['round_id'],metric='冲击强度',persist=True)
    a=report['aggregate']; u=report['uncertainty']; le=report['largest_error']; counts=report['counts']

    print('V0.2-T21 PREDICTION VS MEASUREMENT')
    print(f'campaign_id: {cid}')
    print(f'round_id: {r["round_id"]}')
    print('metric: 冲击强度')
    print()
    print('COUNTS')
    print(f'planned_experiments: {counts["planned_experiments"]}')
    print(f'eligible_completed_experiments: {counts["eligible_completed_experiments"]}')
    print(f'evaluated: {counts["evaluated"]}')
    print(f'excluded_missing_prediction: {counts["excluded_missing_prediction"]}')
    print()
    print('AGGREGATE ERROR')
    print(f'MAE: {a["mae"]:.6f}')
    print(f'RMSE: {a["rmse"]:.6f}')
    print(f'Bias(actual-predicted): {a["bias_actual_minus_predicted"]:.6f}')
    print(f'R2: {a["r2"]:.6f}')
    print(f'MRAE: {a["mean_relative_absolute_error"]:.6f}')
    print()
    print('LARGEST ERROR')
    print(f'candidate_id: {le["candidate_id"]}')
    print(f'predicted: {le["predicted"]:.6f}')
    print(f'actual: {le["actual"]:.6f}')
    print(f'residual: {le["residual"]:.6f}')
    print(f'absolute_error: {le["absolute_error"]:.6f}')
    print()
    print('UNCERTAINTY DIAGNOSTIC')
    print(f'samples_with_std: {u["samples_with_std"]}')
    print(f'coverage_1sigma: {u["coverage_1sigma"]:.6f}')
    print(f'coverage_2sigma: {u["coverage_2sigma"]:.6f}')
    print(f'mean_absolute_z_score: {u["mean_absolute_z_score"]:.6f}')
    print(f'overconfident_2sigma_miss_count: {u["overconfident_2sigma_miss_count"]}')
    print('overconfident_candidate_ids: '+json.dumps(u['overconfident_candidate_ids'],ensure_ascii=False))
    print()
    print('PER-EXPERIMENT')
    for row in report['rows']:
        std='NA' if row['prediction_std'] is None else f'{row["prediction_std"]:.3f}'
        print(f'{row["candidate_id"]} | predicted={row["predicted"]:.3f} | actual={row["actual"]:.3f} | residual={row["residual"]:+.3f} | abs_error={row["absolute_error"]:.3f} | std={std}')
    print()
    print('OUTPUT')
    print('report_json: '+report['report_json'])
    print()
    print('NOTE: residual = actual - predicted. T21 evaluates prediction quality only; it does not retrain or promote a model.')
    print()
    print('V0.2-T21 PREDICTION VS MEASUREMENT PASS')

if __name__=='__main__': main()
