from __future__ import annotations
import argparse, json
from pathlib import Path

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='.runtime/v020/fixtures/t21')
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    campaign={
        'campaign_id':'V020_T21_DEMO','project_id':9021,
        'name':'预测值与实测值评估演示','target_metrics':['冲击强度'],
        'metadata':{'fixture':True,'purpose':'V0.2-T21 engineering acceptance'},
    }
    plan={
        'planned_experiment_count':7,'dataset_version':'dataset_v001',
        'model_versions':{'冲击强度':'gp_model_v001'},
        'search_space_snapshot':{'version':'search_space_v001'},
        'constraints_snapshot':{'version':'constraints_v001'},
        'optimizer_config':{'engine':'GaussianProcess','acquisition':'EI'},
        'source':'V0.1.4-T18',
    }

    predicted=[50.0,52.0,48.0,55.0,49.0,51.0,47.0]
    stds=[1.0,1.0,1.5,1.0,None,0.8,1.2]
    actual=[49.0,53.5,46.0,60.0,49.5,50.5,None]
    planned=[]; results=[]
    for i in range(7):
        snap={'value':predicted[i],'source':'GP posterior mean'}
        if stds[i] is not None: snap['posterior_std']=stds[i]
        cid=f'V020_T21_EXP_{i+1:02d}'
        planned.append({
            'candidate_id':cid,'required_metrics':['冲击强度'],
            'expected_test_condition_signature':'T21_STANDARD_23C',
            'units':{'冲击强度':'kJ/m²'},
            'features':{'x1':i+1,'x2':100-i},
            'prediction_snapshot':{'冲击强度':snap},
        })
        if i < 6:
            results.append({
                'candidate_id':cid,'status':'COMPLETED',
                'test_condition_signature':'T21_STANDARD_23C',
                'measurements':{'冲击强度':actual[i]},
                'units':{'冲击强度':'kJ/m²'},
            })
        else:
            results.append({
                'candidate_id':cid,'status':'FAILED',
                'test_condition_signature':'T21_STANDARD_23C',
                'measurements':{},'units':{},
                'failure_reason':'fixture: specimen preparation failed',
            })

    write_json(out/'campaign_create.json',campaign)
    write_json(out/'round_plan.json',plan)
    write_json(out/'planned_experiments.json',planned)
    write_json(out/'results.json',results)
    print('V0.2-T21 FIXTURE BUILDER')
    print(f'campaign_create: {out/"campaign_create.json"}')
    print(f'round_plan: {out/"round_plan.json"}')
    print(f'planned_experiments: {out/"planned_experiments.json"}')
    print(f'results: {out/"results.json"}')
    print()
    print('EXPECTED')
    print('- 6 COMPLETED experiments enter evaluation')
    print('- 1 FAILED experiment excluded from error metrics')
    print('- one deliberately overconfident GP point should be diagnosed')
    print()
    print('V0.2-T21 FIXTURE BUILD PASS')

if __name__=='__main__': main()
