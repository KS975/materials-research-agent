from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def write_json(path:Path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def oracle(a,tough,temp,speed):
    return (
        21.5 + 0.30*a + 0.98*tough
        - 0.010*(temp-249.0)**2
        + 0.013*speed
        + 2.1*np.sin(a/6.0)
        - 0.7*np.cos(tough/2.7)
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='.runtime/v020/fixtures/t26')
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(260)
    pid=9026; cid='V020_T26_DEMO'
    fields=[
        'candidate_id','project_id','test_condition_signature','source_campaign','source_round',
        'formula::ABS','formula::PC','formula::增韧剂','process::加工温度','process::螺杆转速','冲击强度'
    ]

    base=[]
    for i in range(35):
        a=float(rng.uniform(22,40)); tough=float(rng.uniform(10,17)); pc=100-a-tough
        temp=float(rng.uniform(228,272)); speed=float(rng.uniform(190,315))
        y=float(oracle(a,tough,temp,speed)+rng.normal(0,0.55))
        base.append({
            'candidate_id':f'BASE_{i+1:03d}','project_id':pid,
            'test_condition_signature':'T26_STANDARD_23C','source_campaign':'BASE_IMPORT','source_round':'BASE',
            'formula::ABS':a,'formula::PC':pc,'formula::增韧剂':tough,
            'process::加工温度':temp,'process::螺杆转速':speed,'冲击强度':y,
        })
    with (out/'dataset_v001.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(base)

    campaign={
        'campaign_id':cid,'project_id':pid,'name':'V0.2 三轮闭环总验收',
        'target_metrics':['冲击强度'],'metadata':{'fixture':True,'stage':'V0.2-T26'}
    }
    plan={
        'planned_experiment_count':5,'dataset_version':'dataset_v001',
        'model_versions':{'冲击强度':'model_v001'},
        'search_space_snapshot':{'version':'search_space_v001'},
        'constraints_snapshot':{'version':'constraints_v001'},
        'optimizer_config':{'engine':'GaussianProcess','acquisition':'EI','batch_strategy':'kriging_believer'},
        'source':'V0.2-T26_initial_round',
    }

    # Round 1 deliberately probes a stronger region than the initial dataset.
    r1_feats=[
        (32.0,18.5,248.0,325.0),(34.0,19.0,249.0,340.0),(30.0,18.0,247.0,320.0),
        (36.0,19.5,250.0,350.0),(33.0,20.0,249.0,345.0),
    ]
    planned=[]; results=[]
    for i,(a,tough,temp,speed) in enumerate(r1_feats,start=1):
        pc=100-a-tough; actual=float(oracle(a,tough,temp,speed))
        cid_exp=f'V020_T26_R1_{i:02d}'
        features={
            'formula::ABS':a,'formula::PC':pc,'formula::增韧剂':tough,
            'process::加工温度':temp,'process::螺杆转速':speed,
        }
        planned.append({
            'candidate_id':cid_exp,'required_metrics':['冲击强度'],
            'expected_test_condition_signature':'T26_STANDARD_23C','units':{'冲击强度':'kJ/m²'},
            'features':features,
            'prediction_snapshot':{'冲击强度':{'value':actual-0.9+0.15*i,'posterior_std':1.1,'source':'T26_initial_fixture'}},
        })
        results.append({
            'candidate_id':cid_exp,'status':'COMPLETED','test_condition_signature':'T26_STANDARD_23C',
            'measurements':{'冲击强度':actual},'units':{'冲击强度':'kJ/m²'},
        })

    pool=[]
    # Put exact Round-1 IDs in the pool to exercise used-candidate filtering.
    for item in planned:
        row={'candidate_id':item['candidate_id'],'hard_valid':'true','soft_penalty':'0'}
        row.update(item['features']); pool.append(row)
    # Put exact base feature duplicates under new IDs to exercise feature dedupe.
    for i,row0 in enumerate(base[:5],start=1):
        pool.append({
            'candidate_id':f'V020_T26_DUP_BASE_{i:02d}','hard_valid':'true','soft_penalty':'0',
            'formula::ABS':row0['formula::ABS'],'formula::PC':row0['formula::PC'],'formula::增韧剂':row0['formula::增韧剂'],
            'process::加工温度':row0['process::加工温度'],'process::螺杆转速':row0['process::螺杆转速'],
        })
    for i in range(790):
        a=float(rng.uniform(23,41)); tough=float(rng.uniform(10.5,20.5)); pc=100-a-tough
        temp=float(rng.uniform(229,271)); speed=float(rng.uniform(200,355))
        pool.append({
            'candidate_id':f'V020_T26_POOL_{i+1:04d}',
            'hard_valid':'false' if i in {40,140,240,340,440} else 'true',
            'soft_penalty':'0.12' if tough>20 else '0',
            'formula::ABS':a,'formula::PC':pc,'formula::增韧剂':tough,
            'process::加工温度':temp,'process::螺杆转速':speed,
        })
    pfields=['candidate_id','hard_valid','soft_penalty','formula::ABS','formula::PC','formula::增韧剂','process::加工温度','process::螺杆转速']
    with (out/'candidate_pool.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=pfields); w.writeheader(); w.writerows(pool)

    gate={'decision':'PASS','training_allowed':True,'official_model_allowed':True}
    write_json(out/'campaign_create.json',campaign)
    write_json(out/'round1_plan.json',plan)
    write_json(out/'round1_planned_experiments.json',planned)
    write_json(out/'round1_results.json',results)
    write_json(out/'gate_pass.json',gate)
    write_json(out/'oracle_config.json',{'type':'synthetic_fixture_only','formula':'hidden deterministic T26 oracle','seed':260})

    print('V0.2-T26 FIXTURE BUILDER')
    print(f'base_dataset_csv: {out/"dataset_v001.csv"}')
    print('initial_rows: 35')
    print('rounds: 3')
    print('experiments_per_round: 5')
    print('expected_final_rows: 50')
    print(f'candidate_pool_rows: {len(pool)}')
    print('synthetic_oracle: fixture_only')
    print()
    print('V0.2-T26 FIXTURE BUILD PASS')

if __name__=='__main__': main()
