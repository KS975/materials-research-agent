from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from experiments import CampaignConflictError, CampaignStore

def load_json(path):
    if not path.exists(): raise SystemExit(f"ERROR: file not found: {path}")
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict): raise SystemExit(f"ERROR: expected JSON object: {path}")
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--campaign-json",default=".runtime/v020/fixtures/t19/campaign_create.json")
    p.add_argument("--round1-json",default=".runtime/v020/fixtures/t19/round1_plan.json")
    p.add_argument("--round2-json",default=".runtime/v020/fixtures/t19/round2_plan.json")
    p.add_argument("--runtime-root",default=".runtime")
    p.add_argument("--reset",action="store_true")
    a=p.parse_args()
    cdoc=load_json(Path(a.campaign_json)); r1p=load_json(Path(a.round1_json)); r2p=load_json(Path(a.round2_json))
    store=CampaignStore(a.runtime_root); cid=cdoc["campaign_id"]
    if a.reset:
        d=store.campaign_dir(cid)
        if d.exists(): shutil.rmtree(d)
    print("V0.2-T19 EXPERIMENT CAMPAIGN + ROUND")
    print(f"campaign_id: {cid}"); print(f"project_id: {cdoc['project_id']}\n")
    campaign=store.create(campaign_id=cid,project_id=cdoc["project_id"],name=cdoc["name"],target_metrics=cdoc["target_metrics"],metadata=cdoc.get("metadata"))
    print("CAMPAIGN"); print(f"status: {campaign['status']}"); print(f"round_count: {len(campaign['rounds'])}\n")
    r1=store.add_round(cid,plan=r1p)
    print("ROUND 1"); print(f"round_id: {r1['round_id']}"); print(f"status: {r1['status']}"); print(f"planned_experiments: {r1['plan']['planned_experiment_count']}"); print(f"dataset_version: {r1['plan']['dataset_version']}"); print(f"model_versions: {json.dumps(r1['plan']['model_versions'],ensure_ascii=False)}\n")
    blocked=False
    try: store.add_round(cid,plan=r2p)
    except CampaignConflictError as exc:
        blocked=True; print("ROUND 2 EARLY-CREATE GUARD"); print("blocked: true"); print(f"reason: {exc}\n")
    if not blocked: raise SystemExit("ERROR: Round 2 should have been blocked")
    running=store.transition_round(cid,round_id=r1["round_id"],new_status="RUNNING",reason="T19 lifecycle acceptance")
    print("ROUND 1 TRANSITION"); print(f"status_after_start: {running['status']}")
    completed=store.transition_round(cid,round_id=r1["round_id"],new_status="COMPLETED",reason="T19 lifecycle acceptance completed")
    print(f"status_after_complete: {completed['status']}\n")
    r2=store.add_round(cid,plan=r2p)
    final=store.load(cid)
    print("ROUND 2"); print(f"round_id: {r2['round_id']}"); print(f"status: {r2['status']}"); print(f"dataset_version: {r2['plan']['dataset_version']}\n")
    print("FINAL CAMPAIGN STATE"); print(f"status: {final['status']}"); print(f"current_round_no: {final['current_round_no']}"); print(f"round_count: {len(final['rounds'])}"); print(f"event_count: {len(final['events'])}\n")
    print("OUTPUT"); print(f"campaign_json: {store.campaign_path(cid)}\n")
    if len(final["rounds"])!=2 or final["rounds"][0]["status"]!="COMPLETED" or final["rounds"][1]["status"]!="PLANNED" or final["current_round_no"]!=2: raise SystemExit("ERROR: final campaign lifecycle state invalid")
    print("V0.2-T19 EXPERIMENT CAMPAIGN + ROUND PASS")

if __name__=="__main__": main()
