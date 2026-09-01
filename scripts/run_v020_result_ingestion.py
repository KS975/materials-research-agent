from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from experiments import (
    CampaignConflictError, CampaignStore,
    ExperimentalResultConflictError, ExperimentalResultError,
    ExperimentalResultNotFoundError, ExperimentalResultService,
    ExperimentalResultValidationError,
)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture-dir",default=".runtime/v020/fixtures/t20")
    parser.add_argument("--runtime-root",default=".runtime")
    parser.add_argument("--reset",action="store_true")
    args=parser.parse_args()
    fixture=Path(args.fixture_dir)
    cdoc=load_json(fixture/"campaign_create.json")
    plan=load_json(fixture/"round_plan.json")
    planned=load_json(fixture/"planned_experiments.json")
    store=CampaignStore(args.runtime_root)
    service=ExperimentalResultService(args.runtime_root)
    cid=cdoc["campaign_id"]

    if args.reset:
        d=store.campaign_dir(cid)
        if d.exists(): shutil.rmtree(d)

    print("V0.2-T20 EXPERIMENTAL RESULT INGESTION")
    print(f"campaign_id: {cid}")
    print(f"project_id: {cdoc['project_id']}")
    print()

    store.create(campaign_id=cid,project_id=cdoc["project_id"],name=cdoc["name"],target_metrics=cdoc["target_metrics"],metadata=cdoc.get("metadata"))
    r=store.add_round(cid,plan=plan)
    service.register_planned_experiments(cid,round_id=r["round_id"],experiments=planned)
    store.transition_round(cid,round_id=r["round_id"],new_status="RUNNING",reason="T20 acceptance start")

    print("ROUND")
    print(f"round_id: {r['round_id']}")
    print("status: RUNNING")
    print("planned_experiments: 5")
    print()

    result1=load_json(fixture/"result_01_completed.json")
    first=service.ingest(cid,round_id=r["round_id"],payload=result1)
    print("NORMAL COMPLETED RESULT")
    print("candidate_id: V020_T20_EXP_01")
    print("measurement_冲击强度: 48.7")
    print("training_eligible: true")
    print("round_status_after_first_result: "+store.load(cid)["rounds"][0]["status"])
    print()

    replay=service.ingest(cid,round_id=r["round_id"],payload=result1)
    print("IDEMPOTENT REPLAY")
    print("idempotent_replay: "+str(replay["idempotent_replay"]).lower())
    print()

    checks=[
        ("UNKNOWN CANDIDATE","invalid_unknown_candidate.json",(ExperimentalResultNotFoundError,)),
        ("NON-NUMERIC MEASUREMENT","invalid_non_numeric.json",(ExperimentalResultValidationError,)),
        ("TEST CONDITION MISMATCH","invalid_condition_mismatch.json",(ExperimentalResultConflictError,)),
        ("UNIT MISMATCH","invalid_unit_mismatch.json",(ExperimentalResultConflictError,)),
        ("CONFLICTING DUPLICATE","conflicting_duplicate_01.json",(ExperimentalResultConflictError,)),
    ]
    for label,filename,errs in checks:
        blocked=False
        try:
            service.ingest(cid,round_id=r["round_id"],payload=load_json(fixture/filename))
        except errs as exc:
            blocked=True
            print(label); print("blocked: true"); print(f"reason: {exc}"); print()
        if not blocked: raise SystemExit(f"ERROR: {label} should have been blocked")

    for filename in ("result_02_failed.json","result_03_invalid.json","result_04_not_tested.json"):
        service.ingest(cid,round_id=r["round_id"],payload=load_json(fixture/filename))

    print("NON-TRAINING TERMINAL RESULTS")
    print("FAILED: recorded, training_eligible=false")
    print("INVALID: recorded, training_eligible=false")
    print("NOT_TESTED: recorded, training_eligible=false")
    print()

    blocked=False
    try:
        store.transition_round(cid,round_id=r["round_id"],new_status="COMPLETED",reason="early close")
    except CampaignConflictError as exc:
        blocked=True
        print("ROUND EARLY-CLOSE GUARD"); print("blocked: true"); print(f"reason: {exc}"); print()
    if not blocked: raise SystemExit("ERROR: early close should be blocked")

    service.ingest(cid,round_id=r["round_id"],payload=load_json(fixture/"result_05_completed.json"))
    summary_before=service.summary(cid,round_id=r["round_id"])
    store.transition_round(cid,round_id=r["round_id"],new_status="COMPLETED",reason="all terminal")
    summary=service.summary(cid,round_id=r["round_id"])

    print("FINAL RESULT SUMMARY")
    p=summary["progress"]
    for key in ("completed","failed","invalid","not_tested","pending","training_eligible"):
        print(f"{key}: {p[key]}")
    print("can_close_round_before_final_transition: "+str(summary_before["can_close_round"]).lower())
    print("round_status: "+store.load(cid)["rounds"][0]["status"])
    print()
    print("OUTPUT")
    print(f"campaign_json: {store.campaign_path(cid)}")
    print()

    expected={"completed":2,"failed":1,"invalid":1,"not_tested":1,"pending":0,"training_eligible":2}
    for k,v in expected.items():
        if p[k]!=v: raise SystemExit(f"ERROR: {k}={p[k]} expected={v}")
    if len(summary["training_eligible_candidate_ids"])!=2:
        raise SystemExit("ERROR: training eligible count mismatch")

    print("V0.2-T20 EXPERIMENTAL RESULT INGESTION PASS")

if __name__=="__main__": main()
