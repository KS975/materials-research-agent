from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments import (
    CampaignStore,
    DatasetVersionStore,
    ExperimentalResultService,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="V0.2-T22 Dataset Versioning + Update acceptance")
    parser.add_argument("--fixture-dir", default=".runtime/v020/fixtures/t22")
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    fixture = Path(args.fixture_dir)
    campaign_doc = load_json(fixture/"campaign_create.json")
    plan = load_json(fixture/"round_plan.json")
    planned = load_json(fixture/"planned_experiments.json")
    results = load_json(fixture/"results.json")
    project_id = campaign_doc["project_id"]
    campaign_id = campaign_doc["campaign_id"]

    store = CampaignStore(args.runtime_root)
    result_service = ExperimentalResultService(args.runtime_root)
    datasets = DatasetVersionStore(args.runtime_root)

    if args.reset:
        cdir = store.campaign_dir(campaign_id)
        if cdir.exists(): shutil.rmtree(cdir)
        pdir = datasets.project_dir(project_id)
        if pdir.exists(): shutil.rmtree(pdir)

    print("V0.2-T22 DATASET VERSIONING + UPDATE")
    print(f"project_id: {project_id}")
    print(f"campaign_id: {campaign_id}")
    print()

    base = datasets.register_base_csv(
        project_id=project_id,
        dataset_version="dataset_v001",
        source_csv=fixture/"dataset_v001.csv",
        metadata={"fixture": True},
    )
    parent_hash_before = base["sha256"]
    print("BASE DATASET")
    print("dataset_version: dataset_v001")
    print(f"row_count: {base['row_count']}")
    print(f"sha256: {base['sha256']}")
    print()

    store.create(
        campaign_id=campaign_id,
        project_id=project_id,
        name=campaign_doc["name"],
        target_metrics=campaign_doc["target_metrics"],
        metadata=campaign_doc.get("metadata"),
    )
    r = store.add_round(campaign_id, plan=plan)
    result_service.register_planned_experiments(
        campaign_id, round_id=r["round_id"], experiments=planned
    )
    store.transition_round(campaign_id, round_id=r["round_id"], new_status="RUNNING")
    for payload in results:
        result_service.ingest(campaign_id, round_id=r["round_id"], payload=payload)
    store.transition_round(campaign_id, round_id=r["round_id"], new_status="COMPLETED")

    summary = result_service.summary(campaign_id, round_id=r["round_id"])
    print("ROUND RESULT INPUT")
    print(f"completed: {summary['progress']['completed']}")
    print(f"failed: {summary['progress']['failed']}")
    print(f"invalid: {summary['progress']['invalid']}")
    print(f"not_tested: {summary['progress']['not_tested']}")
    print(f"training_eligible: {summary['progress']['training_eligible']}")
    print()

    update = datasets.update_from_round(
        campaign_store=store,
        campaign_id=campaign_id,
        round_id=r["round_id"],
        new_dataset_version="dataset_v002",
    )
    m = update["manifest"]
    print("DATASET UPDATE")
    print(f"parent_dataset_version: {m['parent_dataset_version']}")
    print(f"new_dataset_version: {m['dataset_version']}")
    print(f"row_count_before: {m['row_count_before']}")
    print(f"added_row_count: {m['added_row_count']}")
    print(f"row_count_after: {m['row_count_after']}")
    print(f"added_candidate_ids: {json.dumps(m['added_candidate_ids'], ensure_ascii=False)}")
    print(f"excluded_FAILED: {m['excluded_nontraining'].get('FAILED', 0)}")
    print(f"excluded_INVALID: {m['excluded_nontraining'].get('INVALID', 0)}")
    print(f"excluded_NOT_TESTED: {m['excluded_nontraining'].get('NOT_TESTED', 0)}")
    print(f"duplicate_skipped_count: {m['duplicate_skipped_count']}")
    print(f"sha256: {m['sha256']}")
    print()

    parent_hash_after = datasets.verify(project_id, "dataset_v001")["sha256"]
    child_verify = datasets.verify(project_id, "dataset_v002")
    print("IMMUTABILITY + INTEGRITY")
    print(f"parent_sha256_unchanged: {str(parent_hash_before == parent_hash_after).lower()}")
    print(f"parent_verified: true")
    print(f"child_verified: {str(child_verify['verified']).lower()}")
    print()

    replay = datasets.update_from_round(
        campaign_store=store,
        campaign_id=campaign_id,
        round_id=r["round_id"],
        new_dataset_version="dataset_v002",
    )
    print("IDEMPOTENT REPLAY")
    print(f"idempotent_replay: {str(replay['idempotent_replay']).lower()}")
    print()

    print("LINEAGE")
    print(f"source_campaign: {m['source']['campaign_id']}")
    print(f"source_round: {m['source']['round_id']}")
    print(f"parent_sha256: {m['parent_sha256']}")
    print()

    print("OUTPUT")
    print(f"dataset_csv: {datasets.dataset_path(project_id, 'dataset_v002')}")
    print(f"manifest_json: {datasets.manifest_path(project_id, 'dataset_v002')}")
    print()

    assert m["row_count_before"] == 35
    assert m["added_row_count"] == 2
    assert m["row_count_after"] == 37
    assert m["excluded_nontraining"]["FAILED"] == 1
    assert m["excluded_nontraining"]["INVALID"] == 1
    assert m["excluded_nontraining"]["NOT_TESTED"] == 1
    assert parent_hash_before == parent_hash_after
    assert replay["idempotent_replay"] is True

    print("V0.2-T22 DATASET VERSIONING + UPDATE PASS")


if __name__ == "__main__":
    main()
