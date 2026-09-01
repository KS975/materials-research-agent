from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments import (
    CampaignStore,
    ClosedLoopBOService,
    DatasetVersionStore,
    ExperimentalResultService,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-dir",
        default=".runtime/v020/fixtures/t24",
    )
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    fixture = Path(args.fixture_dir)
    campaign_doc = load_json(fixture / "campaign_create.json")
    round1_plan = load_json(fixture / "round1_plan.json")
    planned = load_json(fixture / "round1_planned_experiments.json")
    results = load_json(fixture / "round1_results.json")
    gate = load_json(fixture / "gate_pass.json")

    campaign_id = campaign_doc["campaign_id"]
    project_id = campaign_doc["project_id"]

    campaigns = CampaignStore(args.runtime_root)
    result_service = ExperimentalResultService(args.runtime_root)
    datasets = DatasetVersionStore(args.runtime_root)
    closed_loop = ClosedLoopBOService(args.runtime_root)

    if args.reset:
        for path in [
            campaigns.campaign_dir(campaign_id),
            datasets.project_dir(project_id),
            Path(args.runtime_root) / "v020" / "closed_loop_bo" / campaign_id,
            Path(args.runtime_root) / "v020" / "models" / f"project_{project_id}",
        ]:
            if path.exists():
                shutil.rmtree(path)

    print("V0.2-T24 CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print(f"campaign_id: {campaign_id}")
    print(f"project_id: {project_id}")
    print("target_metric: 冲击强度")
    print()

    base = datasets.register_base_csv(
        project_id=project_id,
        dataset_version="dataset_v001",
        source_csv=fixture / "dataset_v001.csv",
        metadata={"fixture": True},
    )

    campaigns.create(
        campaign_id=campaign_id,
        project_id=project_id,
        name=campaign_doc["name"],
        target_metrics=campaign_doc["target_metrics"],
        metadata=campaign_doc.get("metadata"),
    )
    round1 = campaigns.add_round(
        campaign_id,
        plan=round1_plan,
    )
    result_service.register_planned_experiments(
        campaign_id,
        round_id=round1["round_id"],
        experiments=planned,
    )
    campaigns.transition_round(
        campaign_id,
        round_id=round1["round_id"],
        new_status="RUNNING",
    )
    for payload in results:
        result_service.ingest(
            campaign_id,
            round_id=round1["round_id"],
            payload=payload,
        )
    campaigns.transition_round(
        campaign_id,
        round_id=round1["round_id"],
        new_status="COMPLETED",
    )

    child = datasets.update_from_round(
        campaign_store=campaigns,
        campaign_id=campaign_id,
        round_id=round1["round_id"],
        new_dataset_version="dataset_v002",
    )["manifest"]

    print("ROUND 1 FEEDBACK")
    print(f"round_id: {round1['round_id']}")
    print("status: COMPLETED")
    print(f"dataset_before_rows: {base['row_count']}")
    print(f"dataset_after_rows: {child['row_count']}")
    print(f"feedback_rows_added: {child['added_row_count']}")
    print()

    report = closed_loop.generate_next_round(
        campaign_id=campaign_id,
        source_round_id=round1["round_id"],
        latest_dataset_version="dataset_v002",
        candidate_pool_csv=fixture / "candidate_pool.csv",
        target_metric="冲击强度",
        target_unit="kJ/m²",
        gate=gate,
        batch_size=5,
        acquisition="EI",
        direction="maximize",
        xi=0.01,
        min_batch_distance=0.20,
        soft_penalty_weight=0.10,
        allow_borderline_for_exploration=True,
        random_state=42,
    )

    best = report["best_so_far"]
    flow = report["candidate_flow"]
    print("BEST-SO-FAR")
    print(f"previous_dataset_best: {best['previous_dataset_best']:.6f}")
    print(f"current_dataset_best: {best['current_dataset_best']:.6f}")
    print(f"improvement_from_round1: {best['improvement_from_feedback_round']:.6f}")
    print()

    print("CANDIDATE FLOW")
    for key in (
        "candidate_pool_rows",
        "hard_invalid_excluded",
        "used_candidate_id_filtered",
        "already_observed_feature_filtered",
        "out_of_domain_excluded",
        "borderline_kept_for_exploration",
        "eligible_for_bo",
    ):
        print(f"{key}: {flow[key]}")
    print()

    print("ROUND 2")
    print(f"round_id: {report['next_round_id']}")
    campaign = campaigns.load(campaign_id)
    round2 = campaign["rounds"][-1]
    print(f"status: {round2['status']}")
    print(f"dataset_version: {round2['plan']['dataset_version']}")
    print(f"planned_experiments: {len(round2['experiments'])}")
    print()

    print("NEXT EXPERIMENTS")
    for i, item in enumerate(report["next_experiments"], start=1):
        print(
            f"{i}. {item['candidate_id']} | "
            f"mean={item['posterior_mean']:.6f} | "
            f"std={item['posterior_std']:.6f} | "
            f"EI={item['acquisition_value']:.6f} | "
            f"adjusted_EI={item['adjusted_acquisition']:.6f} | "
            f"AD={item['applicability_domain']['status']} | "
            f"soft_penalty={item['soft_penalty']:.6f}"
        )
    print()

    replay = closed_loop.generate_next_round(
        campaign_id=campaign_id,
        source_round_id=round1["round_id"],
        latest_dataset_version="dataset_v002",
        candidate_pool_csv=fixture / "candidate_pool.csv",
        target_metric="冲击强度",
        target_unit="kJ/m²",
        gate=gate,
        batch_size=5,
    )
    print("IDEMPOTENT REPLAY")
    print(f"idempotent_replay: {str(replay['idempotent_replay']).lower()}")
    print()

    # Acceptance guards.
    if report["best_so_far"]["improvement_from_feedback_round"] <= 0:
        raise SystemExit("ERROR: fixture expected positive best-so-far improvement")
    if flow["already_observed_feature_filtered"] < 5:
        raise SystemExit("ERROR: explicit observed duplicates were not filtered")
    if round2["status"] != "PLANNED":
        raise SystemExit("ERROR: Round 2 must remain PLANNED")
    if round2["plan"]["dataset_version"] != "dataset_v002":
        raise SystemExit("ERROR: Round 2 must reference dataset_v002")
    if len(round2["experiments"]) != 5:
        raise SystemExit("ERROR: Round 2 must contain 5 planned experiments")
    selected_ids = {x["candidate_id"] for x in report["next_experiments"]}
    round1_ids = {x["candidate_id"] for x in planned}
    if selected_ids & round1_ids:
        raise SystemExit("ERROR: previously completed candidate ID was re-proposed")

    print("OUTPUT")
    print(f"report_json: {report['report_json']}")
    print(f"campaign_json: {campaigns.campaign_path(campaign_id)}")
    print()
    print("NOTE: posterior mean 是模型估计，不是 Round 2 的真实实验结果。")
    print("V0.2-T24 CLOSED-LOOP BAYESIAN OPTIMIZATION PASS")


if __name__ == "__main__":
    main()
