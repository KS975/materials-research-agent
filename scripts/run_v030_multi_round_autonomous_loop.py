from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from experiments import (
    AutonomousMultiRoundLoop,
    CampaignStore,
    DatasetVersionStore,
    ExperimentalResultService,
)
from scripts.build_v030_t34_fixture import (
    CAMPAIGN_ID,
    CHALLENGER_MODEL_VERSIONS,
    DATASET_VERSIONS,
    PROJECT_ID,
    TARGET,
    UNIT,
    campaign_create,
    device_profile,
    gate_pass,
    planned_experiments,
    protocol_template,
    round1_plan,
    safety_policy,
)


def _remove(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument(
        "--fixture-dir",
        default=".runtime/v030/fixtures/t34",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root)
    fixture = Path(args.fixture_dir)
    base_csv = fixture / "dataset_v001.csv"
    pool_csv = fixture / "candidate_pool.csv"
    if not base_csv.exists() or not pool_csv.exists():
        raise SystemExit(
            "ERROR: 请先运行 python -m scripts.build_v030_t34_fixture --reset"
        )

    store = CampaignStore(root)
    datasets = DatasetVersionStore(root)
    results = ExperimentalResultService(str(root))

    if args.reset:
        cleanup = [
            store.campaign_dir(CAMPAIGN_ID),
            datasets.project_dir(PROJECT_ID),
            root / "v020" / "evaluations" / CAMPAIGN_ID,
            root / "v020" / "models" / f"project_{PROJECT_ID}",
            root / "v020" / "model_promotion" / f"project_{PROJECT_ID}",
            root / "v020" / "closed_loop_bo" / CAMPAIGN_ID,
            root / "v030" / "result_capture" / CAMPAIGN_ID,
            root / "v030" / "autonomous_round" / CAMPAIGN_ID,
            root / "v030" / "autonomous_loop" / CAMPAIGN_ID,
        ]
        for target in cleanup:
            _remove(target)
        # T34 uses globally keyed protocol/telemetry/safety/scheduler ids that all
        # include this campaign id. Their specific campaign-scoped directories
        # are removed lazily below when known; leaving unrelated runtime intact.
        for parent in (
            root / "v030" / "scheduler",
            root / "v030" / "telemetry",
            root / "v030" / "safety",
        ):
            if parent.exists():
                for child in list(parent.iterdir()):
                    if CAMPAIGN_ID in child.name:
                        _remove(child)

    c = campaign_create()
    store.create(
        campaign_id=c["campaign_id"],
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    round1 = store.add_round(CAMPAIGN_ID, plan=round1_plan())
    results.register_planned_experiments(
        CAMPAIGN_ID,
        round_id=round1["round_id"],
        experiments=planned_experiments(),
    )
    datasets.register_base_csv(
        project_id=PROJECT_ID,
        dataset_version=DATASET_VERSIONS[0],
        source_csv=base_csv,
        metadata={"fixture": True, "stage": "V0.3-T34"},
    )

    loop = AutonomousMultiRoundLoop(root)

    print("V0.3-T34 MULTI-ROUND AUTONOMOUS LOOP")
    print(f"campaign_id: {CAMPAIGN_ID}")
    print(f"first_round_id: {round1['round_id']}")
    print()
    print("BOUNDARY")
    print("device_adapter: SimulatorDeviceAdapter only")
    print("real_device_connected: false")
    print("target_rounds: 3")
    print("experiments_per_round: 5")
    print("manual_result_submission_required: false")
    print("automatic_model_activation: false")
    print("final_round_creates_round4: false")
    print()

    report = loop.run(
        campaign_id=CAMPAIGN_ID,
        first_round_id=round1["round_id"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        dataset_versions=DATASET_VERSIONS,
        challenger_model_versions=CHALLENGER_MODEL_VERSIONS,
        rounds_to_run=3,
        active_incumbent_model_version="model_v001",
        batch_size=5,
        scheduler_timeout_ticks=30,
    )

    print("ROUND PROGRESSION")
    for index, round_report in enumerate(report["round_reports"], start=1):
        dataset = round_report["dataset"]
        print(
            f"R{index}: {round_report['round_id']} COMPLETED | "
            f"{dataset['parent_dataset_version']} "
            f"{dataset['row_count_before']} -> "
            f"{dataset['child_dataset_version']} "
            f"{dataset['row_count_after']} | "
            f"model={round_report['model_decision']}"
        )
    print()

    print("EXPERIMENTS")
    print(f"total_planned: {report['experiments']['total_planned']}")
    print(f"automatic_capture_count: {report['experiments']['automatic_capture_count']}")
    print(f"manual_result_submission_count: {report['experiments']['manual_result_submission_count']}")
    print(f"capture_receipts: {report['experiments']['receipt_count']}")
    print(f"duplicate_candidate_ids: {report['experiments']['duplicate_candidate_ids']}")
    print(f"duplicate_feature_points: {report['experiments']['duplicate_feature_points']}")
    print("measurement_origin: SIMULATOR_FIXTURE")
    print("is_real_measurement: false")
    print()

    print("DATASET LINEAGE")
    print(f"dataset_lineage: {report['dataset']['lineage']}")
    print(f"row_counts: {report['dataset']['row_counts']}")
    print(f"rows_added_per_round: {report['dataset']['rows_added_per_round']}")
    print(f"initial_rows: {report['dataset']['initial_rows']}")
    print(f"final_rows: {report['dataset']['final_rows']}")
    print()

    print("SAFETY + TELEMETRY")
    print(f"runtime_safety_stop_count: {report['safety']['runtime_safety_stop_count']}")
    print(f"all_telemetry_completed: {str(report['safety']['all_telemetry_completed']).lower()}")
    print(f"all_telemetry_hash_chains_valid: {str(report['safety']['all_telemetry_hash_chains_valid']).lower()}")
    print(f"automatic_resume_used: {str(report['safety']['automatic_resume_used']).lower()}")
    print()

    print("MODEL GOVERNANCE")
    print(f"decisions: {report['model_governance']['decisions']}")
    print(f"active_model_versions: {report['model_governance']['active_model_versions']}")
    print(f"automatic_activation_count: {report['model_governance']['automatic_activation_count']}")
    print()

    print("BAYESIAN OPTIMIZATION")
    print(f"bo_transition_count: {report['bayesian_optimization']['bo_transition_count']}")
    print(f"selected_out_of_domain_count: {report['bayesian_optimization']['selected_out_of_domain_count']}")
    print(f"final_round_created_next_round: {str(report['bayesian_optimization']['final_round_created_next_round']).lower()}")
    print()

    replay = loop.run(
        campaign_id=CAMPAIGN_ID,
        first_round_id=round1["round_id"],
        protocol_template=protocol_template(),
        device_profile=device_profile(),
        safety_policy=safety_policy(),
        candidate_pool_csv=pool_csv,
        target_metric=TARGET,
        target_unit=UNIT,
        gate=gate_pass(),
        dataset_versions=DATASET_VERSIONS,
        challenger_model_versions=CHALLENGER_MODEL_VERSIONS,
        rounds_to_run=3,
    )
    final_campaign = store.load(CAMPAIGN_ID)
    print("IDEMPOTENT REPLAY")
    print(f"idempotent_replay: {str(replay['idempotent_replay']).lower()}")
    print(f"round_count_after_replay: {len(final_campaign['rounds'])}")
    print(f"dataset_final_rows_after_replay: {datasets.verify(PROJECT_ID, DATASET_VERSIONS[-1])['row_count']}")
    print()

    print("OUTPUT")
    print(f"report_json: {report['report_json']}")
    print()
    print("EXECUTION BOUNDARY")
    print("T34 runs exactly three autonomous simulator rounds and then stops.")
    print("No Round 4 is created; active-job crash reconciliation remains T35.")
    print("No model promotion is auto-approved.")
    print("Synthetic simulator measurements are engineering fixtures, not real material data.")

    if report["round_count"] != 3:
        raise SystemExit("ERROR: expected exactly 3 rounds")
    if report["round_statuses"] != ["COMPLETED", "COMPLETED", "COMPLETED"]:
        raise SystemExit("ERROR: all three rounds must be COMPLETED")
    if report["experiments"]["total_planned"] != 15:
        raise SystemExit("ERROR: expected 15 experiments")
    if report["experiments"]["automatic_capture_count"] != 15:
        raise SystemExit("ERROR: expected 15 automatic captures")
    if report["experiments"]["manual_result_submission_count"] != 0:
        raise SystemExit("ERROR: manual submission count must be 0")
    if report["experiments"]["duplicate_candidate_ids"] != 0:
        raise SystemExit("ERROR: duplicate candidate ids detected")
    if report["experiments"]["duplicate_feature_points"] != 0:
        raise SystemExit("ERROR: duplicate feature points detected")
    if report["dataset"]["row_counts"] != [35, 40, 45, 50]:
        raise SystemExit("ERROR: dataset lineage row counts mismatch")
    if report["dataset"]["rows_added_per_round"] != [5, 5, 5]:
        raise SystemExit("ERROR: each round must add exactly 5 rows")
    if report["safety"]["runtime_safety_stop_count"] != 0:
        raise SystemExit("ERROR: normal fixture should have no safety stop")
    if not report["safety"]["all_telemetry_hash_chains_valid"]:
        raise SystemExit("ERROR: telemetry hash integrity failed")
    if report["model_governance"]["automatic_activation_count"] != 0:
        raise SystemExit("ERROR: challenger model was auto-activated")
    if report["bayesian_optimization"]["selected_out_of_domain_count"] != 0:
        raise SystemExit("ERROR: BO selected OOD candidate")
    if report["bayesian_optimization"]["final_round_created_next_round"]:
        raise SystemExit("ERROR: final round must not create Round 4")
    if len(final_campaign["rounds"]) != 3:
        raise SystemExit("ERROR: replay created unexpected Round 4")
    if not replay["idempotent_replay"]:
        raise SystemExit("ERROR: loop replay should be idempotent")

    print()
    print("V0.3-T34 MULTI-ROUND AUTONOMOUS LOOP PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
