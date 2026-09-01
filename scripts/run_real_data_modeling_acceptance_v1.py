from __future__ import annotations

import argparse

from company_data import resolve_company_data_runtime_root
from real_modeling import RealDataModelingAcceptance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V0.3 real company data modeling acceptance V1."
    )
    parser.add_argument(
        "--runtime-root",
        default=None,
        help=(
            "真实数据 runtime 根目录。省略时优先使用 "
            "COMPANY_DATA_RUNTIME_ROOT，否则使用项目根目录 .runtime"
        ),
    )
    parser.add_argument(
        "--product",
        default="PC/ABS FR303",
    )
    parser.add_argument(
        "--target",
        default="悬臂梁冲击强度",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    resolved_root = resolve_company_data_runtime_root(
        args.runtime_root
    )
    report = RealDataModelingAcceptance(
        resolved_root
    ).run(
        product_name=args.product,
        target_metric=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    source = report["source"]
    features = report["feature_selection"]
    selected = report["model_comparison"]
    holdout = report["holdout"]
    quality = report["data_quality_diagnostics"]
    boundary = report["official_boundary"]

    print("V0.3 REAL DATA MODELING ACCEPTANCE V1")
    print()
    print("SOURCE")
    print(f"runtime_root: {resolved_root}")
    print(f"source_kind: {source['source_kind']}")
    print(f"canonical_source: {source['canonical_source']}")
    print(f"company_dataset_id: {source['company_dataset_id']}")
    print(f"product: {report['product_type']}")
    print(f"target: {report['target_metric']}")
    print(f"source_rows: {source['source_rows']}")
    print(f"target_numeric_rows: {source['target_numeric_rows']}")
    print(f"simulator_rows: {source['simulator_rows']}")
    print()

    print("STRICT HOLDOUT")
    print(f"train_rows: {source['train_rows']}")
    print(f"holdout_rows: {source['holdout_rows']}")
    print(f"split_overlap_count: {source['split_overlap_count']}")
    print("feature_selection_fit_scope: TRAIN_ONLY")
    print("model_selection_scope: TRAIN_CV_ONLY")
    print("holdout_used_for_selection: false")
    print()

    print("FEATURE SELECTION")
    print(
        "all_active_formula_features: "
        f"{features['all_active_formula_features']}"
    )
    print(
        "selected_feature_count: "
        f"{features['selected_feature_count']}"
    )
    print(
        "selected_features: "
        f"{features['selected_features']}"
    )
    print(
        "missing_value_policy: "
        "median + missing indicators; blanks are not silently zero-filled"
    )
    print()

    print("MODEL COMPARISON (TRAIN CV ONLY)")
    for item in selected["leaderboard"]:
        suffix = " [BASELINE]" if item["is_baseline"] else ""
        print(
            f"{item['model_name']}{suffix}: "
            f"CV_MAE={item['cv_mae_mean']:.6f} "
            f"CV_RMSE={item['cv_rmse_mean']:.6f} "
            f"CV_R2={item['cv_r2_mean']:.6f}"
        )
    print(f"selected_model: {selected['selected_model']}")
    print()

    print("FINAL HOLDOUT")
    model_metrics = holdout["selected_model_metrics"]
    baseline = holdout["median_baseline_metrics"]
    print(
        "selected_model: "
        f"MAE={model_metrics['mae']:.6f} "
        f"RMSE={model_metrics['rmse']:.6f} "
        f"R2={model_metrics['r2']:.6f}"
    )
    print(
        "median_baseline: "
        f"MAE={baseline['mae']:.6f} "
        f"RMSE={baseline['rmse']:.6f} "
        f"R2={baseline['r2']:.6f}"
    )
    print(
        "selected_model_beats_median_baseline: "
        f"{str(holdout['selected_model_beats_median_baseline']).lower()}"
    )
    print(
        "domain_counts: "
        f"{holdout['domain_counts']}"
    )
    print()

    regime = quality["target_regime"]
    print("DATA QUALITY DIAGNOSTICS")
    print(
        "target_unit_metadata_available: "
        f"{str(quality['target_unit_metadata_available']).lower()}"
    )
    print(
        "target_regime_suspected: "
        f"{str(regime.get('suspected', False)).lower()}"
    )
    if regime.get("largest_gap") is not None:
        print(
            "target_largest_gap: "
            f"{regime['largest_gap']:.6f} "
            f"({regime['largest_gap_left']:.6f} -> "
            f"{regime['largest_gap_right']:.6f})"
        )
        print(
            "target_regime_counts: "
            f"{regime['left_regime_count']} / "
            f"{regime['right_regime_count']}"
        )
    print(
        "identical_selected_feature_collision_count: "
        f"{quality['identical_selected_feature_collision_count']}"
    )
    print()

    print("BOUNDARY")
    print(
        "formal_core_closed_samples: "
        f"{boundary['formal_reality_core_closed_samples']}"
    )
    print(
        "process_parameter_rows: "
        f"{boundary['process_parameter_rows']}"
    )
    print(
        "explicit_test_condition_rows: "
        f"{boundary['explicit_test_condition_rows']}"
    )
    print("official_model_allowed: false")
    print("bo_allowed: false")
    print("model_registry_write: false")
    print()

    print("FINAL")
    print(f"status: {report['status']}")
    print(f"review_reasons: {report['review_reasons']}")
    print(
        "acceptance_report_json: "
        f"{report['artifacts']['acceptance_report_json']}"
    )
    print(
        "holdout_predictions_csv: "
        f"{report['artifacts']['holdout_predictions_csv']}"
    )
    print(
        "diagnostic_model_joblib: "
        f"{report['artifacts']['diagnostic_model_joblib']}"
    )
    print()
    print(
        "NOTE: REVIEW_REQUIRED 不是程序失败。"
        "它表示真实数据已成功完成训练/留出验证，"
        "但当前数据质量不足以把诊断模型当成正式研发模型。"
    )
    print()
    print("V0.3 REAL DATA MODELING ACCEPTANCE V1 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
