from __future__ import annotations

import argparse

from company_data import CompanyDataRepository


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将单位真实数据中的某产品/性能导出为 V0.1.3 Reality/Gate 输入。"
    )
    parser.add_argument("--product", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--runtime-root", default=".runtime")
    args = parser.parse_args()

    repo = CompanyDataRepository(args.runtime_root)
    result = repo.export_modeling_dataset(
        product_name=args.product,
        target_metric=args.target,
    )

    print("V0.2 COMPANY REAL DATA -> MODELING EXPORT")
    print(f"dataset_id: {result['dataset_id']}")
    print(f"local_project_id: {result['product']['local_project_id']}")
    print(f"product: {result['product']['product_type']}")
    print(f"target_metric: {result['target_metric']}")
    print(f"rows: {result['rows']}")
    print(f"target_numeric_count: {result['target_numeric_count']}")
    print(f"active_formula_features: {result['active_formula_features']}")
    print(f"process_parameter_rows: {result['process_parameter_rows']}")
    print(f"condition_rows: {result['condition_rows']}")
    print("official_model_allowed_from_import_alone: false")
    print()
    print("OUTPUT")
    print(f"dataset_csv: {result['dataset_csv']}")
    print(f"reality_json: {result['reality_json']}")
    print()
    print("NEXT")
    print(
        "python -m scripts.run_v013_modeling_gate "
        f"--project-id {result['product']['local_project_id']} "
        f'--target "{result["target_metric"]}"'
    )
    print()
    print("V0.2 COMPANY REAL DATA MODELING EXPORT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
