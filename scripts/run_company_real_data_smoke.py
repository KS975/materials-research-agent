from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from company_data import CompanyDataRepository, import_company_archive
from runtime.company_data_ui import build_company_data_overview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", required=True)
    parser.add_argument(
        "--runtime-root",
        default=".runtime/company_data_smoke_root",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root)
    if args.reset and root.exists():
        shutil.rmtree(root)

    first = import_company_archive(
        source_zip=args.source_zip,
        runtime_root=root,
    )
    second = import_company_archive(
        source_zip=args.source_zip,
        runtime_root=root,
    )

    manifest = first["manifest"]
    summary = manifest["summary"]
    repo = CompanyDataRepository(root)
    product = repo.product(product_name="PC/ABS FR303")
    exported = repo.export_modeling_dataset(
        product_name="PC/ABS FR303",
        target_metric="悬臂梁冲击强度",
    )
    view = build_company_data_overview(
        root,
        product_name="PC/ABS FR303",
    )

    print("V0.2 COMPANY REAL DATA SMOKE")
    print(f"dataset_id: {manifest['dataset_id']}")
    print(f"samples: {summary['samples']}")
    print(f"products: {summary['products']}")
    print(f"materials: {summary['materials']}")
    print(f"performance_metrics: {summary['performance_metrics']}")
    print(f"explicit_test_condition_rows: {summary['explicit_test_condition_rows']}")
    print(f"material_process_parameter_rows: {summary['material_process_parameter_rows']}")
    print(f"named_subset_directories: {summary['named_subset_directories']}")
    print()
    print("PRODUCT")
    print(f"product: {product['product_type']}")
    print(f"local_project_id: {product['local_project_id']}")
    print(f"sample_count: {product['sample_count']}")
    print()
    print("TARGET EXPORT")
    print(f"target: {exported['target_metric']}")
    print(f"target_numeric_count: {exported['target_numeric_count']}")
    print(f"active_formula_features: {exported['active_formula_features']}")
    print("official_model_allowed_from_import_alone: false")
    print()
    print("IDEMPOTENCY")
    print(f"second_import_idempotent: {str(second['idempotent_replay']).lower()}")
    print()
    print("UI")
    print(f"kind: {view['kind']}")
    print(f"status: {view['status']}")

    expected = {
        "samples": 496,
        "products": 101,
        "materials": 473,
        "performance_metrics": 36,
        "explicit_test_condition_rows": 0,
        "material_process_parameter_rows": 0,
        "named_subset_directories": 6,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SystemExit(
                f"ERROR: summary[{key}]={summary.get(key)!r}, expected={value!r}"
            )
    if product["sample_count"] != 83:
        raise SystemExit("ERROR: PC/ABS FR303 sample_count expected 83")
    if exported["target_numeric_count"] != 82:
        raise SystemExit(
            "ERROR: PC/ABS FR303 悬臂梁冲击强度 numeric count expected 82"
        )
    if not second["idempotent_replay"]:
        raise SystemExit("ERROR: second import should be idempotent")

    print()
    print("V0.2 COMPANY REAL DATA INTEGRATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
