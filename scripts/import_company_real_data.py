from __future__ import annotations

import argparse
from pathlib import Path

from company_data import import_company_archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description="导入单位提供的海科真实数据 ZIP 到本地 runtime。"
    )
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--runtime-root", default=".runtime")
    args = parser.parse_args()

    result = import_company_archive(
        source_zip=args.source_zip,
        runtime_root=args.runtime_root,
    )
    manifest = result["manifest"]
    summary = manifest["summary"]

    print("V0.2 COMPANY REAL DATA IMPORT")
    print(f"dataset_id: {manifest['dataset_id']}")
    print(f"source_sha256: {manifest['source']['sha256']}")
    print(f"idempotent_replay: {str(result['idempotent_replay']).lower()}")
    print()
    print("CANONICAL SOURCE")
    print(manifest["source"]["canonical_source"])
    print()
    print("SUMMARY")
    print(f"samples: {summary['samples']}")
    print(f"products: {summary['products']}")
    print(f"materials: {summary['materials']}")
    print(f"performance_metrics: {summary['performance_metrics']}")
    print(f"formula_present_samples: {summary['formula_present_samples']}")
    print(f"performance_present_samples: {summary['performance_present_samples']}")
    print(f"workflow_metadata_rows: {summary['workflow_metadata_rows']}")
    print(f"material_process_parameter_rows: {summary['material_process_parameter_rows']}")
    print(f"explicit_test_condition_rows: {summary['explicit_test_condition_rows']}")
    print(f"named_subset_directories: {summary['named_subset_directories']}")
    print()
    print("SAFETY")
    print("official_model_allowed_from_import_alone: false")
    print(manifest["safety"]["reason"])
    print()
    print("OUTPUT")
    print(f"manifest_json: {result['manifest_path']}")
    print()
    print("V0.2 COMPANY REAL DATA IMPORT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
