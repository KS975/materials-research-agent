from __future__ import annotations

import json
from pathlib import Path


def main():
    """
    V0.1.3 T09 fixture

    Purpose:
    Build an artificial Dataset Reality report where:

    - sample size is sufficient
    - formula/process/target closure is complete
    - test condition coverage is 100%
    - target values are numeric
    - no unresolved dynamic fields
    - no duplicate issues

    BUT:

    - two incompatible test-condition signatures exist

    Modeling Gate should therefore reject the dataset
    specifically because test conditions are inconsistent.
    """

    output_dir = (
        Path(".runtime")
        / "v013"
        / "fixtures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "t09_condition_mismatch_reality.json"
    )

    fixture = {
        "stage": "V0.1.3-A_dataset_reality_check",

        "project_id": 9009,

        "company_id": "V013-T09-TEST",

        "target_metric": "冲击强度",

        "summary": {
            "total_samples": 30,
            "formula_present": 30,
            "process_present": 30,
            "target_present": 30,
            "conditions_present": 30,

            "core_closed_formula_process_target": 30,

            "strict_closed_with_conditions": 30,
        },

        "target": {
            "metric": "冲击强度",

            "resolved_field_ids": [
                15774
            ],

            "present_count": 30,

            "numeric_count": 30,

            "non_numeric_count": 0,

            "conflicting_sample_ids": [],

            "min": 10.0,

            "max": 60.0,

            "mean": 35.0,
        },

        "test_conditions": {
            "present_count": 30,

            "missing_count": 0,

            "unique_nonempty_signatures": 2,

            "signature_counts": {
                "ISO_179_23C_notched": 15,
                "ASTM_D256_23C_notched": 15,
            },
        },

        "field_coverage": {
            "formula": {
                "formula::ABS": 30,
                "formula::PC": 30,
            },

            "process": {
                "process::temperature": 30,
                "process::speed": 30,
            },
        },

        "unresolved_dynamic_fields": {},

        "duplicate_sample_name_groups": 0,

        "duplicate_formula_process_target_groups": 0,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fixture,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("V0.1.3 T09 FIXTURE BUILDER")
    print()
    print("Fixture assumptions:")
    print("total_samples: 30")
    print("core_closed_samples: 30")
    print("strict_closed_samples: 30")
    print("condition_coverage: 100%")
    print("condition_signatures: 2")
    print()
    print("Intentional defect:")
    print(
        "- Two incompatible test-condition "
        "signatures are mixed"
    )
    print()
    print("OUTPUT")
    print(f"fixture_json: {output_path}")
    print()
    print("V0.1.3 T09 FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()