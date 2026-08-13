from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_float(value):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V0.1.3 T13 Applicability Domain test samples."
    )
    parser.add_argument("--dataset-csv", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_csv)

    if not dataset_path.exists():
        raise SystemExit(f"ERROR: dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise SystemExit("ERROR: dataset CSV has no header.")

        feature_cols = [
            col
            for col in reader.fieldnames
            if col.startswith("formula::")
            or col.startswith("process::")
        ]

        rows = list(reader)

    matrix = []

    for row in rows:
        vector = [parse_float(row.get(col)) for col in feature_cols]

        if all(value is not None for value in vector):
            matrix.append([float(value) for value in vector])

    if len(matrix) < 10:
        raise SystemExit(
            "ERROR: not enough numeric rows to build T13 samples."
        )

    n = len(matrix)
    p = len(feature_cols)

    means = [
        sum(row[j] for row in matrix) / n
        for j in range(p)
    ]

    stds = []

    for j in range(p):
        variance = (
            sum(
                (row[j] - means[j]) ** 2
                for row in matrix
            )
            / n
        )
        stds.append(math.sqrt(variance) or 1.0)

    # Choose the actual training row closest to the standardized centroid.
    def centroid_distance(row):
        return math.sqrt(
            sum(
                ((row[j] - means[j]) / stds[j]) ** 2
                for j in range(p)
            )
        )

    central_row = min(matrix, key=centroid_distance)

    mins = [
        min(row[j] for row in matrix)
        for j in range(p)
    ]
    maxs = [
        max(row[j] for row in matrix)
        for j in range(p)
    ]

    # Prefer a process variable for synthetic perturbation so the fixture
    # does not intentionally break formulation closure.
    perturb_index = next(
        (
            idx
            for idx, col in enumerate(feature_cols)
            if col.startswith("process::")
            and maxs[idx] > mins[idx]
        ),
        None,
    )

    if perturb_index is None:
        perturb_index = next(
            (
                idx
                for idx in range(p)
                if maxs[idx] > mins[idx]
            ),
            None,
        )

    if perturb_index is None:
        raise SystemExit(
            "ERROR: all features are constant; cannot build T13 fixtures."
        )

    width = maxs[perturb_index] - mins[perturb_index]

    in_values = list(central_row)

    borderline_values = list(central_row)
    borderline_values[perturb_index] = (
        mins[perturb_index] + 0.04 * width
    )

    out_values = list(central_row)
    out_values[perturb_index] = (
        maxs[perturb_index] + 0.25 * width
    )

    output_dir = (
        Path(".runtime")
        / "v013"
        / "fixtures"
        / "t13"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = [
        (
            "in_domain_sample.json",
            "T13_in_domain",
            in_values,
            "Expected: IN_DOMAIN",
        ),
        (
            "borderline_sample.json",
            "T13_borderline",
            borderline_values,
            "Expected: BORDERLINE",
        ),
        (
            "out_of_domain_sample.json",
            "T13_out_of_domain",
            out_values,
            "Expected: OUT_OF_DOMAIN",
        ),
    ]

    for filename, sample_name, values, note in fixtures:
        payload = {
            "sample_name": sample_name,
            "features": {
                feature: value
                for feature, value in zip(feature_cols, values)
            },
            "fixture_note": note,
            "perturbed_feature": feature_cols[perturb_index],
        }

        with (output_dir / filename).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

    print("V0.1.3 T13 AD FIXTURE BUILDER")
    print(f"dataset_csv: {dataset_path}")
    print(f"training_rows: {n}")
    print(f"feature_count: {p}")
    print(
        f"perturbed_feature: "
        f"{feature_cols[perturb_index]}"
    )
    print()
    print("OUTPUT")
    print(
        f"in_domain: "
        f"{output_dir / 'in_domain_sample.json'}"
    )
    print(
        f"borderline: "
        f"{output_dir / 'borderline_sample.json'}"
    )
    print(
        f"out_of_domain: "
        f"{output_dir / 'out_of_domain_sample.json'}"
    )
    print()
    print("V0.1.3 T13 AD FIXTURE BUILD PASS")


if __name__ == "__main__":
    main()
