from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from optimization.search_space import SearchSpaceError, load_search_space


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ERROR: invalid JSON: {path}\n{exc}"
        ) from exc

    if not isinstance(data, dict):
        raise SystemExit(
            f"ERROR: expected JSON object: {path}"
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V0.1.4-T14 Search Space + Constraints Engine"
        )
    )
    parser.add_argument(
        "--search-space-json",
        required=True,
    )
    parser.add_argument(
        "--candidate-json",
        action="append",
        default=[],
        help=(
            "Candidate JSON to validate. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
    )
    args = parser.parse_args()

    space_path = Path(args.search_space_json)

    print("V0.1.4-T14 SEARCH SPACE + CONSTRAINTS")
    print(f"search_space_json: {space_path}")
    print()

    try:
        space = load_search_space(load_json(space_path))
    except SearchSpaceError as exc:
        raise SystemExit(
            f"SEARCH SPACE INVALID\n- {exc}"
        ) from exc

    summary = space.summary()

    print("SEARCH SPACE")
    print(f"name: {summary['name']}")
    print(f"project_id: {summary['project_id']}")
    print(f"variable_count: {summary['variable_count']}")
    print(f"continuous_count: {summary['continuous_count']}")
    print(f"integer_count: {summary['integer_count']}")
    print(f"categorical_count: {summary['categorical_count']}")
    print(f"constraint_count: {summary['constraint_count']}")
    print(f"hard_constraint_count: {summary['hard_constraint_count']}")
    print(f"soft_constraint_count: {summary['soft_constraint_count']}")
    print()

    candidate_reports = []

    for raw_path in args.candidate_json:
        candidate_path = Path(raw_path)
        candidate = load_json(candidate_path)
        report = space.validate_candidate(candidate)
        report["candidate_json"] = str(candidate_path)
        candidate_reports.append(report)

        print("CANDIDATE")
        print(f"file: {candidate_path}")
        print(f"sample_name: {report.get('sample_name')}")
        print(f"status: {report['status']}")
        print(f"hard_valid: {str(report['hard_valid']).lower()}")
        print(f"soft_penalty: {report['soft_penalty']:.6f}")

        if report["variable_errors"]:
            print("VARIABLE ERRORS")
            for item in report["variable_errors"]:
                print(f"- {item['message']}: {item.get('variable') or item.get('variables')}")

        if report["hard_violations"]:
            print("HARD VIOLATIONS")
            for item in report["hard_violations"]:
                print(
                    f"- [{item['constraint_id']}] "
                    f"{item['message']}"
                )

        if report["soft_violations"]:
            print("SOFT VIOLATIONS")
            for item in report["soft_violations"]:
                print(
                    f"- [{item['constraint_id']}] "
                    f"{item['message']} "
                    f"(penalty={item['penalty']:.6f})"
                )
        print()

    output = {
        "stage": "V0.1.4-T14_search_space_validation",
        "search_space_json": str(space_path),
        "search_space_summary": summary,
        "candidate_reports": candidate_reports,
    }

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = (
            Path(".runtime")
            / "v014"
            / "search_space"
            / f"{space.name}_validation.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("OUTPUT")
    print(f"validation_json: {output_path}")
    print()
    print("V0.1.4-T14 SEARCH SPACE + CONSTRAINTS PASS")


if __name__ == "__main__":
    main()
