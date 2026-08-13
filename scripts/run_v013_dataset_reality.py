from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from app.config import Settings
from data.dataset import DatasetRealityService
from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories.column_repository import ColumnDefinitionRepository
from data.mysql.repositories.dataset_repository import DatasetRealityRepository
from data.mysql.repositories.material_repository import MaterialRepository
from schemas.user_context import UserContext


def _ctx_from_env(project_id: int) -> UserContext:
    user_id = os.getenv("DEV_USER_ID", "").strip()
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    raw_projects = os.getenv("DEV_PROJECT_IDS", "").strip()

    if not user_id or not company_id or not raw_projects:
        raise RuntimeError(
            "请在 .env 配置 DEV_USER_ID、DEV_COMPANY_ID、DEV_PROJECT_IDS"
        )

    try:
        project_ids = tuple(
            int(part.strip())
            for part in raw_projects.split(",")
            if part.strip()
        )
    except ValueError as exc:
        raise RuntimeError("DEV_PROJECT_IDS 必须是逗号分隔的整数") from exc

    ctx = UserContext(
        user_id=user_id,
        company_id=company_id,
        project_ids=project_ids,
        permission_source="development_header",
    )
    if not ctx.can_access_project(project_id):
        raise PermissionError(
            f"当前开发权限不包含 project_id={project_id}"
        )
    return ctx


def _safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", value).strip()
    return value[:80] or "target"


def _print_report(report: dict) -> None:
    summary = report["summary"]
    target = report["target"]
    conditions = report["test_conditions"]

    print("V0.1.3-A DATASET REALITY CHECK")
    print("project_id:", report["project_id"])
    print("target_metric:", report["target_metric"])
    print()
    print("total_samples:", summary.get("total_samples", 0))
    print("formula_present:", summary.get("formula_present", 0))
    print("process_present:", summary.get("process_present", 0))
    print("target_present:", summary.get("target_present", 0))
    print("conditions_present:", summary.get("conditions_present", 0))
    print(
        "core_closed_formula_process_target:",
        summary.get("core_closed_formula_process_target", 0),
    )
    print(
        "strict_closed_with_conditions:",
        summary.get("strict_closed_with_conditions", 0),
    )
    print()
    print("target_field_ids:", target.get("resolved_field_ids"))
    print("target_numeric_count:", target.get("numeric_count"))
    print(
        "test_condition_unique_nonempty_signatures:",
        conditions.get("unique_nonempty_signatures"),
    )
    print(
        "duplicate_sample_name_groups:",
        len(report["duplicates"]["duplicate_sample_name_groups"]),
    )
    print(
        "duplicate_formula_process_target_groups:",
        len(
            report["duplicates"][
                "duplicate_formula_process_target_groups"
            ]
        ),
    )
    print(
        "unresolved_dynamic_field_instances:",
        sum(report["unresolved_dynamic_fields"].values()),
    )

    if report["warnings"]:
        print("\nWARNINGS")
        for warning in report["warnings"]:
            print("-", warning)

    print("\nBOUNDARY")
    print(report["decision_boundary"])


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "V0.1.3-A: inspect whether one real project can form a "
            "sample-level modeling table. No model is trained."
        )
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--target", default="冲击强度")
    parser.add_argument(
        "--output-dir",
        default=".runtime/v013/reality",
    )
    args = parser.parse_args()

    settings = Settings()
    settings.require_business_db()
    ctx = _ctx_from_env(args.project_id)

    db = BusinessMySQLClient(settings)
    service = DatasetRealityService(
        samples=DatasetRealityRepository(db),
        materials=MaterialRepository(db),
        columns=ColumnDefinitionRepository(db),
    )

    result = service.run(
        project_id=args.project_id,
        target_metric=args.target,
        ctx=ctx,
    )

    _print_report(result.report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"project_{args.project_id}_{_safe_name(args.target)}"

    json_path = output_dir / f"{stem}_reality.json"
    csv_path = output_dir / f"{stem}_wide.csv"

    json_path.write_text(
        json.dumps(
            result.report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=result.wide_columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in result.wide_rows:
            writer.writerow(row)

    print("\nOUTPUT")
    print("report_json:", json_path)
    print("wide_csv:", csv_path)
    print("V0.1.3-A DATASET REALITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
