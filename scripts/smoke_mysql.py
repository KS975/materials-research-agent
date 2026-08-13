from __future__ import annotations

import argparse
import os

from agent.tools import MaterialsTools
from app.config import get_settings
from data.dynamic_fields import DynamicFieldResolver
from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories import (
    ArchiveRepository,
    ColumnDefinitionRepository,
    ExperimentRepository,
    MaterialRepository,
    ProjectRepository,
    SampleRepository,
)
from schemas.user_context import UserContext


def build_tools():
    settings = get_settings()
    db = BusinessMySQLClient(settings)
    samples = SampleRepository(db)
    projects = ProjectRepository(db)
    materials = MaterialRepository(db)
    columns = ColumnDefinitionRepository(db)
    archives = ArchiveRepository(db)
    experiments = ExperimentRepository(db)
    resolver = DynamicFieldResolver(materials, columns)
    return db, MaterialsTools(samples, projects, archives, experiments, resolver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="trial_10")
    args = parser.parse_args()

    user_id = os.getenv("DEV_USER_ID", "").strip()
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    projects = tuple(
        int(x.strip())
        for x in os.getenv("DEV_PROJECT_IDS", "").split(",")
        if x.strip()
    )
    if not user_id or not company_id or not projects:
        raise SystemExit(
            "请设置 DEV_USER_ID / DEV_COMPANY_ID / DEV_PROJECT_IDS"
        )

    ctx = UserContext(
        user_id=user_id,
        company_id=company_id,
        project_ids=projects,
        permission_source="smoke_script",
    )
    db, tools = build_tools()

    print("DB ping:")
    print(db.ping())
    print()
    print(f"Sample context: {args.sample}")
    result = tools.get_sample_context(args.sample, ctx)
    import json
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
