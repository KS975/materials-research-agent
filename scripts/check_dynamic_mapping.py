from __future__ import annotations

import os

from app.config import get_settings
from data.dynamic_fields import DynamicFieldResolver
from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories import ColumnDefinitionRepository, MaterialRepository


def main():
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    if not company_id:
        raise SystemExit("请设置 DEV_COMPANY_ID")

    db = BusinessMySQLClient(get_settings())
    resolver = DynamicFieldResolver(
        MaterialRepository(db),
        ColumnDefinitionRepository(db),
    )
    print(
        resolver.resolve_formula(
            {"R3-401": "81.1064", "R3-402": "16.0889"},
            company_id,
        )
    )
    print(
        resolver.resolve_dynamic(
            {"P14598": "41.2052"},
            company_id,
            "performance",
        )
    )
    print(
        resolver.resolve_dynamic(
            {"SP14741": "11"},
            company_id,
            "service_performance",
        )
    )


if __name__ == "__main__":
    main()
