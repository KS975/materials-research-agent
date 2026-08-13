from __future__ import annotations

from typing import Any, Iterable

from data.mysql.client import BusinessMySQLClient


class ColumnDefinitionRepository:
    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def get_by_ids(self, ids: Iterable[int], company_id: str) -> dict[int, dict[str, Any]]:
        clean_ids = sorted({int(x) for x in ids})
        if not clean_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(clean_ids))
        sql = f"""
            SELECT
                id, name, `describe`, database_type, belonging_column,
                datacenter, data_type, unit, company, option_value, object_name
            FROM data_column
            WHERE id IN ({placeholders})
              AND (company = %s OR company IS NULL OR company = '')
              AND (`delete` IS NULL OR `delete` = 0)
        """
        rows = self.db.query_all(sql, [*clean_ids, company_id])
        return {int(row["id"]): row for row in rows}
