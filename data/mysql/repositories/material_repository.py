from __future__ import annotations

from typing import Any, Iterable

from data.mysql.client import BusinessMySQLClient


class MaterialRepository:
    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def get_sample_materials(self, ids: Iterable[int], company_id: str) -> dict[int, dict[str, Any]]:
        clean_ids = sorted({int(x) for x in ids})
        if not clean_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(clean_ids))
        sql = f"""
            SELECT id, name, type, datacenter, unit, properties, company,
                   creator, default_batch_id, create_time
            FROM sample_materials
            WHERE id IN ({placeholders})
              AND company = %s
        """
        rows = self.db.query_all(sql, [*clean_ids, company_id])
        return {int(row["id"]): row for row in rows}
