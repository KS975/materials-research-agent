from __future__ import annotations

from typing import Any

from data.mysql.client import BusinessMySQLClient


class ArchiveRepository:
    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def get_sample_archive(self, sample_id: int, company_id: str) -> dict[str, Any] | None:
        sql = """
            SELECT
                id, data_name, company, material_system, data_type,
                project_id, data_properties, data_id,
                contributor_id, contributor_name, auditor_id, auditor_name,
                contribute_time, show_tags, create_time, update_time
            FROM archive_data
            WHERE data_type = 1
              AND data_id = %s
              AND company = %s
            ORDER BY id DESC
            LIMIT 1
        """
        return self.db.query_one(sql, [str(int(sample_id)), company_id])
