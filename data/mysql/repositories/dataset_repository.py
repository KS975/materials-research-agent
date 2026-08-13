from __future__ import annotations

from typing import Any

from data.mysql.client import BusinessMySQLClient
from schemas.user_context import UserContext


class DatasetRealityRepository:
    """Read-only source rows for V0.1.3 dataset reality mapping."""

    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def list_project_samples(
        self,
        *,
        project_id: int,
        ctx: UserContext,
    ) -> list[dict[str, Any]]:
        if not ctx.can_access_project(project_id):
            raise PermissionError("当前用户无权读取该项目的建模数据")

        sql = """
            SELECT
                id,
                name,
                company,
                project_id,
                sample_type,
                recipes,
                recipe_batches,
                craft_detail,
                craft_param,
                performances,
                service_performances,
                conditions,
                create_time,
                update_time
            FROM eln_sample
            WHERE company = %s
              AND project_id = %s
              AND (`delete` IS NULL OR `delete` = 0)
            ORDER BY id ASC
        """
        return self.db.query_all(sql, [ctx.company_id, int(project_id)])
