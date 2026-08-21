from __future__ import annotations

from typing import Any

from data.mysql.client import BusinessMySQLClient
from schemas.user_context import UserContext


class ProjectRepository:
    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def get_by_id(self, project_id: int | None, ctx: UserContext) -> dict[str, Any] | None:
        if not ctx.can_access_project(project_id):
            return None
        sql = """
            SELECT
                id, name, `describe`, company, organization, org_level,
                creator_id, creator_name, create_time, update_time,
                state, target, begin_time, end_time, planned_end_time,
                importance_level, budget, actual_cost
            FROM mat_project
            WHERE id = %s
              AND company = %s
              AND (`delete` IS NULL OR `delete` = 0)
            LIMIT 1
        """
        return self.db.query_one(sql, [int(project_id), ctx.company_id])
