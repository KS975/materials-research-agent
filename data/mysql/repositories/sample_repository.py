from __future__ import annotations

from typing import Any

from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories.common import bounded_limit, project_scope_clause
from schemas.user_context import UserContext


_SAMPLE_COLUMNS = """
id, name, `describe`, company, organization, org_level,
creator_id, creator_name, create_time, update_time, `delete`,
sample_type_id, craft_id, need_synthesis, synthesis_state,
recipes, recipe_batches, craft_detail, craft_param,
performances, service_performances, conditions,
project_id, datacenter, sample_type, recipe_type, state
"""


class SampleRepository:
    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def get_by_id(self, sample_id: int, ctx: UserContext) -> dict[str, Any] | None:
        scope_sql, scope_params = project_scope_clause(ctx.project_ids)
        sql = f"""
            SELECT {_SAMPLE_COLUMNS}
            FROM eln_sample
            WHERE id = %s
              AND company = %s
              AND (`delete` IS NULL OR `delete` IN (0, 2))
              AND {scope_sql}
            LIMIT 1
        """
        return self.db.query_one(sql, [int(sample_id), ctx.company_id, *scope_params])

    def find_exact_name(self, name: str, ctx: UserContext, limit: int = 20) -> list[dict[str, Any]]:
        scope_sql, scope_params = project_scope_clause(ctx.project_ids)
        lim = bounded_limit(limit)
        sql = f"""
            SELECT {_SAMPLE_COLUMNS}
            FROM eln_sample
            WHERE name = %s
              AND company = %s
              AND (`delete` IS NULL OR `delete` IN (0, 2))
              AND {scope_sql}
            ORDER BY id DESC
            LIMIT {lim}
        """
        return self.db.query_all(sql, [name, ctx.company_id, *scope_params])

    def find(self, keyword: str, ctx: UserContext, limit: int = 20) -> list[dict[str, Any]]:
        scope_sql, scope_params = project_scope_clause(ctx.project_ids)
        lim = bounded_limit(limit)
        sql = f"""
            SELECT id, name, project_id, company, sample_type, create_time, update_time
            FROM eln_sample
            WHERE company = %s
              AND (`delete` IS NULL OR `delete` IN (0, 2))
              AND {scope_sql}
              AND name LIKE %s
            ORDER BY id DESC
            LIMIT {lim}
        """
        return self.db.query_all(
            sql,
            [ctx.company_id, *scope_params, f"%{keyword}%"],
        )