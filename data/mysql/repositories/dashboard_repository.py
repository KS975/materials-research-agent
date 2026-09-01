from __future__ import annotations

from typing import Any

from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories.common import bounded_limit, project_scope_clause
from schemas.user_context import UserContext


class DashboardRepository:
    """Read-only, company-scoped data navigator queries.

    The company identity is never accepted as a method argument. It always
    comes from the already resolved ``UserContext`` so callers cannot browse a
    different company by changing a dashboard query parameter.
    """

    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def summary(self, ctx: UserContext) -> dict[str, Any]:
        sample_scope_sql, sample_scope_params = project_scope_clause(
            ctx.project_ids,
            column="project_id",
            allow_all=ctx.all_projects,
        )
        project_scope_sql, project_scope_params = project_scope_clause(
            ctx.project_ids,
            column="id",
            allow_all=ctx.all_projects,
        )
        projects = self.db.query_one(
            f"""
            SELECT COUNT(*) AS project_count, MAX(update_time) AS latest_project_update
            FROM mat_project
            WHERE company = %s
              AND (`delete` IS NULL OR `delete` = 0)
              AND {project_scope_sql}
            """,
            [ctx.company_id, *project_scope_params],
        ) or {}
        samples = self.db.query_one(
            f"""
            SELECT COUNT(*) AS sample_count, MAX(update_time) AS latest_sample_update
            FROM eln_sample
            WHERE company = %s
              AND (`delete` IS NULL OR `delete` IN (0, 2))
              AND {sample_scope_sql}
            """,
            [ctx.company_id, *sample_scope_params],
        ) or {}
        return {
            "project_count": int(projects.get("project_count") or 0),
            "sample_count": int(samples.get("sample_count") or 0),
            "latest_project_update": projects.get("latest_project_update"),
            "latest_sample_update": samples.get("latest_sample_update"),
        }

    def list_projects(
        self,
        ctx: UserContext,
        *,
        query: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        scope_sql, scope_params = project_scope_clause(
            ctx.project_ids,
            column="p.id",
            allow_all=ctx.all_projects,
        )
        lim = bounded_limit(limit, default=50, maximum=100)
        off = max(0, min(int(offset), 100000))
        keyword = str(query or "").strip()
        search_sql = ""
        search_params: list[Any] = []
        if keyword:
            search_sql = "AND (p.name LIKE %s OR CAST(p.id AS CHAR) LIKE %s)"
            search_params = [f"%{keyword}%", f"%{keyword}%"]

        total_row = self.db.query_one(
            f"""
            SELECT COUNT(*) AS count
            FROM mat_project p
            WHERE p.company = %s
              AND (p.`delete` IS NULL OR p.`delete` = 0)
              AND {scope_sql}
              {search_sql}
            """,
            [ctx.company_id, *scope_params, *search_params],
        ) or {}
        rows = self.db.query_all(
            f"""
            SELECT
                p.id, p.name, p.`describe`, p.state,
                p.create_time, p.update_time,
                COUNT(s.id) AS sample_count,
                MAX(s.update_time) AS latest_sample_update
            FROM mat_project p
            LEFT JOIN eln_sample s
              ON s.project_id = p.id
             AND s.company = %s
             AND (s.`delete` IS NULL OR s.`delete` IN (0, 2))
            WHERE p.company = %s
              AND (p.`delete` IS NULL OR p.`delete` = 0)
              AND {scope_sql}
              {search_sql}
            GROUP BY p.id, p.name, p.`describe`, p.state, p.create_time, p.update_time
            ORDER BY COALESCE(MAX(s.update_time), p.update_time) DESC, p.id DESC
            LIMIT {lim} OFFSET {off}
            """,
            [ctx.company_id, ctx.company_id, *scope_params, *search_params],
        )
        return {
            "total": int(total_row.get("count") or 0),
            "limit": lim,
            "offset": off,
            "projects": rows,
        }

    def list_samples(
        self,
        ctx: UserContext,
        *,
        query: str = "",
        project_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        scope_sql, scope_params = project_scope_clause(
            ctx.project_ids,
            column="s.project_id",
            allow_all=ctx.all_projects,
        )
        lim = bounded_limit(limit, default=20, maximum=50)
        off = max(0, min(int(offset), 100000))
        keyword = str(query or "").strip()
        filters = []
        filter_params: list[Any] = []
        if keyword:
            filters.append("(s.name LIKE %s OR CAST(s.id AS CHAR) LIKE %s)")
            filter_params.extend([f"%{keyword}%", f"%{keyword}%"])
        if project_id is not None:
            filters.append("s.project_id = %s")
            filter_params.append(int(project_id))
        filter_sql = "" if not filters else "AND " + " AND ".join(filters)

        total_row = self.db.query_one(
            f"""
            SELECT COUNT(*) AS count
            FROM eln_sample s
            WHERE s.company = %s
              AND (s.`delete` IS NULL OR s.`delete` IN (0, 2))
              AND {scope_sql}
              {filter_sql}
            """,
            [ctx.company_id, *scope_params, *filter_params],
        ) or {}
        rows = self.db.query_all(
            f"""
            SELECT
                s.id, s.name, s.project_id, s.sample_type,
                s.create_time, s.update_time,
                s.recipes, s.craft_param, s.performances,
                s.service_performances, s.conditions,
                p.name AS project_name
            FROM eln_sample s
            LEFT JOIN mat_project p
              ON p.id = s.project_id
             AND p.company = %s
             AND (p.`delete` IS NULL OR p.`delete` = 0)
            WHERE s.company = %s
              AND (s.`delete` IS NULL OR s.`delete` IN (0, 2))
              AND {scope_sql}
              {filter_sql}
            ORDER BY s.id DESC
            LIMIT {lim} OFFSET {off}
            """,
            [ctx.company_id, ctx.company_id, *scope_params, *filter_params],
        )
        return {
            "total": int(total_row.get("count") or 0),
            "limit": lim,
            "offset": off,
            "samples": rows,
        }
