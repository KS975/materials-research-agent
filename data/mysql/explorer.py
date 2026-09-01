from __future__ import annotations

from typing import Any, Sequence

from agent.database_explorer_sql import (
    ALLOWED_VIRTUAL_SOURCES,
    DATABASE_EXPLORER_SCHEMA_VERSION,
    validate_explorer_sql,
)
from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories.common import project_scope_clause
from schemas.user_context import UserContext


class AuthorizedDatabaseExplorer:
    """Execute model SELECTs only through backend-owned permission CTEs."""

    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    @staticmethod
    def schema_catalog(ctx: UserContext) -> dict[str, Any]:
        return {
            "schema_version": DATABASE_EXPLORER_SCHEMA_VERSION,
            "virtual_sources_only": True,
            "contains_sample_values": False,
            "scope": {
                "company": "current",
                "projects": "all_authorized" if ctx.all_projects else list(ctx.project_ids),
            },
            "sources": {
                "authorized_samples": {
                    "description": "当前公司和项目权限范围内的样品主记录",
                    "columns": [
                        "id", "name", "describe", "organization", "org_level",
                        "create_time", "update_time", "sample_type_id", "craft_id",
                        "need_synthesis", "synthesis_state", "recipes",
                        "recipe_batches", "craft_detail", "craft_param",
                        "performances", "service_performances", "conditions",
                        "project_id", "datacenter", "sample_type", "recipe_type",
                        "state",
                    ],
                },
                "authorized_projects": {
                    "description": "当前公司和项目权限范围内的项目元数据",
                    "columns": [
                        "id", "name", "describe", "organization", "org_level",
                        "create_time", "update_time", "state", "target",
                        "begin_time", "end_time", "planned_end_time",
                        "importance_level", "budget", "actual_cost",
                    ],
                },
                "authorized_materials": {
                    "description": "当前公司的结构化配方字段定义",
                    "columns": [
                        "id", "name", "type", "datacenter", "unit", "properties",
                        "default_batch_id", "create_time",
                    ],
                },
                "authorized_data_columns": {
                    "description": "当前公司可用的工艺、性能和服役性能字段定义",
                    "columns": [
                        "id", "name", "describe", "database_type",
                        "belonging_column", "datacenter", "data_type", "unit",
                        "option_value", "object_name",
                    ],
                },
            },
            "dynamic_field_mapping": {
                "recipes": "JSON 对象；R3-{id} 对应 authorized_materials.id",
                "performances": "JSON 对象；P{id} 对应 authorized_data_columns.id",
                "service_performances": "JSON 对象；SP{id} 对应 authorized_data_columns.id",
                "craft_param": "JSON 对象；S{id} 对应 authorized_data_columns.id",
                "conditions": "样品测试条件 JSON；键名因样品而异",
            },
            "allowed_sources": sorted(ALLOWED_VIRTUAL_SOURCES),
            "restrictions": [
                "只允许单条 SELECT",
                "只允许从四个 authorized_* 虚拟数据源读取",
                "禁止物理表、系统表、WITH、UNION、子查询和危险函数",
                "后端强制公司/项目权限、超时和返回行数上限",
            ],
        }

    @staticmethod
    def build_authorized_query(
        model_sql: str,
        ctx: UserContext,
        *,
        max_rows: int,
        timeout_ms: int,
    ) -> tuple[str, list[Any]]:
        safe_sql = validate_explorer_sql(model_sql)
        row_cap = max(1, min(int(max_rows), 1000)) + 1
        timeout_hint = max(100, min(int(timeout_ms), 120000))

        sample_scope_sql, sample_scope_params = project_scope_clause(
            ctx.project_ids,
            column="s.project_id",
            allow_all=ctx.all_projects,
        )
        project_scope_sql, project_scope_params = project_scope_clause(
            ctx.project_ids,
            column="p.id",
            allow_all=ctx.all_projects,
        )

        sql = f"""
            WITH authorized_samples AS (
                SELECT
                    s.id, s.name, s.`describe` AS `describe`,
                    s.organization, s.org_level,
                    s.create_time, s.update_time,
                    s.sample_type_id, s.craft_id,
                    s.need_synthesis, s.synthesis_state,
                    s.recipes, s.recipe_batches,
                    s.craft_detail, s.craft_param,
                    s.performances, s.service_performances, s.conditions,
                    s.project_id, s.datacenter, s.sample_type,
                    s.recipe_type, s.state
                FROM eln_sample AS s
                WHERE s.company = %s
                  AND (s.`delete` IS NULL OR s.`delete` IN (0, 2))
                  AND {sample_scope_sql}
            ),
            authorized_projects AS (
                SELECT
                    p.id, p.name, p.`describe` AS `describe`,
                    p.organization, p.org_level,
                    p.create_time, p.update_time, p.state, p.target,
                    p.begin_time, p.end_time, p.planned_end_time,
                    p.importance_level, p.budget, p.actual_cost
                FROM mat_project AS p
                WHERE p.company = %s
                  AND (p.`delete` IS NULL OR p.`delete` = 0)
                  AND {project_scope_sql}
            ),
            authorized_materials AS (
                SELECT
                    m.id, m.name, m.type, m.datacenter, m.unit,
                    m.properties, m.default_batch_id, m.create_time
                FROM sample_materials AS m
                WHERE m.company = %s
            ),
            authorized_data_columns AS (
                SELECT
                    d.id, d.name, d.`describe` AS `describe`,
                    d.database_type, d.belonging_column,
                    d.datacenter, d.data_type, d.unit,
                    d.option_value, d.object_name
                FROM data_column AS d
                WHERE (d.company = %s OR d.company IS NULL OR d.company = '')
                  AND (d.`delete` IS NULL OR d.`delete` = 0)
            )
            SELECT /*+ MAX_EXECUTION_TIME({timeout_hint}) */ *
            FROM (
                {safe_sql}
            ) AS explorer_result
            LIMIT {row_cap}
        """
        params: list[Any] = [
            ctx.company_id,
            *sample_scope_params,
            ctx.company_id,
            *project_scope_params,
            ctx.company_id,
            ctx.company_id,
        ]
        return sql, params

    def execute(
        self,
        model_sql: str,
        ctx: UserContext,
        *,
        max_rows: int,
        timeout_ms: int,
    ) -> list[dict[str, Any]]:
        sql, params = self.build_authorized_query(
            model_sql,
            ctx,
            max_rows=max_rows,
            timeout_ms=timeout_ms,
        )
        query_with_timeout = getattr(self.db, "query_all_explorer", None)
        if callable(query_with_timeout):
            return query_with_timeout(sql, params, timeout_ms=timeout_ms)
        return self.db.query_all(sql, params)
