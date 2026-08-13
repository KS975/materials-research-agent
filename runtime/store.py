from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from app.config import Settings


class RuntimeStore(Protocol):
    def start_run(self, user_id: str, company_id: str, message: str) -> str:
        ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        intent: str | None,
        tool_name: str | None,
        result: Any,
        answer: str | None,
        error: str | None = None,
    ) -> None:
        ...


class NullRuntimeStore:
    def start_run(self, user_id: str, company_id: str, message: str) -> str:
        return str(uuid.uuid4())

    def finish_run(
        self,
        run_id: str,
        status: str,
        intent: str | None,
        tool_name: str | None,
        result: Any,
        answer: str | None,
        error: str | None = None,
    ) -> None:
        return None


class MySQLRuntimeStore:
    """Writes only to a separate Agent Runtime database."""

    def __init__(self, settings: Settings):
        if not settings.runtime_enabled:
            raise RuntimeError("RUNTIME_ENABLED=false")
        if settings.runtime_db_name.lower() == settings.business_db_name.lower():
            raise RuntimeError("Runtime DB 不能与业务数据库相同")
        self.settings = settings

    def _connect(self):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("缺少 PyMySQL") from exc
        return pymysql.connect(
            host=self.settings.runtime_db_host,
            port=self.settings.runtime_db_port,
            user=self.settings.runtime_db_user,
            password=self.settings.runtime_db_password.get_secret_value(),
            database=self.settings.runtime_db_name,
            charset=self.settings.runtime_db_charset,
            autocommit=True,
        )

    def start_run(self, user_id: str, company_id: str, message: str) -> str:
        run_id = str(uuid.uuid4())
        sql = """
            INSERT INTO agent_run
                (run_id, user_id, company_id, user_message, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        user_id,
                        company_id,
                        message,
                        "RUNNING",
                        datetime.now(timezone.utc).replace(tzinfo=None),
                    ),
                )
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        intent: str | None,
        tool_name: str | None,
        result: Any,
        answer: str | None,
        error: str | None = None,
    ) -> None:
        sql = """
            UPDATE agent_run
            SET status=%s,
                intent=%s,
                tool_name=%s,
                tool_result_json=%s,
                answer=%s,
                error_message=%s,
                finished_at=%s
            WHERE run_id=%s
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        status,
                        intent,
                        tool_name,
                        json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
                        answer,
                        error,
                        datetime.now(timezone.utc).replace(tzinfo=None),
                        run_id,
                    ),
                )


def create_runtime_store(settings: Settings) -> RuntimeStore:
    if settings.runtime_enabled:
        return MySQLRuntimeStore(settings)
    return NullRuntimeStore()
