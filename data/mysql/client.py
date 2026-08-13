from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from app.config import Settings


class ReadOnlyViolation(RuntimeError):
    pass


_DANGEROUS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|ALTER|DROP|CREATE|TRUNCATE|GRANT|REVOKE|CALL|LOAD|LOCK|UNLOCK)\b",
    re.IGNORECASE,
)


def assert_read_only_sql(sql: str) -> None:
    normalized = " ".join(sql.strip().split())
    if not normalized:
        raise ReadOnlyViolation("SQL 不能为空")

    # 允许一个末尾分号，但禁止多条 SQL。
    without_trailing = normalized[:-1] if normalized.endswith(";") else normalized

    if ";" in without_trailing:
        raise ReadOnlyViolation("业务数据库禁止执行多语句 SQL")

    first = without_trailing.split(" ", 1)[0].upper()

    allowed = {
        "SELECT",
        "SHOW",
        "DESCRIBE",
        "DESC",
        "EXPLAIN",
        "WITH",
    }

    if first not in allowed:
        raise ReadOnlyViolation(
            f"业务数据库只允许只读 SQL，当前开头为：{first}"
        )

    # SELECT 本身可能存在文件写出/锁表副作用，也禁止。
    dangerous_select = re.compile(
        r"\bINTO\s+(OUTFILE|DUMPFILE)\b"
        r"|\bFOR\s+UPDATE\b"
        r"|\bLOCK\s+IN\s+SHARE\s+MODE\b",
        re.IGNORECASE,
    )

    if dangerous_select.search(without_trailing):
        raise ReadOnlyViolation(
            "业务数据库禁止 SELECT INTO OUTFILE/DUMPFILE、FOR UPDATE 或 LOCK IN SHARE MODE"
        )

    # WITH 在 MySQL 中可能是：
    # WITH ... UPDATE ...
    # WITH ... DELETE ...
    # 因此只对 WITH 做额外写操作检查。
    #
    # 先去掉反引号字段名，避免真实字段 `delete`
    # 被误识别成 DELETE 语句。
    if first == "WITH":
        cleaned = re.sub(r"`[^`]*`", "", without_trailing)

        if _DANGEROUS.search(cleaned):
            raise ReadOnlyViolation(
                "业务数据库 WITH SQL 包含禁止的写入/DDL 关键字"
            )

class BusinessMySQLClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        self.settings.require_business_db()
        try:
            import pymysql
            import pymysql.cursors
        except ImportError as exc:
            raise RuntimeError("缺少 PyMySQL，请先执行 pip install -r requirements.txt") from exc

        conn = pymysql.connect(
            host=self.settings.business_db_host,
            port=self.settings.business_db_port,
            user=self.settings.business_db_user,
            password=self.settings.business_db_password.get_secret_value(),
            database=self.settings.business_db_name,
            charset=self.settings.business_db_charset,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=self.settings.business_db_connect_timeout,
            read_timeout=self.settings.business_db_read_timeout,
            write_timeout=self.settings.business_db_read_timeout,
            autocommit=True,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
            yield conn
        finally:
            conn.close()

    def query_all(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        assert_read_only_sql(sql)
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params or ()))
                rows = cursor.fetchall()
        return list(rows)

    def query_one(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    def ping(self) -> dict[str, Any]:
        # Avoid using CURRENT_USER as an unquoted alias: it is a MySQL keyword/function name.
        base = self.query_one(
            "SELECT DATABASE() AS database_name, CURRENT_USER() AS authenticated_user"
        ) or {}

        # MySQL variants/versions may expose either transaction_read_only or tx_read_only.
        try:
            read_only = self.query_one(
                "SELECT @@session.transaction_read_only AS session_read_only"
            ) or {}
        except Exception:
            read_only = self.query_one(
                "SELECT @@session.tx_read_only AS session_read_only"
            ) or {}

        return {**base, **read_only}