from __future__ import annotations

import json
import re
from typing import Any

from data.mysql.client import ReadOnlyViolation, assert_read_only_sql


DATABASE_EXPLORER_SCHEMA_VERSION = "DBE-0.1"

ALLOWED_VIRTUAL_SOURCES = {
    "authorized_samples",
    "authorized_projects",
    "authorized_materials",
    "authorized_data_columns",
}

_SAFE_FUNCTIONS = {
    "abs",
    "avg",
    "cast",
    "ceil",
    "ceiling",
    "char_length",
    "coalesce",
    "concat",
    "concat_ws",
    "count",
    "date",
    "date_format",
    "datediff",
    "day",
    "floor",
    "group_concat",
    "if",
    "ifnull",
    "json_contains_path",
    "json_extract",
    "json_keys",
    "json_length",
    "json_type",
    "json_unquote",
    "json_valid",
    "left",
    "length",
    "lower",
    "ltrim",
    "max",
    "min",
    "month",
    "nullif",
    "replace",
    "right",
    "round",
    "rtrim",
    "stddev",
    "stddev_pop",
    "stddev_samp",
    "substring",
    "sum",
    "trim",
    "upper",
    "variance",
    "year",
}

_NON_FUNCTION_TOKENS = {
    "as",
    "case",
    "decimal",
    "distinct",
    "exists",
    "in",
    "integer",
    "over",
    "signed",
    "unsigned",
    "varchar",
    "when",
}

_SOURCE_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+"
    r"(?P<name>`?[A-Za-z_][A-Za-z0-9_$]*`?"
    r"(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_$]*`?)?)",
    re.IGNORECASE,
)


class DatabaseExplorerSQLValidationError(ReadOnlyViolation):
    pass


def parse_model_json_object(raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.I | re.S)
    if fence:
        text = fence.group(1)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("DeepSeek 数据库探索结果不是合法 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("DeepSeek 数据库探索结果必须是 JSON 对象")
    return result


def normalize_sql_fingerprint(sql: Any) -> str:
    return " ".join(str(sql or "").strip().rstrip(";").split()).casefold()


def _sql_without_string_literals(sql: str) -> str:
    """Blank quoted values while retaining identifiers and SQL structure."""
    result: list[str] = []
    quote = ""
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            if char == "\\" and index + 1 < len(sql):
                result.extend((" ", " "))
                index += 2
                continue
            if char == quote:
                quote = ""
            result.append(" ")
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            result.append(" ")
        else:
            result.append(char)
        index += 1
    if quote:
        raise DatabaseExplorerSQLValidationError("SQL 字符串引号未闭合")
    return "".join(result)


def _reject_comma_joins(structural_sql: str) -> None:
    for match in re.finditer(r"\bFROM\b", structural_sql, re.I):
        tail = structural_sql[match.end():]
        stop = re.search(
            r"\b(?:WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT)\b",
            tail,
            re.I,
        )
        from_part = tail[: stop.start()] if stop else tail
        if "," in from_part:
            raise DatabaseExplorerSQLValidationError(
                "V0.1 禁止逗号连接表，请使用显式 JOIN"
            )


def validate_explorer_sql(sql: Any, *, max_chars: int = 12000) -> str:
    """Validate one model-generated SELECT over backend-owned virtual sources."""
    text = str(sql or "").strip()
    if len(text) > max_chars:
        raise DatabaseExplorerSQLValidationError(
            f"SQL 长度超过安全上限 {max_chars} 字符"
        )
    if not text:
        raise DatabaseExplorerSQLValidationError("SQL 不能为空")
    if re.search(r"(--|/\*|\*/|#)", text):
        raise DatabaseExplorerSQLValidationError("模型 SQL 禁止包含注释")
    if "%s" in text.casefold():
        raise DatabaseExplorerSQLValidationError("模型 SQL 禁止自行声明参数占位符")

    try:
        assert_read_only_sql(text)
    except ReadOnlyViolation as exc:
        raise DatabaseExplorerSQLValidationError(str(exc)) from exc

    normalized = " ".join(text.rstrip(";").split())
    if not re.match(r"^SELECT\b", normalized, re.I):
        raise DatabaseExplorerSQLValidationError(
            "数据库探索只接受一条 SELECT；WITH/SHOW/DESCRIBE 由后端负责"
        )

    structural = _sql_without_string_literals(normalized)
    if re.search(r"\b(?:WITH|UNION|INTERSECT|EXCEPT)\b", structural, re.I):
        raise DatabaseExplorerSQLValidationError(
            "V0.1 禁止模型使用 WITH、UNION、INTERSECT 或 EXCEPT"
        )
    if re.search(r"\b(?:FROM|JOIN)\s*\(", structural, re.I):
        raise DatabaseExplorerSQLValidationError("V0.1 禁止子查询或派生表")
    if re.search(
        r"\b(?:INFORMATION_SCHEMA|PERFORMANCE_SCHEMA|MYSQL|SYS)\b",
        structural,
        re.I,
    ):
        raise DatabaseExplorerSQLValidationError("禁止访问 MySQL 系统库")
    if re.search(
        r"\b(?:INTO|OUTFILE|DUMPFILE|PROCEDURE|HANDLER)\b|@@|@\w",
        structural,
        re.I,
    ):
        raise DatabaseExplorerSQLValidationError("SQL 包含禁止的导出、会话或过程语法")
    if re.search(
        r"\b(?:CURRENT_USER|SESSION_USER|SYSTEM_USER|CURRENT_ROLE|"
        r"DATABASE|VERSION|CONNECTION_ID|LAST_INSERT_ID|ROW_COUNT)\b",
        structural,
        re.I,
    ):
        raise DatabaseExplorerSQLValidationError("SQL 禁止读取数据库身份或会话信息")

    sources = []
    for match in _SOURCE_PATTERN.finditer(structural):
        name = match.group("name").replace("`", "").replace(" ", "").casefold()
        sources.append(name)
    if not sources:
        raise DatabaseExplorerSQLValidationError("SQL 必须查询至少一个授权虚拟数据源")
    invalid_sources = sorted({name for name in sources if name not in ALLOWED_VIRTUAL_SOURCES})
    if invalid_sources:
        raise DatabaseExplorerSQLValidationError(
            "SQL 只能查询授权虚拟数据源，禁止的数据源："
            + "、".join(invalid_sources)
        )
    _reject_comma_joins(structural)

    for function_name in re.findall(
        r"(?<![A-Za-z0-9_`.])([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        structural,
    ):
        key = function_name.casefold()
        if key not in _SAFE_FUNCTIONS and key not in _NON_FUNCTION_TOKENS:
            raise DatabaseExplorerSQLValidationError(
                f"SQL 函数不在 V0.1 安全白名单中：{function_name}"
            )
    return normalized


def sanitize_database_error(exc: BaseException, *, max_chars: int = 1200) -> dict[str, Any]:
    args = list(getattr(exc, "args", ()) or ())
    error_code = args[0] if args and isinstance(args[0], int) else None
    if len(args) >= 2:
        message = str(args[1])
    else:
        message = str(exc)
    message = re.sub(r"(?i)(password|passwd|pwd)\s*[=:]\s*[^\s,;]+", r"\1=<redacted>", message)
    message = re.sub(r"(?i)(mysql(?:\+\w+)?://)[^\s]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)for user\s+'[^']+'@'[^']+'", "for user <redacted>", message)
    message = re.sub(r"[A-Za-z]:\\[^\r\n]+", "<local-path>", message)
    message = message.strip()[:max_chars]
    return {
        "error_type": type(exc).__name__,
        "mysql_error_code": error_code,
        "message": message or "数据库执行失败",
    }
