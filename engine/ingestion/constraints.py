from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from engine.exceptions import ValidationError


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PACKAGE_REL = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)

HEADER_ALIASES: dict[str, set[str]] = {
    "parameter_name": {
        "parameter", "parameter_name", "param", "name", "参数名称", "参数",
    },
    "role": {"type", "role", "类型", "参数类型"},
    "lower_bound": {"min", "minimum", "lower", "lower_bound", "最小值", "下限"},
    "upper_bound": {"max", "maximum", "upper", "upper_bound", "最大值", "上限"},
    "target_value": {"target", "goal", "target_value", "目标值"},
}

ROLE_ALIASES = {
    "feature": {"feature", "features", "特征", "变量", "参数"},
    "target": {"target", "targets", "目标", "性能", "指标"},
}


def read_constraints_xlsx(path: str | Path) -> dict[str, Any]:
    """Read a legacy constraints workbook into a stable JSON contract.

    Parsing XLSX here keeps host agent tools JSON-only. The source workbook is
    never modified and no optional spreadsheet dependency is introduced.
    """
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"constraints workbook does not exist: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            shared = _shared_strings(archive)
            worksheet_path = _first_worksheet_path(archive)
            rows = _worksheet_rows(archive, worksheet_path, shared)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"constraints workbook cannot be read: {exc}") from exc

    if not rows:
        raise ValidationError("constraints workbook has no rows")
    header_row = rows[0]
    headers = _canonical_headers(header_row)
    required = {"parameter_name", "role", "lower_bound", "upper_bound"}
    missing = required - set(headers)
    if missing:
        raise ValidationError(
            f"constraints workbook is missing headers: {sorted(missing)}"
        )

    variables: list[dict[str, Any]] = []
    target_bounds: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    warnings: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if not any(_cell_text(row.get(column)) for column in headers.values()):
            continue
        name = _cell_text(row.get(headers["parameter_name"]))
        raw_role = _cell_text(row.get(headers["role"])).lower()
        role = _canonical_role(raw_role)
        if not name or role is None:
            warnings.append({
                "code": "CONSTRAINT_ROW_IGNORED",
                "row_number": row_number,
                "message": "row has no valid parameter name or recognizable role",
            })
            continue
        identity = (role, name)
        if identity in seen:
            raise ValidationError(
                f"duplicate constraint row for {role} {name} at row {row_number}"
            )
        seen.add(identity)
        record: dict[str, Any] = {
            "name": name,
            "lower_bound": _optional_number(
                row.get(headers["lower_bound"]), f"{name} lower bound"
            ),
            "upper_bound": _optional_number(
                row.get(headers["upper_bound"]), f"{name} upper bound"
            ),
        }
        if "target_value" in headers:
            record["target_value"] = _optional_number(
                row.get(headers["target_value"]), f"{name} target value"
            )
        if role == "feature":
            variables.append(record)
        else:
            target_bounds.append(record)

    return {
        "record_type": "constraints",
        "schema_version": 1,
        "source_uri": str(source.resolve()),
        "source_headers": {
            key: _cell_text(header_row.get(column))
            for key, column in headers.items()
        },
        "variables": variables,
        "target_bounds": target_bounds,
        "warnings": warnings,
    }


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(path))
    values: list[str] = []
    for item in root.findall(f"{{{NS_MAIN}}}si"):
        values.append("".join(
            node.text or ""
            for node in item.findall(f".//{{{NS_MAIN}}}t")
        ))
    return values


def _first_worksheet_path(archive: zipfile.ZipFile) -> str:
    workbook_path = "xl/workbook.xml"
    rels_path = "xl/_rels/workbook.xml.rels"
    if workbook_path not in archive.namelist() or rels_path not in archive.namelist():
        return "xl/worksheets/sheet1.xml"
    workbook = ET.fromstring(archive.read(workbook_path))
    rels = ET.fromstring(archive.read(rels_path))
    relationships = {
        item.attrib.get("Id"): item.attrib.get("Target")
        for item in rels.findall(f"{{{NS_PACKAGE_REL}}}Relationship")
    }
    first_sheet = workbook.find(f".//{{{NS_MAIN}}}sheet")
    relationship_id = (
        first_sheet.attrib.get(f"{{{NS_REL}}}id") if first_sheet is not None else None
    )
    target = relationships.get(relationship_id or "")
    if not target:
        return "xl/worksheets/sheet1.xml"
    target = target.lstrip("/")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def _worksheet_rows(
    archive: zipfile.ZipFile,
    worksheet_path: str,
    shared_strings: list[str],
) -> list[dict[int, Any]]:
    root = ET.fromstring(archive.read(worksheet_path))
    rows: list[dict[int, Any]] = []
    for row in root.findall(f".//{{{NS_MAIN}}}row"):
        parsed: dict[int, Any] = {}
        for cell in row.findall(f"{{{NS_MAIN}}}c"):
            column = _column_index(cell.attrib.get("r", ""))
            parsed[column] = _cell_value(cell, shared_strings)
        rows.append(parsed)
    return rows


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(f".//{{{NS_MAIN}}}t")
        )
    value_node = cell.find(f"{{{NS_MAIN}}}v")
    if value_node is None or value_node.text is None:
        return None
    value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError) as exc:
            raise ValidationError("workbook contains an invalid shared-string index") from exc
    if cell_type == "b":
        return value == "1"
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    if not letters:
        return 0
    result = 0
    for letter in letters.upper():
        if not "A" <= letter <= "Z":
            break
        result = result * 26 + ord(letter) - ord("A") + 1
    return max(result - 1, 0)


def _canonical_headers(row: dict[int, Any]) -> dict[str, int]:
    aliases = {
        key: {value.strip().lower() for value in values}
        for key, values in HEADER_ALIASES.items()
    }
    headers: dict[str, int] = {}
    for column, raw_value in sorted(row.items()):
        value = _cell_text(raw_value).strip().lower()
        for canonical, candidates in aliases.items():
            if value in candidates and canonical not in headers:
                headers[canonical] = column
                break
    return headers


def _canonical_role(value: str) -> str | None:
    normalized = value.strip().lower()
    for canonical, candidates in ROLE_ALIASES.items():
        if normalized in candidates:
            return canonical
    return None


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_number(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be numeric or empty") from exc
