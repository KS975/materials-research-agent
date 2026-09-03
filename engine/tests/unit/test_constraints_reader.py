from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from engine.ingestion.constraints import read_constraints_xlsx


def _write_constraints_xlsx(path: Path) -> None:
    strings = [
        "参数名称", "类型", "最小值", "最大值",
        "feature_001", "特征", "target_001", "目标",
    ]
    shared = (
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="8" uniqueCount="8">'
        + "".join(f"<si><t>{item}</t></si>" for item in strings)
        + "</sst>"
    )
    rows = [
        [("A1", "s", 0), ("B1", "s", 1), ("C1", "s", 2), ("D1", "s", 3)],
        [("A2", "s", 4), ("B2", "s", 5), ("C2", None, "0.1"), ("D2", None, "0.9")],
        [("A3", "s", 6), ("B3", "s", 7), ("C3", None, "10"), ("D3", None, "20")],
    ]
    row_xml = "".join(
        f'<row r="{number}">'
        + "".join(
            (
                f'<c r="{ref}" t="{type_}"><v>{value}</v></c>'
                if type_ is not None
                else f'<c r="{ref}"><v>{value}</v></c>'
            )
            for ref, type_, value in row
        )
        + "</row>"
        for number, row in enumerate(rows, start=1)
    )
    worksheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{row_xml}</sheetData></worksheet>'
    )
    workbook = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Constraints" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)


class ConstraintsReaderTests(unittest.TestCase):
    def test_chinese_headers_are_parsed_as_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "constraints.xlsx"
            _write_constraints_xlsx(source)
            result = read_constraints_xlsx(source)
            self.assertEqual(
                result["source_headers"],
                {
                    "parameter_name": "参数名称",
                    "role": "类型",
                    "lower_bound": "最小值",
                    "upper_bound": "最大值",
                },
            )
            self.assertEqual(result["variables"][0]["name"], "feature_001")
            self.assertEqual(result["target_bounds"][0]["name"], "target_001")
            # The complete result must round-trip through JSON without replacement chars.
            json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
