from io import BytesIO

from docx import Document

from file_processing.parser import ChatFileParser


def test_parse_docx_text_and_table():
    doc = Document()
    doc.add_heading("实验报告", level=1)
    doc.add_paragraph("样品 A 的冲击强度为 35 kJ/m²。")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "测试温度"
    table.rows[0].cells[1].text = "25 ℃"
    buf = BytesIO()
    doc.save(buf)

    parsed = ChatFileParser().parse_bytes("report.docx", buf.getvalue())

    assert parsed.parser == "python-docx"
    assert "35 kJ/m²" in parsed.text
    assert "测试温度" in parsed.text
    assert len(parsed.chunks) >= 1
