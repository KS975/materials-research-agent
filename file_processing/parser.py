from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from file_processing.models import ParsedChunk, ParsedDocument


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class UnifiedFileParser:
    """Unified V0.1.2 PDF/DOCX parser.

    The same parser output is reused by both:
    - current Chat temporary attachments (T04)
    - long-term Knowledge Index ingestion (T05+)

    The parser itself has no persistence side effects: it writes neither
    ChatAttachmentStore nor Qdrant.
    """

    SUPPORTED_SUFFIXES = {".pdf", ".docx"}

    def __init__(self, chunk_chars: int = 1400, overlap_chars: int = 180):
        self.chunk_chars = max(400, chunk_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.chunk_chars // 2))

    def parse_bytes(self, filename: str, content: bytes, media_type: str = "") -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise UnsupportedFileTypeError("V0.1.2 当前只支持 PDF 和 DOCX")
        if not content:
            raise EmptyDocumentError("上传文件为空")

        if suffix == ".pdf":
            return self._parse_pdf(filename, content, media_type or "application/pdf")
        return self._parse_docx(
            filename,
            content,
            media_type
            or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def _parse_pdf(self, filename: str, content: bytes, media_type: str) -> ParsedDocument:
        reader = PdfReader(BytesIO(content))
        chunks: list[ParsedChunk] = []
        all_text: list[str] = []
        index = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            all_text.append(page_text)
            for piece in self._chunk_text(page_text):
                chunks.append(ParsedChunk(index=index, text=piece, page=page_number))
                index += 1

        text = "\n\n".join(all_text).strip()
        if not text:
            raise EmptyDocumentError(
                "PDF 未提取到可读文本。若这是扫描件，后续需要接 MinerU/OCR 解析器。"
            )

        return ParsedDocument(
            filename=filename,
            media_type=media_type,
            parser="pypdf",
            text=text,
            chunks=tuple(chunks),
            page_count=len(reader.pages),
        )

    def _parse_docx(self, filename: str, content: bytes, media_type: str) -> ParsedDocument:
        doc = DocxDocument(BytesIO(content))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Include table cell text so experiment tables are not silently lost.
        for table in doc.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                line = " | ".join(value for value in values if value)
                if line:
                    paragraphs.append(line)

        text = "\n\n".join(paragraphs).strip()
        if not text:
            raise EmptyDocumentError("DOCX 未提取到可读文本")

        chunks: list[ParsedChunk] = []
        buffer: list[str] = []
        buffer_len = 0
        start = 1
        index = 0

        def flush(end: int) -> None:
            nonlocal buffer, buffer_len, start, index
            if not buffer:
                return
            chunk_text = "\n\n".join(buffer).strip()
            chunks.append(
                ParsedChunk(
                    index=index,
                    text=chunk_text,
                    paragraph_start=start,
                    paragraph_end=end,
                )
            )
            index += 1
            buffer = []
            buffer_len = 0
            start = end + 1

        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            extra = len(paragraph) + (2 if buffer else 0)
            if buffer and buffer_len + extra > self.chunk_chars:
                flush(paragraph_index - 1)
            buffer.append(paragraph)
            buffer_len += extra
        flush(len(paragraphs))

        return ParsedDocument(
            filename=filename,
            media_type=media_type,
            parser="python-docx",
            text=text,
            chunks=tuple(chunks),
            page_count=None,
        )

    def _chunk_text(self, text: str) -> list[str]:
        cleaned = text.strip()
        if len(cleaned) <= self.chunk_chars:
            return [cleaned] if cleaned else []

        result: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + self.chunk_chars)
            if end < len(cleaned):
                # Prefer a sentence/line boundary near the end of the window.
                candidates = [
                    cleaned.rfind(mark, start + self.chunk_chars // 2, end)
                    for mark in ("\n", "。", "；", ";", ".")
                ]
                boundary = max(candidates)
                if boundary > start:
                    end = boundary + 1
            piece = cleaned[start:end].strip()
            if piece:
                result.append(piece)
            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.overlap_chars)
        return result


# Backward-compatible alias: V0.1.2-A code can keep importing ChatFileParser.
ChatFileParser = UnifiedFileParser
