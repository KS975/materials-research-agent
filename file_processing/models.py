from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    index: int
    text: str
    page: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    filename: str
    media_type: str
    parser: str
    text: str
    chunks: tuple[ParsedChunk, ...]
    page_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "media_type": self.media_type,
            "parser": self.parser,
            "text": self.text,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "page_count": self.page_count,
        }
