from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeLocatorType,
    KnowledgeSourceType,
)
from .repository import QdrantKnowledgeRepository


@dataclass(frozen=True)
class KnowledgeSourceSegment:
    """A parser-produced source segment with evidence location."""

    text: str
    locator_type: KnowledgeLocatorType = "chunk"
    page_number: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class KnowledgeIndexResult:
    document_id: str
    filename: str
    chunks_indexed: int
    document_content_hash: str


class KnowledgeChunker:
    """Small deterministic text chunker for already-extracted text.

    It prefers paragraph boundaries and only slices a paragraph when the
    paragraph itself exceeds max_chars.
    """

    def __init__(self, *, max_chars: int = 1200, overlap_chars: int = 120) -> None:
        if max_chars < 100:
            raise ValueError("max_chars must be >= 100")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _slice_long_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(len(text), start + self.max_chars)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)

            if end >= len(text):
                break

            start = end - self.overlap_chars

        return chunks

    def split(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in normalized.split("\n")
            if paragraph.strip()
        ]

        output: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if current:
                output.append("\n".join(current))
                current = []
                current_len = 0

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chars:
                flush()
                output.extend(self._slice_long_text(paragraph))
                continue

            additional = len(paragraph) + (1 if current else 0)
            if current and current_len + additional > self.max_chars:
                flush()

            current.append(paragraph)
            current_len += len(paragraph) + (1 if len(current) > 1 else 0)

        flush()
        return output


class KnowledgeIndexer:
    """Turns parsed text/segments into persistent Qdrant knowledge chunks."""

    def __init__(
        self,
        *,
        repository: QdrantKnowledgeRepository,
        chunker: KnowledgeChunker | None = None,
    ) -> None:
        self.repository = repository
        self.chunker = chunker or KnowledgeChunker()

    def index_text(
        self,
        *,
        document_id: str,
        company_id: str,
        project_id: int,
        filename: str,
        source_type: KnowledgeSourceType,
        text: str,
        source_id: str | None = None,
        mime_type: str | None = None,
        parser: str | None = None,
        uploaded_by_user_id: str | None = None,
    ) -> KnowledgeIndexResult:
        chunk_texts = self.chunker.split(text)
        segments = [
            KnowledgeSourceSegment(text=chunk_text, locator_type="chunk")
            for chunk_text in chunk_texts
        ]

        return self.index_segments(
            document_id=document_id,
            company_id=company_id,
            project_id=project_id,
            filename=filename,
            source_type=source_type,
            segments=segments,
            source_id=source_id,
            mime_type=mime_type,
            parser=parser,
            uploaded_by_user_id=uploaded_by_user_id,
        )

    def index_segments(
        self,
        *,
        document_id: str,
        company_id: str,
        project_id: int,
        filename: str,
        source_type: KnowledgeSourceType,
        segments: Sequence[KnowledgeSourceSegment],
        source_id: str | None = None,
        mime_type: str | None = None,
        parser: str | None = None,
        uploaded_by_user_id: str | None = None,
    ) -> KnowledgeIndexResult:
        clean_segments = [
            segment for segment in segments if segment.text and segment.text.strip()
        ]
        full_text = "\n".join(segment.text.strip() for segment in clean_segments)

        document = KnowledgeDocument.from_text(
            document_id=document_id,
            company_id=company_id,
            project_id=project_id,
            filename=filename,
            source_type=source_type,
            text=full_text,
            source_id=source_id,
            mime_type=mime_type,
            parser=parser,
            uploaded_by_user_id=uploaded_by_user_id,
        )

        chunks = [
            KnowledgeChunk.from_document(
                document,
                chunk_index=index,
                text=segment.text.strip(),
                locator_type=segment.locator_type,
                page_number=segment.page_number,
                paragraph_start=segment.paragraph_start,
                paragraph_end=segment.paragraph_end,
                metadata=segment.metadata or {},
            )
            for index, segment in enumerate(clean_segments)
        ]

        count = self.repository.upsert_chunks(chunks)

        return KnowledgeIndexResult(
            document_id=document.document_id,
            filename=document.filename,
            chunks_indexed=count,
            document_content_hash=document.content_hash,
        )
