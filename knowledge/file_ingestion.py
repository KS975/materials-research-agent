from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from file_processing import UnifiedFileParser
from file_processing.models import ParsedDocument
from schemas.user_context import UserContext

from .indexer import KnowledgeIndexer, KnowledgeSourceSegment
from .repository import QdrantKnowledgeRepository
from .redaction import SecretRedactor


@dataclass(frozen=True)
class KnowledgeFileIndexResult:
    document_id: str
    source_id: str
    filename: str
    parser: str
    page_count: int | None
    char_count: int
    chunks_indexed: int
    content_hash: str
    secrets_redacted: int
    company_id: str
    project_id: int


class KnowledgeFileIngestionService:
    """Long-term knowledge ingestion using the already validated PDF/DOCX parser.

    Important boundary:
    - ChatFileParser is reused as the unified parser.
    - This service does NOT read or write ChatAttachmentStore.
    - Parsed chunks are converted to long-term KnowledgeSourceSegment objects
      and persisted through KnowledgeIndexer -> Qdrant.
    """

    def __init__(self, parser: UnifiedFileParser, redactor: SecretRedactor | None = None):
        self.parser = parser
        self.redactor = redactor or SecretRedactor()

    @staticmethod
    def _derive_source_id(filename: str, content: bytes) -> str:
        digest = sha256(filename.encode("utf-8") + b"\0" + content).hexdigest()
        return f"manual-upload:{digest[:32]}"

    @staticmethod
    def _derive_document_id(
        *,
        company_id: str,
        project_id: int,
        source_id: str,
    ) -> str:
        seed = f"materials-knowledge:{company_id}:{project_id}:{source_id}"
        return str(uuid5(NAMESPACE_URL, seed))

    def parsed_to_segments(self, parsed: ParsedDocument) -> tuple[list[KnowledgeSourceSegment], int]:
        segments: list[KnowledgeSourceSegment] = []
        total_redactions = 0

        for chunk in parsed.chunks:
            if chunk.page is not None:
                locator_type = "page"
            elif chunk.paragraph_start is not None:
                locator_type = "paragraph"
            else:
                locator_type = "chunk"

            redacted = self.redactor.redact(chunk.text)
            total_redactions += redacted.count
            segments.append(
                KnowledgeSourceSegment(
                    text=redacted.text,
                    locator_type=locator_type,
                    page_number=chunk.page,
                    paragraph_start=chunk.paragraph_start,
                    paragraph_end=chunk.paragraph_end,
                    metadata={
                        "parser_chunk_index": chunk.index,
                        "media_type": parsed.media_type,
                    },
                )
            )

        return segments, total_redactions

    def index_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        media_type: str,
        project_id: int,
        ctx: UserContext,
        repository: QdrantKnowledgeRepository,
        source_id: str | None = None,
    ) -> KnowledgeFileIndexResult:
        if not ctx.can_access_project(project_id):
            raise PermissionError("当前用户无权将文件写入该项目知识范围")

        parsed = self.parser.parse_bytes(filename, content, media_type)
        resolved_source_id = (source_id or "").strip() or self._derive_source_id(
            filename,
            content,
        )
        document_id = self._derive_document_id(
            company_id=ctx.company_id,
            project_id=project_id,
            source_id=resolved_source_id,
        )

        segments, secrets_redacted = self.parsed_to_segments(parsed)
        indexer = KnowledgeIndexer(repository=repository)
        result = indexer.index_segments(
            document_id=document_id,
            company_id=ctx.company_id,
            project_id=project_id,
            filename=filename,
            source_type="manual_index",
            source_id=resolved_source_id,
            mime_type=parsed.media_type,
            parser=parsed.parser,
            uploaded_by_user_id=ctx.user_id,
            segments=segments,
        )

        return KnowledgeFileIndexResult(
            document_id=result.document_id,
            source_id=resolved_source_id,
            filename=filename,
            parser=parsed.parser,
            page_count=parsed.page_count,
            char_count=len(parsed.text),
            chunks_indexed=result.chunks_indexed,
            content_hash=result.document_content_hash,
            secrets_redacted=secrets_redacted,
            company_id=ctx.company_id,
            project_id=project_id,
        )
