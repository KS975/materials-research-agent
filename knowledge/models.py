from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator


KnowledgeSourceType = Literal["platform_file", "manual_index", "chat_upload"]
KnowledgeLocatorType = Literal["page", "paragraph", "chunk", "unknown"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class KnowledgeDocument(BaseModel):
    """Metadata for one long-term knowledge document.

    The document itself is not a Qdrant point. Its chunks are.
    company_id/project_id are copied to every KnowledgeChunk so retrieval can
    always apply permission filters before content is returned.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str
    company_id: str
    project_id: int
    filename: str
    source_type: KnowledgeSourceType
    source_id: str | None = None
    mime_type: str | None = None
    parser: str | None = None
    uploaded_by_user_id: str | None = None
    content_hash: str
    indexed_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_text(
        cls,
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
    ) -> "KnowledgeDocument":
        return cls(
            document_id=document_id,
            company_id=company_id,
            project_id=project_id,
            filename=filename,
            source_type=source_type,
            source_id=source_id,
            mime_type=mime_type,
            parser=parser,
            uploaded_by_user_id=uploaded_by_user_id,
            content_hash=sha256_text(text),
        )

    def qdrant_payload_base(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "company_id": self.company_id,
            "project_id": self.project_id,
            "filename": self.filename,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "mime_type": self.mime_type,
            "parser": self.parser,
            "uploaded_by_user_id": self.uploaded_by_user_id,
            "document_content_hash": self.content_hash,
            "indexed_at": self.indexed_at.isoformat(),
        }


class KnowledgeChunk(BaseModel):
    """One searchable knowledge chunk and the payload stored with its vector."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    company_id: str
    project_id: int
    filename: str
    source_type: KnowledgeSourceType

    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)

    source_id: str | None = None
    uploaded_by_user_id: str | None = None

    locator_type: KnowledgeLocatorType = "chunk"
    page_number: int | None = Field(default=None, ge=1)
    paragraph_start: int | None = Field(default=None, ge=0)
    paragraph_end: int | None = Field(default=None, ge=0)

    metadata: dict[str, Any] = Field(default_factory=dict)

    content_hash: str | None = None
    point_id: str | None = None

    @model_validator(mode="after")
    def fill_deterministic_fields(self) -> "KnowledgeChunk":
        if not self.content_hash:
            self.content_hash = sha256_text(self.text)

        if not self.point_id:
            seed = (
                f"materials-research-agent:"
                f"{self.company_id}:{self.project_id}:"
                f"{self.document_id}:{self.chunk_index}:{self.content_hash}"
            )
            self.point_id = str(uuid5(NAMESPACE_URL, seed))

        return self

    @classmethod
    def from_document(
        cls,
        document: KnowledgeDocument,
        *,
        chunk_index: int,
        text: str,
        locator_type: KnowledgeLocatorType = "chunk",
        page_number: int | None = None,
        paragraph_start: int | None = None,
        paragraph_end: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "KnowledgeChunk":
        return cls(
            document_id=document.document_id,
            company_id=document.company_id,
            project_id=document.project_id,
            filename=document.filename,
            source_type=document.source_type,
            source_id=document.source_id,
            uploaded_by_user_id=document.uploaded_by_user_id,
            chunk_index=chunk_index,
            text=text,
            locator_type=locator_type,
            page_number=page_number,
            paragraph_start=paragraph_start,
            paragraph_end=paragraph_end,
            metadata=metadata or {},
        )

    def qdrant_payload(self) -> dict[str, Any]:
        """Payload copied beside the vector in Qdrant.

        Permission-critical fields are deliberately first-class payload fields,
        not buried inside metadata.
        """
        return {
            "company_id": self.company_id,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "filename": self.filename,
            "uploaded_by_user_id": self.uploaded_by_user_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "content_hash": self.content_hash,
            "locator_type": self.locator_type,
            "page_number": self.page_number,
            "paragraph_start": self.paragraph_start,
            "paragraph_end": self.paragraph_end,
            "metadata": self.metadata,
        }
