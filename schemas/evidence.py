from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceSourceType(StrEnum):
    MYSQL = "mysql"
    VECTOR_API = "vector_api"
    UPLOAD = "upload"
    DIALOG = "dialog"
    DERIVED = "derived"


class EvidenceAuthority(StrEnum):
    AUTHORITATIVE = "AUTHORITATIVE"
    REVIEWED = "REVIEWED"
    DOCUMENT = "DOCUMENT"
    EXTRACTED = "EXTRACTED"
    USER_INPUT = "USER_INPUT"


class EvidenceReviewStatus(StrEnum):
    AUTO = "AUTO"
    REVIEWED = "REVIEWED"
    REJECTED = "REJECTED"
    PENDING_REVIEW = "PENDING_REVIEW"


class EvidenceRecord(BaseModel):
    record_id: str = Field(min_length=1, max_length=128)
    subject_id: str = Field(min_length=1, max_length=256)
    entity_type: str = Field(min_length=1, max_length=64)
    attribute: str = Field(min_length=1, max_length=256)
    value: Any = None
    unit: str | None = Field(default=None, max_length=64)
    observed_at: str | None = Field(default=None, max_length=64)
    source_type: EvidenceSourceType
    source_uri: str = Field(min_length=1, max_length=1024)
    chunk_id: str | None = Field(default=None, max_length=256)
    confidence: float = Field(ge=0, le=1, default=1.0)
    authority_level: EvidenceAuthority
    conflict_group: str | None = Field(default=None, max_length=128)
    review_status: EvidenceReviewStatus = EvidenceReviewStatus.AUTO
    permission_scope: dict[str, Any] = Field(default_factory=dict)
    alignment_level: Literal["L1", "L2", "L3", "L4"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceConflict(BaseModel):
    conflict_group: str = Field(min_length=1, max_length=128)
    subject_id: str = Field(min_length=1, max_length=256)
    entity_type: str = Field(min_length=1, max_length=64)
    attribute: str = Field(min_length=1, max_length=256)
    record_ids: list[str] = Field(min_length=1)
    reason: Literal[
        "VALUE_MISMATCH",
        "UNIT_MISMATCH",
        "MISSING_UNIT",
        "TEXT_MISMATCH",
    ]
    resolution: Literal["PRESERVE_ALL", "PREFER_AUTHORITATIVE"] = "PRESERVE_ALL"


class EvidenceEntityLink(BaseModel):
    link_id: str = Field(min_length=1, max_length=128)
    entity_type: str = Field(min_length=1, max_length=64)
    subject_ids: list[str] = Field(min_length=1)
    alignment_level: Literal["L1", "L2", "L3", "L4"]
    basis: dict[str, Any] = Field(default_factory=dict)


class EvidenceFrame(BaseModel):
    schema_version: Literal[1] = 1
    frame_id: str = Field(min_length=8, max_length=64)
    records: list[EvidenceRecord] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    entity_links: list[EvidenceEntityLink] = Field(default_factory=list)
    source_summary: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def new_evidence_frame_id() -> str:
    return f"evf-{uuid4().hex}"
