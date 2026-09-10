from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from schemas.evidence import (
    EvidenceAuthority,
    EvidenceConflict,
    EvidenceEntityLink,
    EvidenceFrame,
    EvidenceRecord,
    EvidenceReviewStatus,
    EvidenceSourceType,
    new_evidence_frame_id,
)
from schemas.user_context import UserContext
from agent.scenario_aggregator import serialize_aggregated_summary


_ENTITY_ID_KEYS = (
    "sample_id",
    "experiment_id",
    "material_id",
    "record_id",
    "id",
)
_SAMPLE_KEYS = (
    "sample",
    "left_sample",
    "right_sample",
    "reference_sample",
)
_FIELD_SECTIONS = (
    "formula",
    "process",
    "performance",
    "service_performance",
)
_UNIT_ALIASES = {
    "": "",
    "percent": "%",
    "％": "%",
    "kj/m2": "kJ/m²",
    "kj/m²": "kJ/m²",
    "mpa": "MPa",
    "g/cm3": "g/cm³",
    "g/cm³": "g/cm³",
    "c": "°C",
    "℃": "°C",
    "degc": "°C",
}


class EvidenceFrameBuilder:
    """Normalize heterogeneous source payloads into one auditable frame.

    The builder is deliberately conservative: it does not invent a subject ID,
    does not convert across physical units, and preserves value/unit conflicts
    instead of selecting one side silently.
    """

    def __init__(self, ctx: UserContext, *, frame_id: str | None = None) -> None:
        self.ctx = ctx
        self.frame_id = frame_id or new_evidence_frame_id()
        self.records: list[EvidenceRecord] = []

    def add_mysql_result(self, result: Mapping[str, Any]) -> None:
        payload = dict(result or {})
        if payload.get("status") != "ok":
            self._add_status_record(
                payload,
                subject_id="mysql_query",
                attribute="status",
                value=payload.get("status"),
            )

        for key in _SAMPLE_KEYS:
            sample = payload.get(key)
            if isinstance(sample, Mapping):
                self._add_mysql_sample(key, sample, payload)

        for section in _FIELD_SECTIONS:
            self._add_field_list(
                subject=self._sample_subject(payload) or "mysql_query",
                entity_type="sample",
                section=section,
                fields=payload.get(section),
            )
        conditions = payload.get("conditions")
        if isinstance(conditions, Mapping):
            for field, value in conditions.items():
                self._add_record(
                    subject_id=self._sample_subject(payload) or "mysql_query",
                    entity_type="sample",
                    attribute=f"conditions.{field}",
                    value=value,
                    source_uri="mysql://authorized_material_query",
                )

        for item in payload.get("evidence") or []:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "mysql_record")
            record_id = item.get("record_id")
            self._add_record(
                subject_id=f"{source}:{record_id}",
                entity_type=self._entity_type(source),
                attribute="record_reference",
                value=record_id,
                source_uri=f"mysql://{source}/{record_id}",
            )

        self._add_collection_entities(payload.get("samples"))
        self._add_collection_entities(payload.get("matched_samples"), row_key="sample")
        self._add_collection_entities(payload.get("ranking"), row_key="sample")
        self._add_diff_records(payload)

    def add_derived_result(self, result: Mapping[str, Any]) -> str:
        """Register deterministic analysis output as traceable derived evidence."""
        analysis_type = str(result.get("analysis_type") or "research_analysis")
        if analysis_type == "scenario_aggregated_summary":
            summary = result.get("summary") or {}
            value = serialize_aggregated_summary(summary)
            record_id = self._stable_id("derived", ["scenario_aggregated_summary", value])
            self.records.append(
                EvidenceRecord(
                    record_id=record_id,
                    subject_id=f"aggregation:scenario_{result.get('scenario_id')}",
                    entity_type="scenario_summary",
                    attribute="structured_summary",
                    value=value,
                    source_type=EvidenceSourceType.DERIVED,
                    source_uri=f"derived://scenario-aggregation/{result.get('scenario_id')}",
                    confidence=1.0,
                    authority_level=EvidenceAuthority.EXTRACTED,
                    review_status=EvidenceReviewStatus.AUTO,
                    permission_scope=self._permission_scope(),
                    alignment_level="L1",
                    metadata={
                        "deterministic": True,
                        "priority": 80,
                    },
                )
            )
            return record_id
        compact = {
            "status": result.get("status"),
            "analysis_type": analysis_type,
            "scenario_id": result.get("scenario_id"),
            "target": result.get("target"),
            "targets": result.get("targets"),
            "filters": result.get("filters"),
            "sample_count": (result.get("dataset") or {}).get("sample_count"),
            "scan_complete": (result.get("dataset") or {}).get("scan_complete"),
            "scan_truncated": (result.get("dataset") or {}).get("scan_truncated"),
            "variables": result.get("variables"),
            "conflicts": result.get("conflicts"),
            "differences": result.get("differences"),
            "windows": result.get("windows"),
            "scope": result.get("scope"),
            "target_statistics": result.get("target_statistics"),
            "feature_coverage": result.get("feature_coverage"),
            "warnings": result.get("warnings"),
            "conclusion_limit": result.get("conclusion_limit"),
        }
        value = json.dumps(
            compact, ensure_ascii=False, sort_keys=True, default=str
        )
        record_id = self._stable_id("derived", [analysis_type, value])
        self.records.append(
            EvidenceRecord(
                record_id=record_id,
                subject_id=f"analysis:{analysis_type}",
                entity_type="analysis_result",
                attribute="deterministic_analysis",
                value=value,
                source_type=EvidenceSourceType.DERIVED,
                source_uri=f"derived://research-analysis/{analysis_type}",
                confidence=1.0,
                authority_level=EvidenceAuthority.EXTRACTED,
                review_status=EvidenceReviewStatus.AUTO,
                permission_scope=self._permission_scope(),
                alignment_level="L1",
                metadata={
                    "deterministic": True,
                    "supporting_record_refs": list(
                        result.get("evidence_refs") or []
                    )[:100],
                    "calculation_policy": (result.get("dataset") or {}).get(
                        "calculation_policy"
                    ),
                },
            )
        )
        return record_id

    def add_vector_result(self, result: Mapping[str, Any]) -> None:
        hits = result.get("hits")
        if not isinstance(hits, list):
            return
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            metadata = dict(hit.get("metadata") or {})
            point_id = str(
                hit.get("point_id")
                or hit.get("chunk_id")
                or metadata.get("point_id")
                or metadata.get("chunk_id")
                or self._stable_id("vector", hit)
            )
            text = str(hit.get("text") or "")
            self.records.append(
                EvidenceRecord(
                    record_id=self._stable_id("vector", [point_id, text]),
                    subject_id=f"vector_chunk:{point_id}",
                    entity_type=str(metadata.get("entity_type") or "knowledge_chunk"),
                    attribute="text",
                    value=text,
                    source_type=EvidenceSourceType.VECTOR_API,
                    source_uri=str(
                        hit.get("source_uri")
                        or metadata.get("source_uri")
                        or f"vector://{point_id}"
                    ),
                    chunk_id=point_id,
                    confidence=self._score(hit.get("score")),
                    authority_level=EvidenceAuthority.DOCUMENT,
                    review_status=EvidenceReviewStatus.AUTO,
                    permission_scope=self._permission_scope(metadata),
                    alignment_level="L4",
                    metadata=metadata,
                )
            )

    def add_upload_records(self, attachments: Iterable[Any]) -> None:
        for attachment in attachments:
            attachment_id = str(getattr(attachment, "attachment_id", "") or "")
            if not attachment_id:
                continue
            chunks = list(getattr(attachment, "chunks", ()) or ())
            selected = chunks[:20]
            for index, chunk in enumerate(selected, start=1):
                payload = dict(chunk or {})
                text = str(payload.get("text") or "")
                self.records.append(
                    EvidenceRecord(
                        record_id=self._stable_id(
                            "upload", [attachment_id, index, text]
                        ),
                        subject_id=f"upload:{attachment_id}",
                        entity_type="upload_chunk",
                        attribute=f"chunk.{index}",
                        value=text,
                        source_type=EvidenceSourceType.UPLOAD,
                        source_uri=f"upload://{attachment_id}#{index}",
                        confidence=1.0,
                        authority_level=EvidenceAuthority.USER_INPUT,
                        review_status=EvidenceReviewStatus.PENDING_REVIEW,
                        permission_scope=self._permission_scope(),
                        metadata={
                            "filename": getattr(attachment, "filename", None),
                            "parser": getattr(attachment, "parser", None),
                            "chunk_total": len(chunks),
                            "chunk_index": index,
                        },
                    )
                )

    def add_dialog_record(self, message: str) -> None:
        self.records.append(
            EvidenceRecord(
                record_id=self._stable_id("dialog", [message]),
                subject_id=f"dialog:{self.ctx.user_id}",
                entity_type="user_request",
                attribute="question",
                value=message,
                source_type=EvidenceSourceType.DIALOG,
                source_uri=f"dialog://{self.ctx.user_id}",
                confidence=1.0,
                authority_level=EvidenceAuthority.USER_INPUT,
                review_status=EvidenceReviewStatus.AUTO,
                permission_scope=self._permission_scope(),
                metadata={"turn_constraint_only": True},
            )
        )

    def build(self, *, warnings: Iterable[str] = ()) -> EvidenceFrame:
        conflicts = self._detect_conflicts()
        self._assign_conflict_groups(conflicts)
        source_counts = Counter(record.source_type.value for record in self.records)
        return EvidenceFrame(
            frame_id=self.frame_id,
            records=self.records,
            conflicts=conflicts,
            entity_links=self._entity_links(),
            source_summary=dict(sorted(source_counts.items())),
            warnings=[str(item) for item in warnings if str(item).strip()],
        )

    def _add_mysql_sample(
        self,
        key: str,
        sample: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        subject = f"sample:{sample.get('id')}"
        entity_type = "sample"
        for field, value in sample.items():
            if field in {"id"}:
                continue
            self._add_record(
                subject_id=subject,
                entity_type=entity_type,
                attribute=field,
                value=value,
                source_uri="mysql://eln_sample",
                observed_at=self._observed_at(sample),
            )
        if key == "sample":
            return
        # Keep left/right values addressable when a comparison contains both.
        prefix = "left" if key == "left_sample" else "right"
        for section in _FIELD_SECTIONS:
            diff = payload.get(f"{self._short_section(section)}_diff") or payload.get(
                f"{section}_diff"
            )
            if not isinstance(diff, Mapping):
                continue
            for item in diff.get("changed") or []:
                if not isinstance(item, Mapping):
                    continue
                field_name = str(item.get("field") or "unknown")
                value_key = prefix if prefix in item else f"{prefix}_value"
                if value_key not in item:
                    continue
                self._add_record(
                    subject_id=subject,
                    entity_type=entity_type,
                    attribute=f"{section}.{field_name}",
                    value=item.get(value_key),
                    unit=item.get(f"{prefix}_unit") or item.get("unit"),
                    source_uri="mysql://compare_samples",
                )

    def _add_field_list(
        self,
        *,
        subject: str,
        entity_type: str,
        section: str,
        fields: Any,
    ) -> None:
        if isinstance(fields, Mapping):
            fields = [
                {"field": key, "value": value} for key, value in fields.items()
            ]
        if not isinstance(fields, list):
            return
        for item in fields:
            if not isinstance(item, Mapping):
                continue
            field_name = str(
                item.get("name") or item.get("field") or item.get("raw_key") or "-"
            )
            self._add_record(
                subject_id=subject,
                entity_type=entity_type,
                attribute=f"{section}.{field_name}",
                value=item.get("value"),
                unit=item.get("unit"),
                source_uri=str(item.get("source") or "mysql://authorized_material_query"),
                metadata={
                    key: item.get(key)
                    for key in ("resolved", "raw_key", "field_id")
                    if key in item
                },
            )

    def _add_collection_entities(
        self,
        rows: Any,
        *,
        row_key: str | None = None,
    ) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if row_key is not None:
                row = row.get(row_key) if isinstance(row, Mapping) else None
            payload = row or {}
            sample = payload.get("sample") if isinstance(payload, Mapping) else None
            if isinstance(sample, Mapping) and sample.get("id") is not None:
                subject = f"sample:{sample['id']}"
                for section in _FIELD_SECTIONS:
                    self._add_field_list(
                        subject=subject,
                        entity_type="sample",
                        section=section,
                        fields=payload.get(section),
                    )
                if isinstance(payload.get("conditions"), Mapping):
                    for field, value in payload["conditions"].items():
                        self._add_record(
                            subject_id=subject,
                            entity_type="sample",
                            attribute=f"conditions.{field}",
                            value=value,
                            source_uri="mysql://authorized_material_query",
                        )
                # Add the sample's own scalar fields under the sample subject.
                # project_id is a foreign key reference, not the entity id, so it
                # must never become the subject of the sample's own fields.
                for field, value in sample.items():
                    if field == "id" or isinstance(value, (Mapping, list)):
                        continue
                    self._add_record(
                        subject_id=subject,
                        entity_type="sample",
                        attribute=field,
                        value=value,
                        source_uri="mysql://eln_sample",
                        observed_at=self._observed_at(sample),
                    )
            else:
                self._walk_entity(payload)

    def _add_diff_records(self, payload: Mapping[str, Any]) -> None:
        left = payload.get("left_sample") or {}
        right = payload.get("right_sample") or {}
        left_id = str(left.get("id") or "left")
        right_id = str(right.get("id") or "right")
        for key, value in payload.items():
            if not key.endswith("_diff") or not isinstance(value, Mapping):
                continue
            section = key[:-5]
            for item in value.get("changed") or []:
                if not isinstance(item, Mapping):
                    continue
                field_name = str(item.get("field") or "unknown")
                for side, sample_id in (("left", left_id), ("right", right_id)):
                    self._add_record(
                        subject_id=f"sample:{sample_id}",
                        entity_type="sample",
                        attribute=f"{section}.{field_name}",
                        value=item.get(side),
                        unit=item.get(f"{side}_unit") or item.get("unit"),
                        source_uri="mysql://compare_samples",
                    )

    def _walk_entity(self, payload: Mapping[str, Any], parent: str = "mysql") -> None:
        if not isinstance(payload, Mapping):
            return
        subject_id = next(
            (f"{key}:{payload[key]}" for key in _ENTITY_ID_KEYS if payload.get(key) is not None),
            None,
        )
        if subject_id is not None:
            entity_type = self._entity_type(subject_id.split(":", 1)[0])
            for field, value in payload.items():
                if field in _ENTITY_ID_KEYS:
                    continue
                if isinstance(value, (Mapping, list)):
                    continue
                self._add_record(
                    subject_id=subject_id,
                    entity_type=entity_type,
                    attribute=field,
                    value=value,
                    source_uri=f"mysql://{parent}",
                    observed_at=self._observed_at(payload),
                )
        for value in payload.values():
            if isinstance(value, Mapping):
                self._walk_entity(value, parent=parent)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        self._walk_entity(item, parent=parent)

    def _add_record(
        self,
        *,
        subject_id: str,
        entity_type: str,
        attribute: str,
        value: Any,
        source_uri: str,
        unit: str | None = None,
        observed_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.records.append(
            EvidenceRecord(
                record_id=self._stable_id("mysql", [subject_id, attribute, value, unit]),
                subject_id=subject_id,
                entity_type=entity_type,
                attribute=attribute,
                value=value,
                unit=unit,
                observed_at=observed_at,
                source_type=EvidenceSourceType.MYSQL,
                source_uri=source_uri,
                confidence=1.0,
                authority_level=EvidenceAuthority.AUTHORITATIVE,
                review_status=EvidenceReviewStatus.AUTO,
                permission_scope=self._permission_scope(),
                alignment_level="L1",
                metadata=dict(metadata or {}),
            )
        )

    def _add_status_record(
        self,
        payload: Mapping[str, Any],
        *,
        subject_id: str,
        attribute: str,
        value: Any,
    ) -> None:
        self._add_record(
            subject_id=subject_id,
            entity_type="query_status",
            attribute=attribute,
            value=value,
            source_uri="mysql://authorized_material_query",
            metadata={
                key: payload.get(key)
                for key in ("identifier", "warnings", "calculation_policy")
                if key in payload
            },
        )

    def _detect_conflicts(self) -> list[EvidenceConflict]:
        grouped: dict[tuple[str, str, str], list[EvidenceRecord]] = defaultdict(list)
        for record in self.records:
            if record.entity_type == "user_request":
                continue
            grouped[
                (
                    record.subject_id.casefold(),
                    record.entity_type.casefold(),
                    self._normalize_attribute(record.attribute),
                )
            ].append(record)

        conflicts: list[EvidenceConflict] = []
        for index, ((subject, entity_type, attribute), records) in enumerate(
            sorted(grouped.items()), start=1
        ):
            if len(records) < 2:
                continue
            values = [self._normalized_value(record.value) for record in records]
            units = [self._normalize_unit(record.unit) for record in records]
            reason: str | None = None
            if any(unit != units[0] for unit in units):
                reason = "UNIT_MISMATCH" if all(units) else "MISSING_UNIT"
            elif len(set(values)) > 1:
                reason = (
                    "VALUE_MISMATCH"
                    if all(value is not None for value in values)
                    else "TEXT_MISMATCH"
                )
            if reason is None:
                continue
            conflicts.append(
                EvidenceConflict(
                    conflict_group=f"conflict-{self.frame_id}-{index}",
                    subject_id=subject,
                    entity_type=entity_type,
                    attribute=attribute,
                    record_ids=[record.record_id for record in records],
                    reason=reason,  # type: ignore[arg-type]
                )
            )
        return conflicts

    def _assign_conflict_groups(self, conflicts: list[EvidenceConflict]) -> None:
        by_record = {
            record_id: conflict.conflict_group
            for conflict in conflicts
            for record_id in conflict.record_ids
        }
        for record in self.records:
            if record.record_id in by_record:
                record.conflict_group = by_record[record.record_id]

    def _entity_links(self) -> list[EvidenceEntityLink]:
        grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in self.records:
            if record.alignment_level == "L1":
                grouped[(record.entity_type, record.subject_id.casefold())].add(
                    record.subject_id
                )
        return [
            EvidenceEntityLink(
                link_id=self._stable_id("link", [entity_type, *sorted(subject_ids)]),
                entity_type=entity_type,
                subject_ids=sorted(subject_ids),
                alignment_level="L1",
                basis={"key": "normalized_subject_id"},
            )
            for (entity_type, _), subject_ids in sorted(grouped.items())
            if len(subject_ids) > 1
        ]

    def _permission_scope(
        self,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = metadata or {}
        project_ids = payload.get("project_id")
        return {
            "company_id": self.ctx.company_id,
            "project_ids": (
                [int(project_ids)]
                if project_ids is not None
                else list(self.ctx.project_ids)
            ),
            "all_projects": bool(
                payload.get("all_projects", self.ctx.all_projects)
            ),
        }

    @staticmethod
    def _sample_subject(payload: Mapping[str, Any]) -> str | None:
        sample = payload.get("sample")
        if isinstance(sample, Mapping) and sample.get("id") is not None:
            return f"sample:{sample['id']}"
        return None

    @staticmethod
    def _entity_type(source: str) -> str:
        value = re.sub(r"[^a-z0-9_]+", "_", source.casefold()).strip("_")
        return value or "record"

    @staticmethod
    def _short_section(section: str) -> str:
        return "service_performance" if section == "service_performance" else section

    @staticmethod
    def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
        serialized = json.dumps(
            list(parts), ensure_ascii=False, sort_keys=True, default=str
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}-{digest}"

    @staticmethod
    def _observed_at(payload: Mapping[str, Any]) -> str | None:
        for key in ("observed_at", "test_time", "create_time", "update_time"):
            if payload.get(key) is not None:
                return str(payload[key])
        return None

    @staticmethod
    def _normalize_attribute(attribute: str) -> str:
        return re.sub(r"\s+", "", str(attribute or "")).casefold()

    @staticmethod
    def _normalize_unit(unit: str | None) -> str:
        value = re.sub(r"\s+", "", str(unit or "")).casefold()
        return _UNIT_ALIASES.get(value, str(unit or "").strip())

    @staticmethod
    def _normalized_value(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value)).normalize()
        try:
            return Decimal(str(value)).normalize()
        except (InvalidOperation, ValueError):
            return str(value).strip()

    @staticmethod
    def _score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 1.0
        if 0 <= score <= 1:
            return score
        if 0 <= score <= 100:
            return score / 100
        return 1.0
