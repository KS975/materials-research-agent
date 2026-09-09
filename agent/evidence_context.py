from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from schemas.evidence import EvidenceFrame, EvidenceRecord


_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*|\d{2,}")


class EvidenceContextCompressor:
    """Build a bounded LLM view without mutating the auditable evidence frame."""

    def __init__(
        self,
        *,
        max_records: int = 80,
        max_chars: int = 24_000,
        max_value_chars: int = 1_200,
    ) -> None:
        self.max_records = max(10, min(int(max_records), 300))
        self.max_chars = max(4_000, min(int(max_chars), 80_000))
        self.max_value_chars = max(120, min(int(max_value_chars), 5_000))

    def build(
        self,
        *,
        frame: EvidenceFrame,
        query: str,
        aggressive: bool = False,
    ) -> tuple[dict[str, Any], str]:
        record_divisor = 4 if aggressive else 1
        char_divisor = 4 if aggressive else 1
        record_limit = max(10, self.max_records // record_divisor)
        char_limit = max(3_000, self.max_chars // char_divisor)
        selected = self._select_records(frame, query=query, record_limit=record_limit)
        context: dict[str, Any] = {
            "schema_version": 1,
            "frame_id": frame.frame_id,
            "question": self._bounded_text(query, 1_000),
            "source_summary": frame.source_summary,
            "warnings": [
                self._bounded_text(item, 300) for item in frame.warnings[:8]
            ],
            "selection": {
                "total_records": len(frame.records),
                "selected_records": 0,
                "omitted_records": len(frame.records),
                "record_limit": record_limit,
                "aggressive_retry": bool(aggressive),
                "omitted_warnings": max(0, len(frame.warnings) - 8),
                "omitted_conflicts": max(0, len(frame.conflicts) - 20),
            },
            "conflicts": [
                self._conflict_row(item) for item in frame.conflicts[:20]
            ],
            "records": [],
        }

        serialized_length = self._serialized_length(context)
        kept: list[dict[str, Any]] = []
        for row in selected:
            candidate = self._record_row(row["record"])
            candidate_length = len(json.dumps(candidate, ensure_ascii=False, default=str))
            if kept and serialized_length + candidate_length > char_limit:
                continue
            if not kept and candidate_length > char_limit:
                candidate = self._record_row(row["record"], force_value_chars=500)
                candidate_length = len(
                    json.dumps(candidate, ensure_ascii=False, default=str)
                )
                if candidate_length > char_limit:
                    continue
            kept.append(candidate)
            serialized_length += candidate_length

        context["records"] = kept
        context["selection"]["selected_records"] = len(kept)
        context["selection"]["omitted_records"] = max(0, len(frame.records) - len(kept))
        serialized = json.dumps(context, ensure_ascii=False, default=str)
        while len(serialized) > char_limit and context["warnings"]:
            context["warnings"].pop()
            context["selection"]["omitted_warnings"] += 1
            serialized = json.dumps(context, ensure_ascii=False, default=str)
        while len(serialized) > char_limit and context["conflicts"]:
            context["conflicts"].pop()
            context["selection"]["omitted_conflicts"] += 1
            serialized = json.dumps(context, ensure_ascii=False, default=str)
        while context["records"] and len(serialized) > char_limit:
            context["records"].pop()
            context["selection"]["selected_records"] -= 1
            context["selection"]["omitted_records"] += 1
            serialized = json.dumps(context, ensure_ascii=False, default=str)
        return context, serialized

    def _select_records(
        self,
        frame: EvidenceFrame,
        *,
        query: str,
        record_limit: int,
    ) -> list[dict[str, Any]]:
        conflict_ids = {
            record_id
            for conflict in frame.conflicts
            for record_id in conflict.record_ids
        }
        terms = self._query_terms(query)
        source_limits = {
            "mysql": max(1, int(record_limit * 0.60)),
            "derived": max(1, int(record_limit * 0.15)),
            "vector_api": max(1, int(record_limit * 0.15)),
            "upload": max(1, int(record_limit * 0.10)),
            "dialog": 1,
        }
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for index, record in enumerate(frame.records):
            groups[(record.source_type.value, record.subject_id.casefold())].append(
                {
                    "record": record,
                    "order": index,
                    "score": self._record_score(
                        record,
                        terms=terms,
                        in_conflict=record.record_id in conflict_ids,
                    ),
                }
            )
        for rows in groups.values():
            rows.sort(key=lambda item: (-item["score"], item["order"]))
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (-max(row["score"] for row in item[1]), item[0]),
        )

        selected: list[dict[str, Any]] = []
        source_counts: dict[str, int] = defaultdict(int)
        cursors = {key: 0 for key, _rows in ordered_groups}
        while len(selected) < record_limit:
            added = False
            for group_key in cursors:
                if len(selected) >= record_limit:
                    break
                source = group_key[0]
                rows = groups[group_key]
                cursor = cursors[group_key]
                if cursor >= len(rows):
                    continue
                if source_counts[source] >= source_limits.get(source, record_limit):
                    cursors[group_key] = len(rows)
                    continue
                selected.append(rows[cursor])
                cursors[group_key] = cursor + 1
                source_counts[source] += 1
                added = True
            if not added:
                break

        for group_key, rows in ordered_groups:
            if len(selected) >= record_limit:
                break
            source = group_key[0]
            cursor = cursors[group_key]
            while (
                cursor < len(rows)
                and len(selected) < record_limit
                and source_counts[source] < record_limit
            ):
                selected.append(rows[cursor])
                cursor += 1
                source_counts[source] += 1
            cursors[group_key] = cursor

        selected.sort(key=lambda item: (-item["score"], item["order"]))
        return selected

    def _record_score(
        self,
        record: EvidenceRecord,
        *,
        terms: tuple[str, ...],
        in_conflict: bool,
    ) -> float:
        searchable = " ".join(
            str(item)
            for item in (
                record.subject_id,
                record.entity_type,
                record.attribute,
                record.value,
                record.metadata.get("filename"),
                record.metadata.get("title"),
            )
        ).casefold()
        score = 0.0
        if record.source_type.value == "mysql":
            score += 20
        elif record.source_type.value == "derived":
            score += 30
        elif record.source_type.value == "vector_api":
            score += 14 + float(record.confidence) * 10
        elif record.source_type.value == "upload":
            score += 10
        if in_conflict:
            score += 35
        for term in terms:
            if term and term.casefold() in searchable:
                score += 18
        return score

    def _record_row(
        self,
        record: EvidenceRecord,
        *,
        force_value_chars: int | None = None,
    ) -> dict[str, Any]:
        value_chars = force_value_chars or self.max_value_chars
        return {
            "record_id": record.record_id,
            "source": record.source_type.value,
            "subject_id": record.subject_id,
            "entity_type": record.entity_type,
            "attribute": record.attribute,
            "value": self._bounded_text(record.value, value_chars),
            "unit": record.unit,
            "confidence": record.confidence,
            "authority_level": record.authority_level.value,
            "alignment_level": record.alignment_level,
            "source_uri": record.source_uri,
            "conflict_group": record.conflict_group,
        }

    @staticmethod
    def _conflict_row(conflict: Any) -> dict[str, Any]:
        record_ids = list(conflict.record_ids[:20])
        return {
            "conflict_group": conflict.conflict_group,
            "subject_id": conflict.subject_id,
            "entity_type": conflict.entity_type,
            "attribute": conflict.attribute,
            "record_ids": record_ids,
            "record_ids_truncated": max(
                0, len(conflict.record_ids) - len(record_ids)
            ),
            "reason": conflict.reason,
            "resolution": conflict.resolution,
        }

    @staticmethod
    def _bounded_text(value: Any, limit: int) -> Any:
        if not isinstance(value, str) or len(value) <= limit:
            return value
        return value[: max(0, limit - 20)] + f"...[truncated {len(value) - limit + 20} chars]"

    @staticmethod
    def _serialized_length(context: dict[str, Any]) -> int:
        return len(json.dumps(context, ensure_ascii=False, default=str))

    @staticmethod
    def _query_terms(query: str) -> tuple[str, ...]:
        normalized = str(query or "").strip()
        if not normalized:
            return ()
        terms = {normalized.casefold()}
        terms.update(
            match.group(0).casefold()
            for match in _LATIN_TOKEN.finditer(normalized)
        )
        for run in _CJK_RUN.finditer(normalized):
            value = run.group(0)
            terms.add(value.casefold())
            terms.update(
                value[index : index + 2].casefold()
                for index in range(max(1, len(value) - 1))
            )
        return tuple(sorted(terms))
