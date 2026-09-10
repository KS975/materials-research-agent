from __future__ import annotations

import json
from typing import Any


class EvidenceContextCompressor:
    """Bound a structured summary context for the LLM.

    The scenario aggregator already bounds semantic content (TopN samples,
    compact fields). This class is only a size guard: it drops optional
    metadata and, as a last resort, hard-truncates the serialized text. It
    never selects individual evidence records.
    """

    def __init__(
        self,
        *,
        max_records: int = 80,
        max_chars: int = 24_000,
        max_value_chars: int = 1_200,
    ) -> None:
        # max_records and max_value_chars are accepted for configuration
        # compatibility; semantic bounding is handled by the aggregator.
        _ = (max_records, max_value_chars)
        self.max_chars = max(4_000, min(int(max_chars), 80_000))

    def bound_context(
        self,
        context: dict[str, Any],
        *,
        aggressive: bool = False,
    ) -> str:
        max_chars = self.max_chars
        if aggressive:
            max_chars = max(3_000, max_chars // 4)

        serialized = json.dumps(context, ensure_ascii=False, default=str)
        if len(serialized) <= max_chars:
            return serialized

        trimmed = dict(context)
        for key in (
            "warnings",
            "source_summary",
            "conflict_count",
            "evidence_record_count",
        ):
            trimmed.pop(key, None)
        serialized = json.dumps(trimmed, ensure_ascii=False, default=str)
        if len(serialized) <= max_chars:
            return serialized

        return serialized[: max(1_000, max_chars - 40)] + "...[truncated]"
