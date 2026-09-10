from __future__ import annotations

import json

from agent.evidence_context import EvidenceContextCompressor


def test_bound_context_returns_json_when_under_limit() -> None:
    compressor = EvidenceContextCompressor(max_chars=4_000)
    context = {
        "structured_summary": {"aggregated_type": "similarity_ranking"},
        "source_summary": {"mysql": 2},
        "warnings": [],
    }
    serialized = compressor.bound_context(context)
    parsed = json.loads(serialized)
    assert parsed["structured_summary"]["aggregated_type"] == "similarity_ranking"


def test_bound_context_drops_optional_metadata_when_over_limit() -> None:
    compressor = EvidenceContextCompressor(max_chars=4_000)
    context = {
        "structured_summary": {"data": "x" * 3_500},
        "source_summary": {"mysql": 1},
        "conflict_count": 1,
        "evidence_record_count": 100,
        "warnings": ["w" * 1_000],
    }
    serialized = compressor.bound_context(context)
    assert len(serialized) <= 4_000
    assert "conflict_count" not in serialized
    assert "warnings" not in serialized
    assert '"data"' in serialized


def test_bound_context_hard_truncates_as_last_resort() -> None:
    compressor = EvidenceContextCompressor(max_chars=4_000)
    context = {
        "structured_summary": {"payload": "y" * 8_000},
    }
    serialized = compressor.bound_context(context)
    assert serialized.endswith("...[truncated]")
    assert len(serialized) <= 4_000


def test_aggressive_bound_uses_smaller_budget() -> None:
    compressor = EvidenceContextCompressor(max_chars=4_000)
    context = {
        "structured_summary": {"payload": "z" * 3_300},
        "warnings": ["w" * 300],
    }
    normal = compressor.bound_context(context)
    aggressive = compressor.bound_context(context, aggressive=True)
    assert len(aggressive) < len(normal)
