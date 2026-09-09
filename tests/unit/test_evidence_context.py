from __future__ import annotations

from agent.evidence_context import EvidenceContextCompressor
from agent.evidence_frame import EvidenceFrameBuilder
from schemas.user_context import UserContext


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-1",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


def test_context_compressor_bounds_llm_view_without_mutating_frame():
    builder = EvidenceFrameBuilder(_ctx())
    builder.add_dialog_record("查找 EXP-128 和相近历史资料")
    for index in range(60):
        builder._add_record(
            subject_id=f"sample:{index}",
            entity_type="sample",
            attribute="formula.PC",
            value=index,
            unit="%",
            source_uri="mysql://eln_sample",
        )
    frame = builder.build()
    compressor = EvidenceContextCompressor(
        max_records=20,
        max_chars=4_000,
        max_value_chars=200,
    )

    context, serialized = compressor.build(
        frame=frame,
        query="查找 EXP-128 和相近历史资料",
    )

    assert len(frame.records) == 61
    assert context["selection"]["total_records"] == 61
    assert context["selection"]["selected_records"] <= 20
    assert context["selection"]["omitted_records"] >= 41
    assert len(serialized) <= 4_000
    assert "permission_scope" not in serialized


def test_aggressive_retry_uses_half_the_record_budget():
    builder = EvidenceFrameBuilder(_ctx())
    for index in range(40):
        builder._add_record(
            subject_id=f"sample:{index}",
            entity_type="sample",
            attribute="formula.PC",
            value=index,
            source_uri="mysql://eln_sample",
        )
    frame = builder.build()
    compressor = EvidenceContextCompressor(max_records=20, max_chars=4_000)

    first, first_serialized = compressor.build(frame=frame, query="配方")
    second, second_serialized = compressor.build(
        frame=frame,
        query="配方",
        aggressive=True,
    )

    assert first["selection"]["selected_records"] <= 20
    assert second["selection"]["selected_records"] <= 10
    assert second["selection"]["aggressive_retry"] is True
    assert len(second_serialized) <= len(first_serialized)


def test_context_compressor_keeps_absolute_bound_with_many_conflicts():
    builder = EvidenceFrameBuilder(_ctx())
    for index in range(30):
        left = index * 2
        right = left + 1
        builder._add_record(
            subject_id=f"sample:{index}",
            entity_type="sample",
            attribute="performance.value",
            value=left,
            source_uri="mysql://left",
        )
        builder._add_record(
            subject_id=f"sample:{index}",
            entity_type="sample",
            attribute="performance.value",
            value=right,
            source_uri="mysql://right",
        )
    frame = builder.build(warnings=[f"warning-{index}" for index in range(30)])
    compressor = EvidenceContextCompressor(max_records=20, max_chars=4_000)

    context, serialized = compressor.build(frame=frame, query="性能冲突")

    assert len(serialized) <= 4_000
    assert len(context["conflicts"]) <= 20
    assert len(context["warnings"]) <= 8
    assert context["selection"]["selected_records"] <= 20
