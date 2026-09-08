from __future__ import annotations

from agent.evidence_frame import EvidenceFrameBuilder
from schemas.user_context import UserContext


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-1",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


def test_evidence_frame_normalizes_mysql_vector_and_dialog_sources():
    builder = EvidenceFrameBuilder(_ctx())
    builder.add_dialog_record("查 EXP-128 并结合历史资料")
    builder.add_mysql_result(
        {
            "status": "ok",
            "sample": {
                "id": 128,
                "name": "EXP-128",
                "project_id": 115,
                "create_time": "2026-01-02",
            },
            "formula": [
                {"name": "PC", "value": 65, "unit": "%", "resolved": True}
            ],
            "evidence": [{"source": "eln_sample", "record_id": 128}],
        }
    )
    builder.add_vector_result(
        {
            "status": "ok",
            "hits": [
                {
                    "point_id": "point-1",
                    "score": 0.82,
                    "text": "历史项目曾使用相近配方。",
                    "metadata": {
                        "company_id": "company-a",
                        "project_id": 115,
                        "filename": "history.docx",
                    },
                    "source_uri": "vector://external/point-1",
                }
            ],
        }
    )

    frame = builder.build()
    assert frame.source_summary == {"dialog": 1, "mysql": 5, "vector_api": 1}
    source_types = {
        record.record_id: record.source_type.value for record in frame.records
    }
    assert set(source_types.values()) == {"mysql", "vector_api", "dialog"}
    assert all(
        record.permission_scope["company_id"] == "company-a"
        for record in frame.records
    )


def test_evidence_frame_preserves_conflicts_instead_of_selecting_one_side():
    builder = EvidenceFrameBuilder(_ctx())
    builder._add_record(
        subject_id="sample:128",
        entity_type="sample",
        attribute="performance.impact",
        value=35,
        unit="kJ/m²",
        source_uri="mysql://eln_sample",
    )
    builder._add_record(
        subject_id="sample:128",
        entity_type="sample",
        attribute="performance.impact",
        value=36,
        unit="kJ/m2",
        source_uri="mysql://eln_sample",
    )

    frame = builder.build()
    assert len(frame.conflicts) == 1
    conflict = frame.conflicts[0]
    assert conflict.reason == "VALUE_MISMATCH"
    assert conflict.resolution == "PRESERVE_ALL"
    assert all(
        record.conflict_group == conflict.conflict_group
        for record in frame.records
        if record.record_id in conflict.record_ids
    )
