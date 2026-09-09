from __future__ import annotations

from agent.evidence_frame import EvidenceFrameBuilder
from agent.evidence_relation import build_source_relation
from schemas.user_context import UserContext


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-1",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


def test_source_relation_keeps_topic_and_entity_levels_separate():
    topic_builder = EvidenceFrameBuilder(_ctx())
    topic_builder._add_record(
        subject_id="sample:128",
        entity_type="sample",
        attribute="name",
        value="EXP-128",
        source_uri="mysql://eln_sample",
    )
    topic_frame = topic_builder.build()
    relation = build_source_relation(
        topic_frame,
        {
            "hits": [
                {
                    "point_id": "doc-1",
                    "score": 0.9,
                    "text": "similar topic",
                    "metadata": {"company_id": "company-a"},
                }
            ]
        },
    )
    assert relation["level"] == "TOPIC_ONLY"
    assert "未建立实体级关联" in relation["warning"]

    entity_builder = EvidenceFrameBuilder(_ctx())
    entity_builder._add_record(
        subject_id="sample:128",
        entity_type="sample",
        attribute="name",
        value="EXP-128",
        source_uri="mysql://eln_sample",
    )
    entity_frame = entity_builder.build()
    relation = build_source_relation(
        entity_frame,
        {
            "hits": [
                {
                    "point_id": "doc-2",
                    "score": 0.9,
                    "text": "sample fact",
                    "metadata": {"sample_id": 128},
                }
            ]
        },
    )
    assert relation["level"] == "ENTITY_LINKED"
    assert relation["entity_linked_hit_count"] == 1
