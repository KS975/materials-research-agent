from __future__ import annotations

from typing import Any


def build_source_relation(frame: Any, vector_result: dict[str, Any]) -> dict[str, Any]:
    mysql_subjects = {
        record.subject_id
        for record in frame.records
        if record.source_type.value == "mysql"
    }
    mysql_projects = {
        str(record.value)
        for record in frame.records
        if record.source_type.value == "mysql"
        and record.attribute == "project_id"
        and record.value is not None
    }
    entity_links = 0
    project_links = 0
    hits = vector_result.get("hits") or []
    for hit in hits:
        metadata = dict(hit.get("metadata") or {})
        sample_id = metadata.get("sample_id") or metadata.get("sampleId")
        experiment_id = metadata.get("experiment_id") or metadata.get("experimentId")
        if (
            sample_id is not None and f"sample:{sample_id}" in mysql_subjects
        ) or (
            experiment_id is not None
            and f"experiment:{experiment_id}" in mysql_subjects
        ):
            entity_links += 1
            continue
        project_id = metadata.get("project_id") or metadata.get("projectId")
        if project_id is not None and str(project_id) in mysql_projects:
            project_links += 1

    if entity_links:
        level = "ENTITY_LINKED"
        warning = "知识片段已与对应样品或实验建立精确关联。"
    elif project_links:
        level = "PROJECT_LINKED"
        warning = "知识片段仅关联到项目层级，不能视为对应某个样品或实验。"
    else:
        level = "TOPIC_ONLY"
        warning = "知识片段与结构化数据本轮仅主题相关，未建立精确关联，仅供参考。"
    return {
        "level": level,
        "mysql_evidence_level": "ENTITY_LEVEL",
        "vector_evidence_level": "DOCUMENT_CHUNK",
        "vector_hit_count": len(hits),
        "entity_linked_hit_count": entity_links,
        "project_linked_hit_count": project_links,
        "policy": (
            "不根据文本相似度推断实体等同；仅显式 sample/experiment/project ID "
            "可提升关联等级。"
        ),
        "warning": warning,
    }
