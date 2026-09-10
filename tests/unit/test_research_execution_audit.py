from __future__ import annotations

from runtime import research_execution_audit as audit


def test_research_execution_audit_is_company_and_user_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "audit.jsonl")
    audit.record_research_execution(
        user_id="user-a",
        company_id="company-a",
        scenario_id=1,
        workflow_id="hybrid_search_rank",
        strategy="structured_similarity",
        status="ok",
        evidence_count=12,
        vector_hit_count=0,
        synthesis_status="ok",
        missing_dimensions=["documents"],
    )
    audit.record_research_execution(
        user_id="user-b",
        company_id="company-b",
        scenario_id=1,
        workflow_id="hybrid_search_rank",
        strategy="structured_similarity",
        status="ok",
        evidence_count=3,
        vector_hit_count=0,
        synthesis_status="ok",
        missing_dimensions=[],
    )

    items = audit.list_research_executions(
        user_id="user-a",
        company_id="company-a",
    )
    assert len(items) == 1
    assert items[0]["scenario_id"] == 1
    assert items[0]["missing_dimensions"] == ["documents"]
