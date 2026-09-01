from __future__ import annotations

from agent.router import RuleIntentRouter
from runtime.progress import emit_progress, progress_context
from schemas.user_context import UserContext
from skills.material_intelligence import MaterialIntelligenceSkill


def field(name, value, unit):
    return {"name": name, "value": value, "unit": unit, "resolved": True}


REFERENCE = {
    "status": "ok",
    "sample": {"id": 3811, "name": "trial_6", "project_id": 115},
    "formula": [field("PC", "60", "%"), field("ABS", "40", "%")],
    "process": [field("注塑温度", "80", "℃"), field("螺杆转速", "100", "r/min")],
    "performance": [],
    "conditions": {},
    "evidence": [{"source": "eln_sample", "record_id": 3811}],
    "warnings": [],
}


def sample(sample_id, name, formula, process):
    return {
        "sample": {"id": sample_id, "name": name, "project_id": 115},
        "formula": formula,
        "process": process,
        "performance": [],
        "conditions": {},
    }


SOURCE = {
    "status": "ok",
    "count": 5,
    "total_matches": 5,
    "scan_page_size": 500,
    "scan_page_count": 1,
    "scan_complete": True,
    "scan_truncated": False,
    "samples": [
        sample(3811, "trial_6", REFERENCE["formula"], REFERENCE["process"]),
        sample(
            3809,
            "close_both",
            [field("PC", "58", "%"), field("ABS", "42", "%")],
            [field("注塑温度", "78", "℃"), field("螺杆转速", "90", "r/min")],
        ),
        sample(
            3808,
            "exact_formula_only",
            [field("PC", "60", "%"), field("ABS", "40", "%")],
            [],
        ),
        sample(
            3807,
            "far",
            [field("PC", "30", "%"), field("ABS", "70", "%")],
            [field("注塑温度", "20", "℃"), field("螺杆转速", "10", "r/min")],
        ),
        sample(
            3806,
            "wrong_units",
            [field("PC", "60", "phr"), field("ABS", "40", "phr")],
            [field("注塑温度", "80", "K")],
        ),
    ],
    "warnings": [],
}


class Registry:
    def __init__(self):
        self.calls = []

    def execute(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "get_sample_context":
            return REFERENCE
        if name == "list_samples_for_analysis":
            return SOURCE
        raise AssertionError(name)


CTX = UserContext(
    user_id="u1",
    company_id="c1",
    project_ids=(),
    permission_source="test",
    all_projects=True,
)


def run(scope="combined", top_n=5):
    registry = Registry()
    result = MaterialIntelligenceSkill(registry).execute_intent(
        "similar_samples",
        "list_samples_for_analysis",
        {
            "identifier": 3811,
            "similarity_scope": scope,
            "top_n": top_n,
        },
        CTX,
    )
    return result, registry


def test_combined_similarity_penalizes_missing_process_coverage():
    result, _ = run("combined")
    assert result["status"] == "ok"
    assert [row["sample"]["id"] for row in result["ranking"][:2]] == [3809, 3808]
    assert result["ranking"][0]["compared_field_count"] == 4
    assert result["ranking"][1]["process_similarity_percent"] == "0.00"
    assert result["ranking"][1]["similarity_percent"] == "50.00"


def test_formula_only_mode_ranks_exact_formula_first():
    result, _ = run("formula", top_n=2)
    assert [row["sample"]["id"] for row in result["ranking"]] == [3808, 3809]
    assert result["ranking"][0]["similarity_percent"] == "100.00"


def test_reference_is_excluded_and_wrong_units_are_not_compared():
    result, registry = run("combined")
    ranked_ids = [row["sample"]["id"] for row in result["ranking"]]
    assert 3811 not in ranked_ids
    assert 3806 not in ranked_ids
    assert result["excluded_candidate_count"] == 1
    assert [name for name, _ in registry.calls] == [
        "get_sample_context",
        "list_samples_for_analysis",
    ]


def test_rule_router_recognizes_scope_and_keeps_history_similarity_separate():
    router = RuleIntentRouter()
    combined = router.route("找和3811最像的5个样品")
    assert combined.intent == "similar_samples"
    assert combined.tool_args == {
        "identifier": "3811",
        "similarity_scope": "combined",
        "top_n": 5,
        "keyword": "",
    }
    formula = router.route("找配方和3811相似的样品")
    assert formula.intent == "similar_samples"
    assert formula.tool_args["similarity_scope"] == "formula"
    assert router.route("以前有没有和3811类似的情况") is None


def test_progress_events_are_request_local_and_user_safe():
    events = []
    emit_progress("ignored", "running", "ignored", "ignored")
    assert events == []
    with progress_context(events.append):
        emit_progress("database_scan", "completed", "读取完成", "已读取5条记录", count=5)
    assert events[0]["stage"] == "database_scan"
    assert events[0]["count"] == 5
    assert isinstance(events[0]["elapsed_ms"], int)
    assert events[0]["schema_version"] == "1.1"
    assert events[0]["source"] == "backend"
