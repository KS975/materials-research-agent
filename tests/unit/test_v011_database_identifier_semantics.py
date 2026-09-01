from __future__ import annotations

import json

from agent.tools import MaterialsTools
from agent.deepseek_intent_router import DeepSeekIntentRouter
from schemas.user_context import UserContext


class FakeSamples:
    def __init__(self):
        self.calls = []
        self.by_id = {
            3811: {
                "id": 3811,
                "name": "trial_6",
                "project_id": 115,
            },
            3809: {
                "id": 3809,
                "name": "trial_4",
                "project_id": 115,
            },
        }

    def get_by_id(self, sample_id, ctx):
        self.calls.append(("id", sample_id))
        return self.by_id.get(sample_id)

    def find_exact_name(self, name, ctx, limit=20):
        self.calls.append(("name", name))
        return []


class Dummy:
    pass


CTX = UserContext(
    user_id="local-test",
    company_id="6a4b19f62d0e000027001eb8",
    project_ids=(115,),
    permission_source="unit",
)


def make_tools(samples):
    return MaterialsTools(
        samples,
        Dummy(),
        Dummy(),
        Dummy(),
        Dummy(),
    )


def test_numeric_string_3811_is_database_id_not_sample_name():
    repo = FakeSamples()

    result = make_tools(repo)._locate_sample("3811", CTX)

    assert result["status"] == "ok"
    assert result["sample"]["id"] == 3811
    assert result["sample"]["name"] == "trial_6"
    assert repo.calls == [("id", 3811)]


def test_numeric_string_3809_is_database_id_not_sample_name():
    repo = FakeSamples()

    result = make_tools(repo)._locate_sample("3809", CTX)

    assert result["status"] == "ok"
    assert result["sample"]["id"] == 3809
    assert result["sample"]["name"] == "trial_4"
    assert repo.calls == [("id", 3809)]


def test_non_numeric_identifier_still_uses_exact_name():
    repo = FakeSamples()

    result = make_tools(repo)._locate_sample("trial_6", CTX)

    assert result["status"] == "not_found"
    assert repo.calls == [("name", "trial_6")]


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        return json.dumps(
            self.payload,
            ensure_ascii=False,
        )


def test_deepseek_numeric_identifiers_remain_numeric_ids():
    router = DeepSeekIntentRouter(
        FakeLLM({
            "intent": "analyze_performance_difference",
            "tool_name": "compare_samples",
            "tool_args": {
                "left_identifier": 3811,
                "right_identifier": 3809,
                "target_metric": "冲击强度",
                "direction_claim": "更低",
            },
            "reasoning_summary": "compare by sample db id",
        })
    )

    decision = router.route(
        "为什么 3811 的冲击强度比 3809 低？"
    )

    assert decision.tool_args["left_identifier"] == 3811
    assert decision.tool_args["right_identifier"] == 3809
    assert isinstance(
        decision.tool_args["left_identifier"],
        int,
    )
    assert isinstance(
        decision.tool_args["right_identifier"],
        int,
    )


def test_database_semantics_fixture_matches_known_v011_samples():
    repo = FakeSamples()
    left = make_tools(repo)._locate_sample(3811, CTX)
    right = make_tools(repo)._locate_sample(3809, CTX)

    assert left["sample"]["name"] == "trial_6"
    assert right["sample"]["name"] == "trial_4"
    assert left["sample"]["project_id"] == 115
    assert right["sample"]["project_id"] == 115
