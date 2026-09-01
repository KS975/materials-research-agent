
from __future__ import annotations

from agent.tools import MaterialsTools
from schemas.user_context import UserContext


class FakeSamples:
    def __init__(self, *, by_name=None, by_id=None):
        self.by_name = by_name or {}
        self.by_id = by_id or {}
        self.calls = []

    def find_exact_name(self, name, ctx, limit=20):
        self.calls.append(("name", name))
        return list(self.by_name.get(name, []))

    def get_by_id(self, sample_id, ctx):
        self.calls.append(("id", sample_id))
        return self.by_id.get(sample_id)


class Dummy:
    pass


CTX = UserContext(
    user_id="test",
    company_id="company",
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


def test_numeric_string_prefers_exact_sample_name():
    sample = {
        "id": 99123,
        "name": "3811",
        "project_id": 115,
    }
    repo = FakeSamples(
        by_name={"3811": [sample]},
        by_id={3811: {"id": 3811, "name": "other"}},
    )

    result = make_tools(repo)._locate_sample("3811", CTX)

    assert result["status"] == "ok"
    assert result["sample"]["id"] == 99123
    assert result["sample"]["name"] == "3811"
    assert result["matched_by"] == "name"
    assert repo.calls == [("name", "3811")]


def test_numeric_string_falls_back_to_id_when_name_missing():
    sample = {
        "id": 3811,
        "name": "trial_3811",
        "project_id": 115,
    }
    repo = FakeSamples(
        by_name={},
        by_id={3811: sample},
    )

    result = make_tools(repo)._locate_sample("3811", CTX)

    assert result["status"] == "ok"
    assert result["sample"]["id"] == 3811
    assert result["matched_by"] == "id_fallback"
    assert repo.calls == [
        ("name", "3811"),
        ("id", 3811),
    ]


def test_explicit_int_means_database_id():
    sample = {
        "id": 3811,
        "name": "whatever",
        "project_id": 115,
    }
    repo = FakeSamples(
        by_name={
            "3811": [
                {
                    "id": 99999,
                    "name": "3811",
                }
            ]
        },
        by_id={3811: sample},
    )

    result = make_tools(repo)._locate_sample(3811, CTX)

    assert result["status"] == "ok"
    assert result["sample"]["id"] == 3811
    assert result["matched_by"] == "id"
    assert repo.calls == [("id", 3811)]


def test_duplicate_numeric_sample_names_remain_ambiguous():
    repo = FakeSamples(
        by_name={
            "3811": [
                {
                    "id": 101,
                    "name": "3811",
                    "project_id": 115,
                    "create_time": "a",
                },
                {
                    "id": 102,
                    "name": "3811",
                    "project_id": 115,
                    "create_time": "b",
                },
            ]
        }
    )

    result = make_tools(repo)._locate_sample("3811", CTX)

    assert result["status"] == "ambiguous"
    assert result["matched_by"] == "name"
    assert len(result["candidates"]) == 2


def test_nonnumeric_name_behavior_is_unchanged():
    sample = {
        "id": 7,
        "name": "trial_10",
        "project_id": 115,
    }
    repo = FakeSamples(by_name={"trial_10": [sample]})

    result = make_tools(repo)._locate_sample("trial_10", CTX)

    assert result["status"] == "ok"
    assert result["sample"]["name"] == "trial_10"
    assert result["matched_by"] == "name"
