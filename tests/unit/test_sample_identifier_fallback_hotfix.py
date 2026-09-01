from __future__ import annotations

from agent.router import RuleIntentRouter
from agent.tools import MaterialsTools


class _Samples:
    def __init__(self):
        self.ids = []
        self.names = []

    def get_by_id(self, sample_id, ctx):
        self.ids.append(sample_id)
        if sample_id == 3811:
            return {"id": 3811, "name": "trial_6", "project_id": 115}
        return None

    def find_exact_name(self, name, ctx):
        self.names.append(name)
        if name == "ABS-051":
            return [{"id": 3073, "name": "ABS-051", "project_id": -1606}]
        return []


class _Noop:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _tools(samples):
    return MaterialsTools(samples, _Noop(), _Noop(), _Noop(), _Noop())


def test_rule_fallback_extracts_numeric_id_from_natural_language_wrapper():
    d = RuleIntentRouter().route("查看样品3811具体信息")
    assert d is not None
    assert d.tool_args == {"identifier": "3811"}


def test_rule_fallback_keeps_named_sample_token():
    d = RuleIntentRouter().route("查看ABS-051具体信息")
    assert d is not None
    assert d.tool_args == {"identifier": "ABS-051"}


def test_db_boundary_unwraps_single_numeric_id_from_phrase():
    samples = _Samples()
    result = _tools(samples)._locate_sample("看样品3811具体信息", object())
    assert result["status"] == "ok"
    assert result["sample"]["id"] == 3811
    assert samples.ids == [3811]
    assert samples.names == []


def test_db_boundary_does_not_break_named_sample():
    samples = _Samples()
    result = _tools(samples)._locate_sample("ABS-051", object())
    assert result["status"] == "ok"
    assert result["sample"]["id"] == 3073
    assert samples.names == ["ABS-051"]
