from agent.router import RuleIntentRouter


def test_route_sample_context():
    d = RuleIntentRouter().route("查 trial_10 的完整研发上下文")
    assert d.tool_name == "get_sample_context"
    assert d.tool_args["identifier"] == "trial_10"


def test_route_compare():
    d = RuleIntentRouter().route("比较 A001 和 A002")
    assert d.tool_name == "compare_samples"
    assert d.tool_args == {
        "left_identifier": "A001",
        "right_identifier": "A002",
    }


def test_route_why_first_gathers_context():
    d = RuleIntentRouter().route("为什么 trial_10 密度差下降？")
    assert d.intent == "analyze_cause"
    assert d.tool_name == "get_sample_context"
    assert d.tool_args["identifier"] == "trial_10"
