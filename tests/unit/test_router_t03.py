from agent.router import RuleIntentRouter


def test_route_t03_performance_difference():
    d = RuleIntentRouter().route("为什么 3811 的冲击强度比 3809 低？")
    assert d is not None
    assert d.intent == "analyze_performance_difference"
    assert d.tool_name == "compare_samples"
    assert d.tool_args == {
        "left_identifier": "3811",
        "right_identifier": "3809",
        "target_metric": "冲击强度",
        "direction": "低",
    }
