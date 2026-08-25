from agent.deepseek_intent_router import DeepSeekIntentRouter


class FakeLLM:
    def __init__(self, output: str):
        self.output = output

    def complete(self, system: str, user: str) -> str:
        return self.output


def test_t07_router_allows_joint_intent():
    llm = FakeLLM(
        '{"intent":"joint_mysql_knowledge_analysis","tool_name":null,'
        '"tool_args":{"left_identifier":3811,"right_identifier":3809,'
        '"target_metric":"冲击强度","direction_claim":"更低","project_id":115},'
        '"reasoning_summary":"同时读取数据库事实和历史资料"}'
    )

    decision = DeepSeekIntentRouter(llm).route(
        "3811 的冲击强度比 3809 低很多，历史上有没有类似问题？结合数据库和历史报告分析一下。",
        history=[],
        attachments=[],
    )

    assert decision.intent == "joint_mysql_knowledge_analysis"
    assert decision.tool_name is None
    assert decision.tool_args["left_identifier"] == 3811
    assert decision.tool_args["right_identifier"] == 3809
    assert decision.tool_args["target_metric"] == "冲击强度"
    # The user did not explicitly name Project 115 in this turn. Router V2
    # removes the ungrounded project id so downstream RAG searches the
    # caller's authorized company-wide history scope.
    assert "project_id" not in decision.tool_args
