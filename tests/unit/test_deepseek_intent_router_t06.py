from agent.deepseek_intent_router import DeepSeekIntentRouter


class FakeLLM:
    def __init__(self, output: str):
        self.output = output

    def complete(self, system: str, user: str) -> str:
        return self.output


def test_t06_allows_historical_knowledge_intent():
    llm = FakeLLM(
        '{"intent":"search_historical_knowledge","tool_name":null,'
        '"tool_args":{},"reasoning_summary":"检索项目历史知识"}'
    )
    decision = DeepSeekIntentRouter(llm).route(
        "历史有没有类似问题？",
        history=[],
        attachments=[],
    )

    assert decision.intent == "search_historical_knowledge"
    assert decision.tool_name is None
    assert decision.tool_args == {}


def test_t06_allows_explicit_project_id():
    llm = FakeLLM(
        '{"intent":"search_historical_knowledge","tool_name":null,'
        '"tool_args":{"project_id":115},"reasoning_summary":"检索项目115历史资料"}'
    )
    decision = DeepSeekIntentRouter(llm).route(
        "项目115历史上有没有类似问题？",
        history=[],
        attachments=[],
    )

    assert decision.intent == "search_historical_knowledge"
    assert decision.tool_args["project_id"] == 115
