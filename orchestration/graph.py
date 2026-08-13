from __future__ import annotations

from agent.core import AgentCore
from agent.state import AgentState


def build_graph(core: AgentCore):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("缺少 langgraph，请先执行 pip install -r requirements.txt") from exc

    def route_node(state: AgentState):
        decision = core.route(state["message"])
        return {
            "intent": decision.intent,
            "tool_name": decision.tool_name,
            "tool_args": decision.tool_args,
        }

    def execute_node(state: AgentState):
        result = core.execute(
            state["intent"],
            state["tool_name"],
            state.get("tool_args", {}),
            state["user_context"],
        )
        evidence = result.get("evidence", []) if isinstance(result, dict) else []
        warnings = result.get("warnings", []) if isinstance(result, dict) else []
        return {
            "tool_result": result,
            "evidence": evidence,
            "warnings": warnings,
        }

    def answer_node(state: AgentState):
        answer = core.answer(
            state["message"],
            state["intent"],
            state.get("tool_result"),
        )
        return {"answer": answer}

    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("execute", execute_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "route")
    builder.add_edge("route", "execute")
    builder.add_edge("execute", "answer")
    builder.add_edge("answer", END)
    return builder.compile()
