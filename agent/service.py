from __future__ import annotations

from agent.core import AgentCore
from orchestration.graph import build_graph
from runtime.store import RuntimeStore
from schemas.user_context import UserContext


class MaterialsAgentService:
    def __init__(self, core: AgentCore, runtime_store: RuntimeStore):
        self.core = core
        self.runtime_store = runtime_store
        self.graph = build_graph(core)

    def chat(self, message: str, ctx: UserContext) -> dict:
        run_id = self.runtime_store.start_run(ctx.user_id, ctx.company_id, message)
        try:
            state = self.graph.invoke(
                {
                    "message": message,
                    "user_context": ctx,
                }
            )
            self.runtime_store.finish_run(
                run_id=run_id,
                status="SUCCEEDED",
                intent=state.get("intent"),
                tool_name=state.get("tool_name"),
                result=state.get("tool_result"),
                answer=state.get("answer"),
            )
            return state
        except Exception as exc:
            self.runtime_store.finish_run(
                run_id=run_id,
                status="FAILED",
                intent=None,
                tool_name=None,
                result=None,
                answer=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
