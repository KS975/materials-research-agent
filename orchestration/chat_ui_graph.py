from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
import uuid

from orchestration.chat_ui_state import ChatUIWorkflowState
from schemas.chat_ui import ChatUIRequest, ChatUIResponse
from schemas.user_context import UserContext


ChatUIExecutor = Callable[
    [ChatUIRequest, UserContext, Any],
    ChatUIResponse,
]
ChatUIStateClassifier = Callable[
    [ChatUIWorkflowState],
    dict[str, Any],
]
ChatUISemanticExecutor = Callable[
    [ChatUIWorkflowState],
    ChatUIResponse,
]


_PRIMARY_FAMILIES = {
    "direct_attachment",
    "deterministic",
    "semantic",
    "resume_cached",
}

_SEMANTIC_FAMILIES = {
    "database_explorer",
    "rag",
    "current_attachment",
    "general_conversation",
    "material_tool",
}


def build_chat_ui_graph(
    executor: ChatUIExecutor,
    *,
    primary_classifier: ChatUIStateClassifier | None = None,
    semantic_classifier: ChatUIStateClassifier | None = None,
    semantic_planner: ChatUIStateClassifier | None = None,
    semantic_executors: Mapping[str, ChatUISemanticExecutor] | None = None,
):
    """Compile the V4 production Chat UI workflow.

    With ``semantic_planner`` and ``semantic_executors`` configured, V4 plans
    an unmatched semantic request once and then enters one native execution
    node. Omitting them keeps the V3 post-response classification path for
    backwards-compatible tests and staged rollout.

    Database/LLM business calls are never automatically retried by this graph.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "缺少 langgraph，请先执行 pip install -r requirements.txt"
        ) from exc

    native_semantic_enabled = (
        semantic_planner is not None or semantic_executors is not None
    )
    if native_semantic_enabled:
        if semantic_planner is None or semantic_executors is None:
            raise ValueError(
                "V4 原生语义路由必须同时配置 semantic_planner 和 semantic_executors"
            )
        missing = sorted(_SEMANTIC_FAMILIES - set(semantic_executors))
        unknown = sorted(set(semantic_executors) - _SEMANTIC_FAMILIES)
        if missing or unknown:
            raise ValueError(
                "V4 语义执行器映射不完整："
                f"missing={missing or '-'}, unknown={unknown or '-'}"
            )

    def receive_request_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, str]:
        if state.get("body") is None:
            raise ValueError("LangGraph Chat UI 缺少请求正文")
        if state.get("user_context") is None:
            raise ValueError("LangGraph Chat UI 缺少用户权限上下文")
        if state.get("container") is None:
            raise RuntimeError("LangGraph Chat UI 缺少应用依赖")
        update = {"workflow_version": "langgraph-chat-ui-v4"}
        _record_checkpoint(state, "request_received", update=update)
        return update

    def classify_primary_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, Any]:
        if state.get("resume_cached") and state.get("response") is not None:
            update = {
                "primary_family": "resume_cached",
                "deterministic_kind": "cached_response_replay",
            }
            _record_checkpoint(state, "cached_response_loaded", update=update)
            return update
        if primary_classifier is None:
            update = {"primary_family": "semantic"}
        else:
            update = dict(primary_classifier(state) or {})
        family = str(update.get("primary_family") or "").strip()
        if family not in _PRIMARY_FAMILIES:
            raise RuntimeError(f"LangGraph Chat UI 未知一级分支：{family or '-'}")
        _record_checkpoint(state, "primary_classified", update=update)
        return update

    def route_primary(
        state: ChatUIWorkflowState,
    ) -> str:
        family = str(state.get("primary_family") or "").strip()
        if family not in _PRIMARY_FAMILIES:
            raise RuntimeError(f"LangGraph Chat UI 一级分支缺失：{family or '-'}")
        if (
            state.get("pause_after") == "classify_primary"
            and not state.get("resuming")
            and family != "resume_cached"
        ):
            return "pause_after_primary"
        return family

    def dispatch_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, ChatUIResponse]:
        response = executor(
            state["body"],
            state["user_context"],
            state["container"],
        )
        _record_checkpoint(
            state,
            "dispatched",
            response=response,
        )
        return {"response": response}

    def pause_after_primary_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, Any]:
        response = ChatUIResponse(
            answer=(
                "工作流已在一级路由完成后暂停，尚未查询数据库或调用回答模型。"
                "使用相同问题、workflow_id，并设置 resume_workflow=true 可继续。"
            ),
            intent="workflow_paused",
            router="langgraph_v4_control",
            reasoning_summary="LangGraph V4 安全暂停点：classify_primary。",
            routing={
                "version": "LANGGRAPH-V4",
                "domain": "workflow_control",
                "primary_intent": "workflow_paused",
                "scope": {"company": "current", "projects": "authorized"},
                "constraints": {
                    "database_not_queried": True,
                    "answer_model_not_called": True,
                },
            },
        )
        update = {
            "response": response,
            "workflow_status": "PAUSED",
        }
        store = _workflow_store(state)
        if store is not None:
            store.pause(
                workflow_id=state["workflow_id"],
                ctx=state["user_context"],
                state={**state, **update},
            )
        return update

    def classify_semantic_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, Any]:
        if semantic_classifier is None:
            update = {"semantic_family": "material_tool"}
        else:
            update = dict(semantic_classifier(state) or {})
        family = str(update.get("semantic_family") or "").strip()
        if family not in _SEMANTIC_FAMILIES:
            raise RuntimeError(f"LangGraph Chat UI 未知语义分支：{family or '-'}")
        _record_checkpoint(state, "semantic_classified", update=update)
        return update

    def plan_semantic_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, Any]:
        if semantic_planner is None:
            raise RuntimeError("LangGraph V4 未配置语义规划器")
        update = dict(semantic_planner(state) or {})
        family = str(update.get("semantic_family") or "").strip()
        if family not in _SEMANTIC_FAMILIES:
            raise RuntimeError(f"LangGraph Chat UI 未知语义分支：{family or '-'}")
        _record_checkpoint(state, "semantic_planned", update=update)
        return update

    def route_semantic(
        state: ChatUIWorkflowState,
    ) -> str:
        family = str(state.get("semantic_family") or "").strip()
        if family not in _SEMANTIC_FAMILIES:
            raise RuntimeError(f"LangGraph Chat UI 语义分支缺失：{family or '-'}")
        return family

    def preserve_response_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, ChatUIResponse]:
        response = state.get("response")
        if response is None:
            raise RuntimeError("LangGraph Chat UI 分支没有执行结果")
        return {"response": response}

    def build_semantic_execution_node(family: str):
        def execute_semantic_node(
            state: ChatUIWorkflowState,
        ) -> dict[str, ChatUIResponse]:
            actual = str(state.get("semantic_family") or "").strip()
            if actual != family:
                raise RuntimeError(
                    "LangGraph V4 语义执行分支不一致："
                    f"planned={actual or '-'}, node={family}"
                )
            if semantic_executors is None:
                raise RuntimeError("LangGraph V4 未配置语义执行器")
            response = semantic_executors[family](state)
            _record_checkpoint(
                state,
                f"semantic_{family}_executed",
                response=response,
            )
            return {"response": response}

        return execute_semantic_node

    def validate_response_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, ChatUIResponse]:
        response = state.get("response")
        if isinstance(response, ChatUIResponse):
            validated = response
        elif isinstance(response, dict):
            validated = ChatUIResponse.model_validate(response)
        else:
            raise RuntimeError("LangGraph Chat UI 未生成有效响应")
        routing = dict(validated.routing or {})
        public_status = (
            "PAUSED"
            if state.get("workflow_status") == "PAUSED"
            else "SUCCEEDED"
        )
        routing["workflow"] = {
            "version": "langgraph-chat-ui-v4",
            "workflow_id": state.get("workflow_id"),
            "status": public_status,
            "resumed": bool(state.get("resuming")),
            "resume_count": int(state.get("resume_count") or 0),
            "cached_response_replayed": bool(state.get("resume_cached")),
        }
        validated.routing = routing
        return {"response": validated}

    def _workflow_store(state: ChatUIWorkflowState):
        return getattr(state.get("container"), "chat_ui_workflow_store", None)

    def _record_checkpoint(
        state: ChatUIWorkflowState,
        stage: str,
        *,
        update: dict[str, Any] | None = None,
        response: ChatUIResponse | None = None,
    ) -> None:
        store = _workflow_store(state)
        if store is None or not state.get("workflow_id"):
            return
        store.record_stage(
            workflow_id=state["workflow_id"],
            ctx=state["user_context"],
            stage=stage,
            state={**state, **(update or {})},
            response=response,
        )

    builder = StateGraph(ChatUIWorkflowState)
    builder.add_node("receive_request", receive_request_node)
    builder.add_node("classify_primary", classify_primary_node)
    builder.add_node("direct_attachment", dispatch_node)
    builder.add_node("deterministic", dispatch_node)
    builder.add_node("resume_cached", preserve_response_node)
    builder.add_node("pause_after_primary", pause_after_primary_node)
    if native_semantic_enabled:
        builder.add_node("plan_semantic", plan_semantic_node)
        for family in sorted(_SEMANTIC_FAMILIES):
            builder.add_node(
                f"semantic_{family}",
                build_semantic_execution_node(family),
            )
    else:
        builder.add_node("semantic", dispatch_node)
        builder.add_node("classify_semantic", classify_semantic_node)
        for family in sorted(_SEMANTIC_FAMILIES):
            builder.add_node(f"semantic_{family}", preserve_response_node)
    builder.add_node("validate_response", validate_response_node)
    builder.add_edge(START, "receive_request")
    builder.add_edge("receive_request", "classify_primary")
    primary_routes = {
        "direct_attachment": "direct_attachment",
        "deterministic": "deterministic",
        "semantic": "plan_semantic" if native_semantic_enabled else "semantic",
        "resume_cached": "resume_cached",
        "pause_after_primary": "pause_after_primary",
    }
    builder.add_conditional_edges(
        "classify_primary",
        route_primary,
        primary_routes,
    )
    builder.add_edge("direct_attachment", "validate_response")
    builder.add_edge("deterministic", "validate_response")
    builder.add_edge("resume_cached", "validate_response")
    builder.add_edge("pause_after_primary", "validate_response")
    semantic_router_node = (
        "plan_semantic" if native_semantic_enabled else "classify_semantic"
    )
    if not native_semantic_enabled:
        builder.add_edge("semantic", "classify_semantic")
    builder.add_conditional_edges(
        semantic_router_node,
        route_semantic,
        {
            family: f"semantic_{family}"
            for family in sorted(_SEMANTIC_FAMILIES)
        },
    )
    for family in sorted(_SEMANTIC_FAMILIES):
        builder.add_edge(f"semantic_{family}", "validate_response")
    builder.add_edge("validate_response", END)
    return builder.compile()


def invoke_chat_ui_graph(
    graph: Any,
    *,
    body: ChatUIRequest,
    user_context: UserContext,
    container: Any,
) -> ChatUIResponse:
    """Run, checkpoint, pause or explicitly resume one V4 workflow."""

    store = getattr(container, "chat_ui_workflow_store", None)
    if store is not None:
        session = store.begin(body=body, ctx=user_context)
    else:
        if body.resume_workflow:
            raise RuntimeError("当前环境没有配置 Chat UI 工作流检查点存储")
        session = {
            "workflow_id": body.workflow_id or str(uuid.uuid4()),
            "resuming": False,
            "cached_response": None,
            "resume_count": 0,
        }
    initial_state: ChatUIWorkflowState = {
        "body": body,
        "user_context": user_context,
        "container": container,
        "workflow_id": session["workflow_id"],
        "workflow_status": "RUNNING",
        "resuming": bool(session.get("resuming")),
        "resume_count": int(session.get("resume_count") or 0),
        "resume_cached": bool(session.get("cached_response")),
        "pause_after": body.pause_after,
    }
    if session.get("cached_response"):
        initial_state["response"] = ChatUIResponse.model_validate(
            session["cached_response"]
        )
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        if store is not None:
            try:
                store.fail(
                    workflow_id=session["workflow_id"],
                    ctx=user_context,
                    error=exc,
                )
            except Exception:
                pass
        raise
    response = final_state.get("response")
    if isinstance(response, dict):
        response = ChatUIResponse.model_validate(response)
    if not isinstance(response, ChatUIResponse):
        raise RuntimeError("LangGraph Chat UI 执行完成但没有响应")
    if store is not None:
        if final_state.get("workflow_status") == "PAUSED":
            pass
        else:
            store.finish(
                workflow_id=session["workflow_id"],
                ctx=user_context,
                state=final_state,
                response=response,
            )
    return response
