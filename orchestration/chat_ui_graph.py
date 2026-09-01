from __future__ import annotations

from collections.abc import Callable
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
):
    """Compile the V3 production Chat UI workflow.

    V3 adds scoped checkpoints, explicit pause/resume and cached-response replay
    around the V2 conditional graph. It never automatically retries the
    production dispatcher, so database and LLM work are not duplicated.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "缺少 langgraph，请先执行 pip install -r requirements.txt"
        ) from exc

    def receive_request_node(
        state: ChatUIWorkflowState,
    ) -> dict[str, str]:
        if state.get("body") is None:
            raise ValueError("LangGraph Chat UI 缺少请求正文")
        if state.get("user_context") is None:
            raise ValueError("LangGraph Chat UI 缺少用户权限上下文")
        if state.get("container") is None:
            raise RuntimeError("LangGraph Chat UI 缺少应用依赖")
        update = {"workflow_version": "langgraph-chat-ui-v3"}
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
            router="langgraph_v3_control",
            reasoning_summary="LangGraph V3 安全暂停点：classify_primary。",
            routing={
                "version": "LANGGRAPH-V3",
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
            "version": "langgraph-chat-ui-v3",
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
    builder.add_node("semantic", dispatch_node)
    builder.add_node("resume_cached", preserve_response_node)
    builder.add_node("pause_after_primary", pause_after_primary_node)
    builder.add_node("classify_semantic", classify_semantic_node)
    for family in sorted(_SEMANTIC_FAMILIES):
        builder.add_node(f"semantic_{family}", preserve_response_node)
    builder.add_node("validate_response", validate_response_node)
    builder.add_edge(START, "receive_request")
    builder.add_edge("receive_request", "classify_primary")
    builder.add_conditional_edges(
        "classify_primary",
        route_primary,
        {
            "direct_attachment": "direct_attachment",
            "deterministic": "deterministic",
            "semantic": "semantic",
            "resume_cached": "resume_cached",
            "pause_after_primary": "pause_after_primary",
        },
    )
    builder.add_edge("direct_attachment", "validate_response")
    builder.add_edge("deterministic", "validate_response")
    builder.add_edge("resume_cached", "validate_response")
    builder.add_edge("pause_after_primary", "validate_response")
    builder.add_edge("semantic", "classify_semantic")
    builder.add_conditional_edges(
        "classify_semantic",
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
    """Run, checkpoint, pause or explicitly resume one V3 workflow."""

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
