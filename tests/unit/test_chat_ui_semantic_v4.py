from __future__ import annotations

from types import SimpleNamespace

from api import chat_ui as chat_ui_module
from agent.scenario_composer import ScenarioWorkflowComposer
from skills.catalog import build_default_skill_registry
from schemas.chat_ui import ChatUIRequest
from schemas.user_context import UserContext


def _ctx() -> UserContext:
    return UserContext(
        user_id="semantic-v4",
        company_id="company-v4",
        project_ids=(115,),
        permission_source="test",
    )


def _base_state(*, intent: str, container, **extra):
    state = {
        "body": ChatUIRequest(message="v4 semantic test"),
        "user_context": _ctx(),
        "container": container,
        "intent": intent,
        "tool_name": None,
        "tool_args": {},
        "router_name": "deepseek",
        "reasoning_summary": "safe plan summary",
        "routing_meta": {
            "scope": {"company": "current", "projects": "authorized"},
        },
        "history": [],
        "database_explorer_enabled": False,
        "needs_clarification": False,
    }
    state.update(extra)
    skill_registry = build_default_skill_registry()
    if not hasattr(container, "skill_registry"):
        container.skill_registry = skill_registry
    if not hasattr(container, "scenario_composer"):
        container.scenario_composer = ScenarioWorkflowComposer(skill_registry)
    if not extra.get("skill_name"):
        state["skill_name"] = skill_registry.resolve(intent).name
    return state


def test_v4_production_planner_calls_deepseek_router_once(monkeypatch):
    calls = []

    class Decision:
        intent = "general_conversation"
        tool_name = None
        tool_args = {}
        reasoning_summary = "general answer"
        needs_clarification = False
        clarification_question = ""

        def to_routing_meta(self):
            return {
                "primary_intent": self.intent,
                "scope": {"company": "current", "projects": "authorized"},
            }

    class Router:
        def __init__(self, llm):
            self.llm = llm

        def route(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Decision()

    monkeypatch.setattr(chat_ui_module, "DeepSeekIntentRouter", Router)
    monkeypatch.setattr(
        chat_ui_module,
        "looks_like_multi_condition_request",
        lambda message: False,
    )
    container = SimpleNamespace(
        settings=SimpleNamespace(llm_enabled=True),
        llm=object(),
        database_explorer_skill=SimpleNamespace(enabled=False, mode="off"),
        core=SimpleNamespace(rule_router=SimpleNamespace(route=lambda message: None)),
    )
    skill_registry = build_default_skill_registry()
    container.skill_registry = skill_registry
    container.scenario_composer = ScenarioWorkflowComposer(skill_registry)
    plan = chat_ui_module._plan_chat_ui_semantic({
        "body": ChatUIRequest(message="为什么尼龙吸水？"),
        "user_context": _ctx(),
        "container": container,
    })

    assert len(calls) == 1
    assert plan["semantic_family"] == "general_conversation"
    assert plan["intent"] == "general_conversation"
    assert plan["routing_meta"]["scope"]["company"] == "current"


def test_v4_native_general_conversation_executor():
    container = SimpleNamespace(
        general_conversation_skill=SimpleNamespace(
            answer=lambda **kwargs: {"status": "ok", "answer": "general-ok", "warnings": []}
        )
    )
    response = chat_ui_module._execute_semantic_general_conversation(
        _base_state(intent="general_conversation", container=container)
    )
    assert response.answer == "general-ok"
    assert response.router == "deepseek_general_answer"


def test_v4_native_database_explorer_executor_preserves_permission_scope():
    received = {}

    def answer(**kwargs):
        received.update(kwargs)
        return {"status": "ok", "answer": "database-ok", "evidence": [], "warnings": []}

    skill = SimpleNamespace(enabled=True, answer=answer)
    container = SimpleNamespace(database_explorer_skill=skill)
    response = chat_ui_module._execute_semantic_database_explorer(
        _base_state(
            intent="database_explorer",
            container=container,
            database_explorer_enabled=True,
        )
    )
    assert response.answer == "database-ok"
    assert received["ctx"].company_id == "company-v4"


def test_v4_native_current_attachment_executor():
    container = SimpleNamespace(
        current_attachment_skill=SimpleNamespace(
            answer=lambda **kwargs: {
                "status": "ok",
                "answer": "attachment-ok",
                "evidence": [],
                "warnings": [],
            }
        )
    )
    response = chat_ui_module._execute_semantic_current_attachment(
        _base_state(
            intent="ask_current_attachment",
            container=container,
            body=ChatUIRequest(message="读取附件", attachment_ids=["attachment-v4"]),
        )
    )
    assert response.answer == "attachment-ok"


def test_v4_native_historical_rag_executor():
    container = SimpleNamespace(
        historical_knowledge_skill=SimpleNamespace(
            answer=lambda **kwargs: {
                "status": "ok",
                "answer": "rag-ok",
                "evidence": [],
                "warnings": [],
            }
        )
    )
    response = chat_ui_module._execute_semantic_rag(
        _base_state(
            intent="search_historical_knowledge",
            container=container,
            tool_args={"project_id": 115},
        )
    )
    assert response.answer == "rag-ok"
    assert response.tool_args["project_id"] == 115


def test_v4_native_material_tool_executor():
    core = SimpleNamespace(
        execute=lambda intent, tool_name, tool_args, ctx: {
            "value": 1,
            "evidence": [],
            "warnings": [],
        },
        answer=lambda message, intent, result: "tool-ok",
    )
    response = chat_ui_module._execute_semantic_material_tool(
        _base_state(
            intent="get_sample_context",
            container=SimpleNamespace(core=core),
            tool_name="get_sample_context",
            tool_args={"identifier":"3811"},
        )
    )
    assert response.answer == "tool-ok"
    assert response.tool_name == "get_sample_context"


def test_v4_async_engine_intent_creates_task_instead_of_sync_execution():
    submitted = {}

    class TaskManager:
        def submit(self, *, ctx, request):
            submitted["request"] = request
            return {
                "task_id": "task-async-1",
                "status": "QUEUED",
                "intent": request.intent,
            }

    container = SimpleNamespace(
        settings=SimpleNamespace(engine_task_mode="async"),
        engine_task_manager=TaskManager(),
    )
    response = chat_ui_module._execute_semantic_engine_workflow(
        _base_state(
            intent="predict_performance",
            container=container,
            tool_name="predict_model",
            tool_args={"project_id": 115, "target_metric": "冲击强度"},
        )
    )

    assert response.data["kind"] == "engine_task_created"
    assert response.data["task"]["task_id"] == "task-async-1"
    assert response.router == "engine_task_async"
    assert submitted["request"].tool_name == "predict_model"


def test_v4_shadow_optimization_compares_legacy_and_engine(monkeypatch):
    def legacy_runtime(*, runtime_root, project_id, message):
        return {
            "status": "SUCCESS",
            "generation": {"generated_count": 10},
            "counts": {
                "generated_hard_valid": 8,
                "qualified_all_targets": 2,
            },
            "design_cards": [
                {"candidate_id": "l1", "values": {"x": 1}, "applicability_domain": {"status": "IN_DOMAIN"}},
                {"candidate_id": "l2", "values": {"x": 2}, "applicability_domain": {"status": "IN_DOMAIN"}},
            ],
        }

    monkeypatch.setattr(chat_ui_module, "run_inverse_design_for_ui", legacy_runtime)
    engine_result = {
        "status": "OK",
        "result": {
            "status": "COMPLETE",
            "selected_candidates": [
                {"candidate_id": "e1", "values": {"x": 3}, "objective_errors": {"impact": 0}, "applicability_domain": "IN_DOMAIN"},
                {"candidate_id": "e2", "values": {"x": 4}, "objective_errors": {"impact": 0}, "applicability_domain": "IN_DOMAIN"},
            ],
            "diagnostics": {"generated_count": 10, "hard_feasible_count": 8, "elapsed_ms": 100},
        },
    }
    shadow = chat_ui_module._run_legacy_optimization_for_shadow(
        ChatUIRequest(message="冲击强度 >= 35，推荐2组"),
        _ctx(),
        "optimize_formula",
        {"project_id": 115},
        engine_result,
    )
    assert shadow["decision"] == "READY"
    assert shadow["rollback_route"] == "legacy"
