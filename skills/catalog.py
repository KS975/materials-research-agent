from __future__ import annotations

from agent.skill_registry import SkillRegistry, SkillSpec


def build_default_skill_registry() -> SkillRegistry:
    """Build the delivery architecture's eight business Skill contracts.

    Existing intent strings remain operation aliases for API compatibility.
    The registry is now the sole owner of operation-to-Skill and Tool access.
    """

    registry = SkillRegistry()
    for spec in (
        _knowledge_qa(),
        _data_governance(),
        _auto_ml(),
        _ensure_model(),
        _prediction(),
        _optimization(),
        _experiment_ingestion(),
        _model_governance(),
        _general_conversation(),
    ):
        registry.register(spec)
    return registry


def _knowledge_qa() -> SkillSpec:
    attachment = {"analyze_current_attachment", "ask_current_attachment"}
    rag = {
        "search_historical_knowledge",
        "historical_similar_case",
        "sample_historical_similarity",
        "joint_mysql_knowledge_analysis",
    }
    database_explorer = {"database_explorer"}
    intents = frozenset({
        "get_sample_context",
        "get_formula",
        "get_process",
        "get_performance",
        "find_samples",
        "compare_samples",
        "analyze_cause",
        "analyze_performance_difference",
        "sample_full_profile",
        "formula_difference",
        "process_difference",
        "comparability_check",
        "find_samples_multi_condition",
        "similar_samples",
        "company_real_data_status",
        *attachment,
        *rag,
        *database_explorer,
    })
    executor_by_intent = {
        **{intent: "current_attachment" for intent in attachment},
        **{intent: "rag" for intent in rag},
        **{intent: "database_explorer" for intent in database_explorer},
        "company_real_data_status": "deterministic",
    }
    return SkillSpec(
        name="knowledge_qa",
        display_name="Knowledge QA",
        description="结构化 MySQL 事实、当前附件和历史知识的受控问答与证据合成。",
        intents=intents,
        tool_allowlist=frozenset({
            "get_sample_context",
            "get_formula",
            "get_process",
            "get_performance",
            "find_samples",
            "compare_samples",
            "list_samples_for_analysis",
            "database_explorer",
            "company_real_data_runtime",
        }),
        allows_no_tool=True,
        workflow=(
            "route_evidence_source",
            "permission_scope",
            "retrieve_mysql_or_knowledge",
            "evidence_synthesis",
            "final_report",
        ),
        input_required_by_intent={
            "get_sample_context": ("identifier",),
            "get_formula": ("identifier",),
            "get_process": ("identifier",),
            "get_performance": ("identifier",),
            "sample_full_profile": ("identifier",),
            "compare_samples": ("left_identifier", "right_identifier"),
            "analyze_performance_difference": (
                "left_identifier",
                "right_identifier",
                "target_metric",
            ),
            "formula_difference": ("left_identifier", "right_identifier"),
            "process_difference": ("left_identifier", "right_identifier"),
            "comparability_check": ("left_identifier", "right_identifier"),
            "similar_samples": ("identifier",),
        },
        evidence_rules=(
            "数据库事实、RAG 证据与工程推断必须分型",
            "所有项目数据继承 UserContext 权限范围",
            "数值和单位不得由 LLM 补全",
        ),
        guardrails=(
            "业务 MySQL 只读",
            "附件与历史知识不得绕过 company_id",
            "Database Explorer 只访问授权虚拟数据源",
        ),
        error_strategy="fail_closed_or_request_clarification",
        executor_family_by_intent=executor_by_intent,
    )


def _data_governance() -> SkillSpec:
    return SkillSpec(
        name="data_governance",
        display_name="Data Governance",
        description="授权配方、工艺、性能数据的探查、统计、质量检查和实验系列结构分析。",
        intents=frozenset({
            "list_samples_for_analysis",
            "performance_rank",
            "performance_statistics",
            "experiment_series_analysis",
            "data_quality_check",
        }),
        tool_allowlist=frozenset({"list_samples_for_analysis"}),
        workflow=(
            "identify_data_scope",
            "profile_authorized_data",
            "quality_or_statistics",
            "evidence_summary",
        ),
        input_required_by_intent={
            "performance_rank": ("target_metric",),
            "performance_statistics": ("target_metric",),
            "experiment_series_analysis": ("keyword",),
        },
        evidence_rules=(
            "统计量只能由后端确定性计算",
            "排序和平均值必须先绑定唯一字段类别及真实字段名",
            "必须报告扫描范围、缺失值和单位一致性",
        ),
        guardrails=("不得把缺失值当作零", "不得跨单位聚合"),
        error_strategy="stop_on_schema_or_unit_ambiguity",
    )


def _auto_ml() -> SkillSpec:
    return SkillSpec(
        name="auto_ml",
        display_name="AutoML",
        description="Dataset 准入、训练、交叉验证、评价和候选模型注册。",
        intents=frozenset({"automl_training", "v013_modeling_status"}),
        tool_allowlist=frozenset({"automl_engine", "v013_runtime_reports"}),
        workflow=(
            "modeling_gate",
            "train_candidates",
            "cross_validation",
            "evaluate",
            "register_candidate",
        ),
        evidence_rules=("模型结果必须引用 dataset_version 和评估指标",),
        guardrails=("Modeling Gate FAIL 时禁止正式建模",),
        error_strategy="checkpoint_and_fail_closed",
        default_executor_family="deterministic",
    )


def _ensure_model() -> SkillSpec:
    return SkillSpec(
        name="ensure_model",
        display_name="Ensure Model",
        description="在预测或优化前选择可用模型，缺失时受控触发 AutoML。",
        intents=frozenset({"ensure_model"}),
        tool_allowlist=frozenset({"model_registry", "automl_engine"}),
        workflow=("query_model_registry", "validate_model", "trigger_automl_if_missing"),
        evidence_rules=("必须返回 model_id、model_version 和 dataset_version",),
        guardrails=("不得自动晋级模型",),
        error_strategy="request_approval_or_fail_closed",
        default_executor_family="deterministic",
    )


def _prediction() -> SkillSpec:
    return SkillSpec(
        name="prediction",
        display_name="Prediction",
        description="输入校验、适用域判断和模型推理。",
        intents=frozenset({"predict_performance"}),
        tool_allowlist=frozenset({"model_predictor", "applicability_domain"}),
        workflow=("ensure_model", "validate_input", "applicability_domain", "predict"),
        evidence_rules=("预测必须引用模型与数据集版本",),
        guardrails=("OUT_OF_DOMAIN 必须显式警告",),
        error_strategy="fail_closed_on_invalid_input",
        default_executor_family="deterministic",
    )


def _optimization() -> SkillSpec:
    return SkillSpec(
        name="optimization",
        display_name="Optimization",
        description="受约束的配方/工艺逆向设计、Pareto 排序与贝叶斯优化。",
        intents=frozenset({"v014_inverse_design", "v014_next_experiments"}),
        tool_allowlist=frozenset({"inverse_design_engine", "gaussian_process_bo"}),
        workflow=(
            "parse_constraints",
            "ensure_model",
            "generate_candidates",
            "predict_and_check_domain",
            "pareto_and_diversity",
            "final_report",
        ),
        evidence_rules=("候选必须引用模型版本、约束和适用域",),
        guardrails=("硬约束不得被 LLM 放宽", "不得把推荐写成实验事实"),
        error_strategy="checkpoint_and_return_near_misses",
        default_executor_family="deterministic",
    )


def _experiment_ingestion() -> SkillSpec:
    return SkillSpec(
        name="experiment_ingestion",
        display_name="Experiment Ingestion",
        description="实验结果识别、匹配、回流和自主实验运行态。",
        intents=frozenset({
            "v020_feedback_loop_status",
            "v020_submit_result",
            "v030_autonomy_status",
        }),
        tool_allowlist=frozenset({
            "v020_campaign_runtime",
            "v020_result_ingestion",
            "v030_autonomy_runtime",
        }),
        workflow=(
            "validate_experiment",
            "match_recommendation",
            "detect_duplicate_or_conflict",
            "append_dataset_version",
        ),
        evidence_rules=("回流必须保留 experiment_id 和关联记录",),
        guardrails=("实验回流不得默认强制重训",),
        error_strategy="checkpoint_and_require_resolution",
        default_executor_family="deterministic",
    )


def _model_governance() -> SkillSpec:
    return SkillSpec(
        name="model_governance",
        display_name="Model Governance",
        description="Challenger 比较、审批、模型晋级和废弃。",
        intents=frozenset({"model_promotion", "model_deprecation", "model_approval"}),
        tool_allowlist=frozenset({"model_registry", "model_evaluator"}),
        workflow=("compare_challenger", "approval", "update_model_status"),
        evidence_rules=("治理动作必须引用评估报告和模型版本",),
        guardrails=("模型晋级必须人工审批",),
        approval_points=("model_promotion", "model_deprecation"),
        error_strategy="pause_for_human_approval",
        default_executor_family="deterministic",
    )


def _general_conversation() -> SkillSpec:
    return SkillSpec(
        name="general_conversation",
        display_name="General Conversation",
        description="不使用企业数据证据的通用回答与澄清兜底。",
        intents=frozenset({
            "general_conversation",
            "unsupported_future_feature",
            "clarification_required",
        }),
        allows_no_tool=True,
        workflow=("clarify_or_answer", "declare_evidence_boundary"),
        evidence_rules=("不得声称本轮读取了数据库、附件或历史知识",),
        guardrails=("不得编造企业内部事实",),
        error_strategy="ask_one_clarifying_question",
        default_executor_family="general_conversation",
    )
