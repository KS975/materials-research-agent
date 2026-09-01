from company_data import (
    resolve_company_data_runtime_root,
)
from runtime.company_data_conversation import (
    classify_company_data_turn,
    resolve_company_conversation_scope,
)
from runtime.company_data_ui import (
    build_company_data_overview,
)


ROOT = resolve_company_data_runtime_root()


def history(*user_messages):
    out = []
    for message in user_messages:
        out.append({
            "role": "user",
            "content": message,
        })
        out.append({
            "role": "assistant",
            "content": "ok",
        })
    return out


def test_fr303_scope_is_inherited_to_anomaly_followup():
    scope = resolve_company_conversation_scope(
        ROOT,
        message="有没有异常值？",
        history=history("FR303 有多少样本？"),
    )
    assert scope["product_type"] == "PC/ABS FR303"
    assert scope["source"] == "HISTORY"
    assert scope["inherited"] is True


def test_fr303_scope_is_inherited_to_modelability_followup():
    decision = classify_company_data_turn(
        ROOT,
        message="哪些字段可以用于建模？",
        history=history("FR303 有多少样本？"),
    )
    assert decision["route"] is True
    assert (
        decision["conversation_scope"]["product_type"]
        == "PC/ABS FR303"
    )
    assert "modelability" in decision["requested_checks"]


def test_abbreviated_followup_uses_product_context():
    decision = classify_company_data_turn(
        ROOT,
        message="那缺失呢？",
        history=history("FR303 有多少样本？"),
    )
    assert decision["route"] is True
    assert decision["contextualized"] is True
    assert (
        decision["conversation_scope"]["product_type"]
        == "PC/ABS FR303"
    )


def test_global_reset_stops_inheritance():
    scope = resolve_company_conversation_scope(
        ROOT,
        message="看一下全库有没有异常值",
        history=history("FR303 有多少样本？"),
    )
    assert scope["product_type"] is None
    assert scope["source"] == "GLOBAL_RESET"


def test_topic_break_stops_old_product_inheritance():
    scope = resolve_company_conversation_scope(
        ROOT,
        message="有没有异常值？",
        history=history(
            "FR303 有多少样本？",
            "Project 9036 查看 V0.3 自主实验状态",
        ),
    )
    assert scope["product_type"] is None


def test_fr303_sample_answer_is_answer_first():
    report = build_company_data_overview(
        ROOT,
        message="FR303 有多少样本？",
        product_name="PC/ABS FR303",
        conversation_scope={
            "product_type": "PC/ABS FR303",
            "source": "CURRENT_MESSAGE",
            "inherited": False,
        },
    )
    assert report["answer"].startswith(
        "PC/ABS FR303 共有 83 个真实样品。"
    )
    assert (
        "悬臂梁冲击强度有 82 条有效记录"
        in report["answer"]
    )
    assert (
        report["presentation"]["card_type"]
        == "sample_count"
    )


def test_inherited_fr303_anomaly_answer_stays_on_83_samples():
    decision = classify_company_data_turn(
        ROOT,
        message="有没有异常值？",
        history=history("FR303 有多少样本？"),
    )
    report = build_company_data_overview(
        ROOT,
        message="有没有异常值？",
        product_name="PC/ABS FR303",
        classification_override=decision,
        conversation_scope=(
            decision["conversation_scope"]
        ),
    )
    assert (
        report["selected_product"]["sample_count"]
        == 83
    )
    assert (
        report["presentation"]["scope_inherited"]
        is True
    )
    assert (
        report["presentation"]["card_type"]
        == "quality"
    )
    assert (
        "PC/ABS FR303 有统计异常候选"
        in report["answer"]
    )
    assert "12 个可计算性能指标" in report["answer"]
    assert "6 个指标出现异常候选" in report["answer"]


def test_inherited_fr303_modelability_uses_product_not_global():
    decision = classify_company_data_turn(
        ROOT,
        message="哪些字段可以用于建模？",
        history=history("FR303 有多少样本？"),
    )
    report = build_company_data_overview(
        ROOT,
        message="哪些字段可以用于建模？",
        product_name="PC/ABS FR303",
        classification_override=decision,
        conversation_scope=(
            decision["conversation_scope"]
        ),
    )
    p = report["presentation"]
    assert p["card_type"] == "modeling_readiness"
    assert p["scope_label"] == "PC/ABS FR303"
    values = {
        x["label"]: x["value"]
        for x in p["metrics"]
    }
    assert values["真实样品"] == 83
    assert values["活跃配方字段"] == 103
    assert values["探索候选特征"] == 5
    assert values["工艺参数行"] == 0
    assert values["测试条件行"] == 0
    assert "可以做探索性建模" in report["answer"]


def test_fr303_impact_missing_is_product_scoped():
    decision = classify_company_data_turn(
        ROOT,
        message="冲击强度缺失多少？",
        history=history("FR303 有多少样本？"),
    )
    report = build_company_data_overview(
        ROOT,
        message="冲击强度缺失多少？",
        product_name="PC/ABS FR303",
        classification_override=decision,
        conversation_scope=(
            decision["conversation_scope"]
        ),
    )
    assert (
        "悬臂梁冲击强度：有效 82/83，缺失 1"
        in report["answer"]
    )
    assert (
        "简支梁冲击强度：有效 0/83，缺失 83"
        in report["answer"]
    )


def test_product_card_keeps_global_data_collapsed():
    report = build_company_data_overview(
        ROOT,
        message="FR303 有多少样本？",
        product_name="PC/ABS FR303",
        conversation_scope={
            "product_type": "PC/ABS FR303",
            "source": "CURRENT_MESSAGE",
            "inherited": False,
        },
    )
    assert (
        report["presentation"]["show_global_details"]
        is True
    )
    assert len(report["presentation"]["metrics"]) <= 5
