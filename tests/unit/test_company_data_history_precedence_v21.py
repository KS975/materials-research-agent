from company_data import resolve_company_data_runtime_root
from runtime.company_data_inspection import classify_company_data_request
from runtime.company_data_conversation import breaks_company_product_context


ROOT = resolve_company_data_runtime_root()


def test_history_anomaly_question_is_not_hijacked_by_reality_check():
    out = classify_company_data_request(
        "历史上有没有和这个类似的冲击强度异常？",
        runtime_root=ROOT,
    )
    assert out["route"] is False
    assert "negative_context" in out["blocked_by"]


def test_plain_outlier_question_still_routes_to_company_reality_check():
    out = classify_company_data_request(
        "冲击强度有没有特别离谱的值？",
        runtime_root=ROOT,
    )
    assert out["route"] is True
    assert "outliers" in out["requested_checks"]


def test_explicit_company_real_data_can_still_override_history_marker():
    out = classify_company_data_request(
        "公司真实数据历史上有没有类似异常？",
        runtime_root=ROOT,
    )
    assert out["route"] is True
    assert out["explicit_company_scope"] is True


def test_history_question_breaks_inherited_company_product_context():
    assert breaks_company_product_context(
        "历史上有没有和这个类似的冲击强度异常？"
    ) is True
