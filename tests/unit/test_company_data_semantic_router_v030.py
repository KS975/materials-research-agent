from pathlib import Path

from company_data import (
    CompanyDataRepository,
    resolve_company_data_runtime_root,
)
from runtime.company_data_inspection import (
    classify_company_data_request,
    inspect_company_data,
    render_inspection_answer,
)


ROOT = resolve_company_data_runtime_root()


def c(text):
    return classify_company_data_request(text, runtime_root=ROOT)


def assert_routes_as(text, check):
    out = c(text)
    assert out["route"] is True, (text, out)
    assert check in out["requested_checks"], (text, out)
    assert out["confidence"] >= 0.45, (text, out)
    return out


def test_missing_paraphrases_are_semantic_not_phrase_exact():
    phrases = [
        "冲击强度缺失多少？",
        "现在的冲击强度缺失了多少？",
        "目前冲击强度还缺多少？",
        "当前的冲击强度缺了几条？",
        "冲击强度有多少是空的？",
        "冲击强度有多少没数据？",
        "冲击强度有效数据还有多少？",
        "冲击强度非空数据有多少？",
        "帮我看看冲击强度缺了多少",
        "请问现在冲击强度完整吗？",
        "冲击强度数据齐不齐？",
        "麻烦查一下冲击强度是不是不完整",
    ]
    for phrase in phrases:
        assert_routes_as(phrase, "target_missing")


def test_sample_count_paraphrases():
    phrases = [
        "真实样本多少？",
        "现在真实样本有多少？",
        "目前一共有多少真实样品？",
        "公司数据现在有几条样品？",
        "帮我看看真实样本量",
        "FR303现在有多少个样本？",
        "FR303目前样品量多少？",
    ]
    for phrase in phrases:
        assert_routes_as(phrase, "sample_count")


def test_product_paraphrases():
    for phrase in [
        "现在有多少种产品？",
        "目前产品类型有几个？",
        "都有哪些产品？",
        "帮我列一下现有产品",
        "产品分布怎么样？",
    ]:
        out = assert_routes_as(phrase, "product_distribution")
        if phrase == "帮我列一下现有产品":
            assert "field_inventory" not in out["requested_checks"]


def test_sparse_field_paraphrases():
    phrases = [
        "哪些配方字段几乎全是空？",
        "配方里哪些字段空得特别多？",
        "原料字段哪些缺失很严重？",
        "目前哪些配方字段缺得多？",
        "哪些字段的缺失率比较高？",
        "帮我看看高缺失的配方字段",
    ]
    for phrase in phrases:
        out = c(phrase)
        assert out["route"] is True, (phrase, out)
        assert (
            "formula_sparsity" in out["requested_checks"]
            or "field_coverage" in out["requested_checks"]
        ), (phrase, out)


def test_constant_paraphrases():
    for phrase in [
        "哪些字段恒定不变？",
        "哪些配方变量一直没变化？",
        "有没有全都一样的配方字段？",
        "哪些特征是零方差？",
        "帮我看看常量字段",
    ]:
        assert_routes_as(phrase, "constant_fields")


def test_duplicate_paraphrases():
    for phrase in [
        "有没有重复样品？",
        "现在有没有重复录入的数据？",
        "配方有没有重复的？",
        "哪些样本重复了？",
        "帮我查一下重复记录",
    ]:
        assert_routes_as(phrase, "duplicates")


def test_outlier_paraphrases():
    for phrase in [
        "有没有异常值？",
        "数据里有没有离群点？",
        "冲击强度有没有特别离谱的值？",
        "帮我看看极端值",
        "现在的数据异常多不多？",
    ]:
        assert_routes_as(phrase, "outliers")


def test_unit_paraphrases():
    for phrase in [
        "有没有单位混乱？",
        "单位是不是不一致？",
        "现在单位统一吗？",
        "量纲有没有问题？",
        "不同性能的单位会不会冲突？",
    ]:
        assert_routes_as(phrase, "units")


def test_metric_stats_paraphrases():
    for phrase in [
        "冲击强度最大值是多少？",
        "现在冲击强度最高有多少？",
        "冲击强度最低是多少？",
        "冲击强度平均大概多少？",
        "冲击强度的中位数呢？",
        "冲击强度取值范围怎么样？",
    ]:
        assert_routes_as(phrase, "metric_stats")


def test_process_condition_paraphrases():
    for phrase in [
        "工艺参数有多少？",
        "现在有没有工艺数据？",
        "测试条件齐不齐？",
        "显式测试条件缺多少？",
        "目前工艺参数完整吗？",
    ]:
        assert_routes_as(phrase, "process_conditions")


def test_modelability_paraphrases():
    for phrase in [
        "哪些字段可以用于建模？",
        "现在这些数据能不能建模？",
        "为什么 Modeling Gate 还过不了？",
        "哪些配方特征适合拿去训练模型？",
        "真实数据现在够不够做模型？",
    ]:
        out = c(phrase)
        assert out["route"] is True, (phrase, out)
        assert (
            "modelability" in out["requested_checks"]
            or "quality_overview" in out["requested_checks"]
        ), (phrase, out)


def test_quality_overview_paraphrases():
    for phrase in [
        "数据质量怎么样？",
        "现在这批数据质量如何？",
        "帮我做一下数据体检",
        "Reality Check",
        "整体数据情况怎么样？",
    ]:
        assert_routes_as(phrase, "quality_overview")


def test_negative_contexts_still_win():
    phrases = [
        "训练集的冲击强度缺失了多少？",
        "测试集现在有多少异常值？",
        "模拟数据的冲击强度缺多少？",
        "Simulator 里有没有重复样品？",
        "附件里的冲击强度现在缺失多少？",
        "这份PDF里哪些字段是空的？",
        "3811样品的冲击强度缺了吗？",
        "项目115现在有多少样品？",
    ]
    for phrase in phrases:
        out = c(phrase)
        assert out["route"] is False, (phrase, out)
        assert out["blocked_by"], (phrase, out)


def test_explicit_real_company_scope_can_override_training_word():
    out = c("公司真实数据里现在可用于训练的样本有多少？")
    assert out["route"] is True
    assert out["explicit_company_scope"] is True


def test_diagnostics_explain_semantic_match():
    out = c("现在的冲击强度缺失了多少？")
    assert out["route"] is True
    assert out["reason"] == "DATA_QUALITY_QUERY"
    assert out["matched_semantics"]["missing"] is True
    assert out["matched_semantics"]["quantity"] is True
    assert out["matched_semantics"]["performance_subject"] is True
    assert "现在" not in out["normalized_message"]


def test_actual_question_from_user_returns_real_counts():
    repo = CompanyDataRepository(ROOT)
    out = c("现在的冲击强度缺失了多少？")
    inspection = inspect_company_data(
        repo,
        message="现在的冲击强度缺失了多少？",
        requested_checks=out["requested_checks"],
    )
    answer = render_inspection_answer(inspection)
    assert "悬臂梁冲击强度：非空 480/496" in answer
    assert "缺失 16（3.2%）" in answer
    assert "简支梁冲击强度：非空 1/496" in answer


def test_metric_alias_still_not_merged():
    repo = CompanyDataRepository(ROOT)
    out = c("目前冲击强度还缺多少？")
    inspection = inspect_company_data(
        repo,
        message="目前冲击强度还缺多少？",
        requested_checks=out["requested_checks"],
    )
    resolution = inspection["metric_resolution"]
    assert resolution["ambiguous"] is True
    assert resolution["merged"] is False
    assert {x["metric"] for x in resolution["matches"]} == {
        "悬臂梁冲击强度",
        "简支梁冲击强度",
    }
