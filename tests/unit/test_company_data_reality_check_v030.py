from pathlib import Path

from company_data import (
    CompanyDataRepository,
    resolve_company_data_runtime_root,
)
from runtime.company_data_inspection import (
    classify_company_data_request,
    inspect_company_data,
    render_inspection_answer,
    resolve_product_from_text,
)


ROOT = resolve_company_data_runtime_root()


def _classify(text):
    return classify_company_data_request(
        text, runtime_root=ROOT
    )


def test_routes_all_requested_reality_check_phrases():
    messages = [
        "真实样本多少？",
        "冲击强度缺失多少？",
        "哪些配方字段几乎全是空？",
        "哪些字段恒定不变？",
        "有没有重复样品？",
        "有没有异常值？",
        "有没有单位混乱？",
        "数据质量怎么样？",
        "哪些字段可以用于建模？",
        "为什么真实数据 Modeling Gate 不通过？",
        "有哪些性能指标？",
        "性能覆盖率怎么样？",
        "FR303 有多少样本？",
        "有多少产品类型？",
        "有哪些产品？",
        "冲击强度最大值是多少？",
        "工艺参数有多少？",
        "测试条件有多少？",
        "哪些字段缺失率高？",
    ]
    for message in messages:
        out = _classify(message)
        assert out["route"] is True, (message, out)


def test_negative_contexts_are_not_hijacked():
    messages = [
        "训练样本多少？",
        "测试集有没有异常值？",
        "模型样本量够吗？",
        "模拟数据有没有异常值？",
        "附件里的字段缺失多少？",
        "项目115有多少样品？",
        "3811样品缺失了什么？",
    ]
    for message in messages:
        out = _classify(message)
        assert out["route"] is False, (message, out)


def test_explicit_company_scope_overrides_training_word_ambiguity():
    out = _classify("公司真实数据里，训练可用样本有多少？")
    assert out["route"] is True
    assert out["explicit_company_scope"] is True


def test_fr303_short_alias_resolves_unique_product():
    repo = CompanyDataRepository(ROOT)
    product = resolve_product_from_text(
        repo, "FR303 有多少样本？"
    )
    assert product is not None
    assert product["product_type"] == "PC/ABS FR303"


def test_impact_strength_is_not_silently_merged():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="冲击强度缺失多少？",
        requested_checks=["target_missing"],
    )
    resolution = inspection["metric_resolution"]
    assert resolution["ambiguous"] is True
    assert resolution["merged"] is False
    names = {
        x["metric"] for x in resolution["matches"]
    }
    assert "悬臂梁冲击强度" in names
    assert "简支梁冲击强度" in names


def test_global_impact_missing_counts_are_real():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="冲击强度缺失多少？",
        requested_checks=["target_missing"],
    )
    by_metric = {
        x["metric"]: x
        for x in inspection["results"]["target_missing"]["metrics"]
    }
    assert by_metric["悬臂梁冲击强度"]["samples"] == 496
    assert by_metric["悬臂梁冲击强度"]["nonempty_count"] == 480
    assert by_metric["悬臂梁冲击强度"]["missing_count"] == 16
    assert by_metric["简支梁冲击强度"]["nonempty_count"] == 1
    assert by_metric["简支梁冲击强度"]["missing_count"] == 495


def test_sparse_fields_report_threshold_and_counts():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="哪些配方字段几乎全是空？",
        requested_checks=["formula_sparsity"],
    )
    item = inspection["results"]["formula_sparsity"]
    assert item["threshold_missing_rate"] == 0.80
    assert item["formula_fields_total"] == 473
    assert item["sparse_active_count"] >= 1
    assert inspection["boundaries"]["sparse_single_observation_is_not_called_constant"] is True


def test_constant_fields_use_minimum_support():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="哪些字段恒定不变？",
        requested_checks=["constant_fields"],
    )
    item = inspection["results"]["constant_fields"]
    assert item["minimum_support_for_constant"] >= 3
    assert "single_value_low_support_count" in item


def test_duplicates_distinguish_identity_from_same_formula():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="有没有重复样品？",
        requested_checks=["duplicates"],
    )
    item = inspection["results"]["duplicates"]
    assert "duplicate_sample_name_group_count" in item
    assert "duplicate_formula_group_count" in item
    assert "不等同于重复录入" in item["identity_note"]


def test_outliers_are_review_candidates_not_declared_errors():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="冲击强度有没有异常值？",
        requested_checks=["outliers"],
    )
    item = inspection["results"]["outliers"]
    assert item["scope"] == "REQUESTED_METRICS"
    assert item["metrics"]
    assert all(
        "warning" in metric for metric in item["metrics"]
    )
    assert inspection["boundaries"]["statistical_outlier_is_not_data_error"] is True


def test_unit_check_fails_closed_when_unit_metadata_missing():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="有没有单位混乱？",
        requested_checks=["units"],
    )
    item = inspection["results"]["units"]
    assert item["status"] == "NOT_AVAILABLE"
    assert item["can_assert_consistency"] is False
    assert item["conflict_count"] is None
    assert inspection["boundaries"]["unknown_units_are_not_reported_as_zero_conflicts"] is True


def test_modelability_never_overrides_gate():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="哪些字段可以用于建模？",
        requested_checks=["modelability"],
    )
    item = inspection["results"]["modelability"]
    assert item["true_process_parameter_rows"] == 0
    assert item["explicit_test_condition_rows"] == 0
    assert item["official_model_allowed_from_import_alone"] is False
    assert item["exploratory_formula_feature_count"] >= 1


def test_combined_question_runs_multiple_checks():
    repo = CompanyDataRepository(ROOT)
    message = (
        "真实样本多少？冲击强度缺失多少？"
        "有没有重复样品、异常值和单位混乱？"
    )
    decision = _classify(message)
    inspection = inspect_company_data(
        repo,
        message=message,
        requested_checks=decision["requested_checks"],
    )
    keys = inspection["results"].keys()
    assert "sample_count" in keys
    assert "target_missing" in keys
    assert "duplicates" in keys
    assert "outliers" in keys
    assert "units" in keys
    answer = render_inspection_answer(inspection)
    assert "496" in answer
    assert "不做合并" in answer
    assert "无法可靠判断" in answer


def test_product_distribution_is_available():
    repo = CompanyDataRepository(ROOT)
    decision = _classify("有哪些产品？")
    inspection = inspect_company_data(
        repo,
        message="有哪些产品？",
        requested_checks=decision["requested_checks"],
    )
    item = inspection["results"]["product_distribution"]
    assert item["product_type_count"] == 101
    assert item["products"][0]["product_type"] == "PC/ABS FR303"
    assert item["products"][0]["sample_count"] == 83


def test_metric_statistics_keep_impact_fields_separate():
    repo = CompanyDataRepository(ROOT)
    decision = _classify("冲击强度最大值是多少？")
    inspection = inspect_company_data(
        repo,
        message="冲击强度最大值是多少？",
        requested_checks=decision["requested_checks"],
    )
    item = inspection["results"]["metric_stats"]
    assert item["metric_resolution"]["ambiguous"] is True
    assert len(item["metrics"]) == 2
    assert all(x["metric"] for x in item["metrics"])


def test_process_and_condition_counts_are_explicit_zero_not_workflow_metadata():
    repo = CompanyDataRepository(ROOT)
    decision = _classify("工艺参数和测试条件有多少？")
    inspection = inspect_company_data(
        repo,
        message="工艺参数和测试条件有多少？",
        requested_checks=decision["requested_checks"],
    )
    item = inspection["results"]["process_conditions"]
    assert item["material_process_parameter_rows"] == 0
    assert item["explicit_test_condition_rows"] == 0
    assert item["workflow_metadata_rows"] == 496
    assert item["workflow_metadata_is_process_feature"] is False


def test_generic_field_missing_query_becomes_coverage_not_fake_target():
    decision = _classify("哪些字段缺失率高？")
    assert "target_missing" not in decision["requested_checks"]
    assert "field_coverage" in decision["requested_checks"]
    assert "formula_sparsity" in decision["requested_checks"]


def test_rendered_impact_missing_answer_uses_metric_name_not_python_dict():
    repo = CompanyDataRepository(ROOT)
    inspection = inspect_company_data(
        repo,
        message="冲击强度缺失多少？",
        requested_checks=["target_missing"],
    )
    answer = render_inspection_answer(inspection)
    assert "悬臂梁冲击强度：非空 480/496" in answer
    assert "{'metric':" not in answer
