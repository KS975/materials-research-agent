from __future__ import annotations

from pathlib import Path
from typing import Any

from company_data import (
    CompanyDataRepository,
    CompanyDataValidationError,
)
from runtime.company_data_inspection import (
    classify_company_data_request,
    inspect_company_data,
    resolve_product_from_text,
)


class CompanyDataUIError(RuntimeError):
    pass


def _coverage_for_metric(
    selected: dict[str, Any] | None,
    metric_name: str,
) -> dict[str, Any] | None:
    if not selected:
        return None
    for item in selected.get("performance_coverage") or []:
        if item.get("metric") == metric_name:
            return item
    return None


def _presentation_for(
    *,
    summary: dict[str, Any],
    selected: dict[str, Any] | None,
    inspection: dict[str, Any] | None,
    conversation_scope: dict[str, Any] | None,
) -> dict[str, Any]:
    checks = list(
        (inspection or {}).get("requested_checks") or []
    )
    results = (inspection or {}).get("results") or {}
    product = (
        selected.get("product_type")
        if selected else None
    )
    scope_label = product or "全部真实数据"
    inherited = bool(
        (conversation_scope or {}).get("inherited")
    )

    base: dict[str, Any] = {
        "scope_label": scope_label,
        "scope_kind": "PRODUCT" if product else "GLOBAL",
        "scope_inherited": inherited,
        "scope_source": (
            conversation_scope or {}
        ).get("source"),
        "card_type": "overview",
        "status": "IMPORTED",
        "eyebrow": "REAL DATA",
        "headline": "",
        "metrics": [],
        "highlights": [],
        "details": [],
        "show_global_details": bool(product),
    }

    if "sample_count" in checks:
        count = (
            selected.get("sample_count")
            if selected else summary.get("samples")
        )
        base.update({
            "card_type": "sample_count",
            "status": "PASS",
            "eyebrow": "SAMPLE SCOPE",
            "headline": (
                f"{product} 共有 {count} 个真实样品"
                if product
                else f"当前共有 {count} 个真实样品"
            ),
            "metrics": (
                [
                    {
                        "label": "真实样品",
                        "value": count,
                    },
                    {
                        "label": "有配方",
                        "value": selected.get(
                            "formula_present_samples"
                        ),
                    },
                    {
                        "label": "有性能",
                        "value": selected.get(
                            "performance_present_samples"
                        ),
                    },
                    {
                        "label": "本地项目",
                        "value": selected.get(
                            "local_project_id"
                        ),
                    },
                ]
                if selected
                else [
                    {
                        "label": "真实样品",
                        "value": summary.get("samples"),
                    },
                    {
                        "label": "产品类型",
                        "value": summary.get("products"),
                    },
                    {
                        "label": "原料字段",
                        "value": summary.get("materials"),
                    },
                    {
                        "label": "性能指标",
                        "value": summary.get(
                            "performance_metrics"
                        ),
                    },
                ]
            ),
        })
        impact = _coverage_for_metric(
            selected,
            "悬臂梁冲击强度",
        )
        if impact:
            base["highlights"].append(
                "悬臂梁冲击强度有 "
                f"{impact.get('nonempty_count', 0)} "
                "条有效记录"
            )
        return base

    if "outliers" in checks:
        item = results.get("outliers") or {}
        if (
            item.get("scope")
            == "ALL_NUMERIC_PERFORMANCE_METRICS"
        ):
            top = [
                x
                for x in (item.get("top_metrics") or [])
                if int(x.get("outlier_count") or 0) > 0
            ]
            candidate_count = sum(
                int(x.get("outlier_count") or 0)
                for x in top
            )
            metrics_with = int(
                item.get("metrics_with_outliers") or 0
            )
            checked = int(
                item.get("metrics_checked") or 0
            )
            base.update({
                "card_type": "quality",
                "status": (
                    "REVIEW_REQUIRED"
                    if metrics_with else "PASS"
                ),
                "eyebrow": "DATA QUALITY",
                "headline": (
                    f"{scope_label} 有统计异常候选，需要复核"
                    if metrics_with
                    else f"{scope_label} 未发现统计异常候选"
                ),
                "metrics": [
                    {
                        "label": "可计算指标",
                        "value": checked,
                    },
                    {
                        "label": "涉及异常指标",
                        "value": metrics_with,
                    },
                    {
                        "label": "异常候选点",
                        "value": candidate_count,
                    },
                    {
                        "label": "样品范围",
                        "value": (
                            selected.get("sample_count")
                            if selected
                            else summary.get("samples")
                        ),
                    },
                ],
                "details": [
                    {
                        "label": x.get("metric"),
                        "value": x.get("outlier_count"),
                        "sub": (
                            f"n={x.get('numeric_count')}"
                        ),
                    }
                    for x in top[:5]
                ],
                "highlights": [
                    "统计异常只代表建议复核，"
                    "不会自动删除或判为错误数据。"
                ],
            })
        return base

    if "modelability" in checks:
        model = results.get("modelability") or {}
        inventory = (
            results.get("field_inventory") or {}
        )
        process_rows = int(
            model.get("true_process_parameter_rows")
            or 0
        )
        condition_rows = int(
            model.get("explicit_test_condition_rows")
            or 0
        )
        exploratory = int(
            model.get(
                "exploratory_formula_feature_count"
            )
            or 0
        )
        base.update({
            "card_type": "modeling_readiness",
            "status": "BLOCKED",
            "eyebrow": "MODELING READINESS",
            "headline": (
                f"{scope_label} 可做探索性建模，"
                "但正式模型暂不可用"
            ),
            "metrics": [
                {
                    "label": "真实样品",
                    "value": (
                        selected.get("sample_count")
                        if selected
                        else summary.get("samples")
                    ),
                },
                {
                    "label": "活跃配方字段",
                    "value": inventory.get(
                        "formula_fields_active"
                    ),
                },
                {
                    "label": "探索候选特征",
                    "value": exploratory,
                },
                {
                    "label": "工艺参数行",
                    "value": process_rows,
                },
                {
                    "label": "测试条件行",
                    "value": condition_rows,
                },
            ],
            "highlights": [
                "探索性预测 / 留出验证：ALLOWED",
                "正式模型 / 逆向设计 / BO：BLOCKED",
            ],
            "details": [
                {
                    "label": x.get("material_id"),
                    "value": (
                        f"{float(x.get('present_rate') or 0) * 100:.0f}%"
                    ),
                    "sub": (
                        f"{x.get('unique_numeric_count')} levels"
                    ),
                }
                for x in (
                    model.get(
                        "exploratory_formula_features"
                    )
                    or []
                )[:5]
            ],
        })
        return base

    if "target_missing" in checks:
        item = results.get("target_missing") or {}
        metrics = item.get("metrics") or []
        ambiguous = bool(
            (
                item.get("metric_resolution") or {}
            ).get("ambiguous")
        )
        base.update({
            "card_type": "target_coverage",
            "status": (
                "REVIEW_REQUIRED"
                if ambiguous else "PASS"
            ),
            "eyebrow": "TARGET COVERAGE",
            "headline": (
                f"{scope_label} 的性能覆盖已检查"
            ),
            "metrics": [
                {
                    "label": x.get("metric"),
                    "value": (
                        f"{x.get('nonempty_count')}/"
                        f"{x.get('samples')}"
                    ),
                    "sub": (
                        f"缺失 {x.get('missing_count')}"
                    ),
                }
                for x in metrics[:4]
            ],
            "highlights": (
                [
                    "“冲击强度”匹配多个真实字段，"
                    "系统分别统计，不做合并。"
                ]
                if ambiguous
                else []
            ),
        })
        return base

    if "quality_overview" in checks:
        q = results.get("quality_overview") or {}
        sparse = q.get("formula_sparsity") or {}
        const = q.get("constant_fields") or {}
        dup = q.get("duplicates") or {}
        units = q.get("units") or {}
        base.update({
            "card_type": "quality_overview",
            "status": "REVIEW_REQUIRED",
            "eyebrow": "REALITY CHECK",
            "headline": (
                f"{scope_label} 已完成数据体检"
            ),
            "metrics": [
                {
                    "label": "高缺失配方字段",
                    "value": sparse.get(
                        "sparse_active_count"
                    ),
                },
                {
                    "label": "恒定配方字段",
                    "value": const.get(
                        "constant_formula_count"
                    ),
                },
                {
                    "label": "重复样品组",
                    "value": dup.get(
                        "duplicate_sample_name_group_count"
                    ),
                },
                {
                    "label": "单位检查",
                    "value": units.get("status"),
                },
            ],
        })
        return base

    base.update({
        "headline": (
            f"{product} 的真实数据已加载"
            if product
            else "单位真实数据已加载"
        ),
        "metrics": (
            [
                {
                    "label": "真实样品",
                    "value": selected.get("sample_count"),
                },
                {
                    "label": "有配方",
                    "value": selected.get(
                        "formula_present_samples"
                    ),
                },
                {
                    "label": "有性能",
                    "value": selected.get(
                        "performance_present_samples"
                    ),
                },
                {
                    "label": "本地项目",
                    "value": selected.get(
                        "local_project_id"
                    ),
                },
            ]
            if selected
            else [
                {
                    "label": "真实样品",
                    "value": summary.get("samples"),
                },
                {
                    "label": "产品类型",
                    "value": summary.get("products"),
                },
                {
                    "label": "原料字段",
                    "value": summary.get("materials"),
                },
                {
                    "label": "性能指标",
                    "value": summary.get(
                        "performance_metrics"
                    ),
                },
            ]
        ),
    })
    return base


def _answer_first(
    presentation: dict[str, Any],
    *,
    selected: dict[str, Any] | None,
    inspection: dict[str, Any] | None,
) -> str:
    card_type = presentation.get("card_type")
    scope = presentation.get("scope_label")
    results = (inspection or {}).get("results") or {}

    if card_type == "sample_count":
        if selected:
            impact = _coverage_for_metric(
                selected,
                "悬臂梁冲击强度",
            )
            extra = (
                f"其中 {selected.get('formula_present_samples')} "
                "个有配方、"
                f"{selected.get('performance_present_samples')} "
                "个有性能数据"
            )
            if impact:
                extra += (
                    "，悬臂梁冲击强度有 "
                    f"{impact.get('nonempty_count')} "
                    "条有效记录"
                )
            return (
                f"{scope} 共有 "
                f"{selected.get('sample_count')} "
                "个真实样品。\n"
                f"{extra}。\n"
                "当前分析对象已设为该产品，"
                "后续可以直接继续问异常值、"
                "缺失情况或建模准备度。"
            )
        return (
            "当前共有 "
            f"{presentation['metrics'][0]['value']} "
            "个真实样品。\n"
            "下面只显示与当前问题有关的关键数据；"
            "需要产品明细时可以直接问产品名。"
        )

    if card_type == "quality":
        item = results.get("outliers") or {}
        metrics_with = int(
            item.get("metrics_with_outliers") or 0
        )
        top = [
            x
            for x in (item.get("top_metrics") or [])
            if int(x.get("outlier_count") or 0) > 0
        ]
        total = sum(
            int(x.get("outlier_count") or 0)
            for x in top
        )
        names = "、".join(
            str(x.get("metric")) for x in top[:3]
        )
        return (
            f"{scope} 有统计异常候选，需要复核。\n"
            f"共检查 {item.get('metrics_checked', 0)} "
            "个可计算性能指标，"
            f"{metrics_with} 个指标出现异常候选，"
            f"合计 {total} 个候选点。"
            + (
                f"\n最明显的几项：{names}。"
                if names else ""
            )
            + "\n这些只是统计离群点，"
            "不会被自动当成错误数据。"
        )

    if card_type == "modeling_readiness":
        model = results.get("modelability") or {}
        inventory = (
            results.get("field_inventory") or {}
        )
        return (
            f"{scope} 可以做探索性建模，"
            "但当前不能作为正式优化模型。\n"
            f"当前范围有 "
            f"{inventory.get('formula_fields_active', 0)} "
            "个活跃配方字段，按覆盖率和数值变化筛选后约 "
            f"{model.get('exploratory_formula_feature_count', 0)} "
            "个配方特征适合进入第一轮探索性模型。\n"
            "主要限制：真实材料工艺参数 "
            f"{model.get('true_process_parameter_rows', 0)} 行，"
            "显式测试条件 "
            f"{model.get('explicit_test_condition_rows', 0)} 行。\n"
            "结论：探索性预测/留出验证可以做；"
            "正式模型、逆向设计和 BO "
            "继续由 Modeling Gate 阻断。"
        )

    if card_type == "target_coverage":
        item = results.get("target_missing") or {}
        rows = item.get("metrics") or []
        lines = [
            (
                f"{metric.get('metric')}：有效 "
                f"{metric.get('nonempty_count')}/"
                f"{metric.get('samples')}，"
                f"缺失 {metric.get('missing_count')}。"
            )
            for metric in rows[:4]
        ]
        ambiguous = bool(
            (
                item.get("metric_resolution") or {}
            ).get("ambiguous")
        )
        prefix = (
            f"{scope} 的“冲击强度”对应多个真实字段，"
            "我分别统计："
            if ambiguous
            else f"{scope} 的目标性能覆盖如下："
        )
        return prefix + "\n" + "\n".join(lines)

    if card_type == "quality_overview":
        q = results.get("quality_overview") or {}
        sparse = q.get("formula_sparsity") or {}
        dup = q.get("duplicates") or {}
        units = q.get("units") or {}
        return (
            f"{scope} 已完成数据体检，当前结论是"
            "“可继续分析，但正式建模前仍需补数据”。\n"
            "高缺失配方字段 "
            f"{sparse.get('sparse_active_count', 0)} 个，"
            "重复样品组 "
            f"{dup.get('duplicate_sample_name_group_count', 0)} 个，"
            "单位元数据状态 "
            f"{units.get('status', 'UNKNOWN')}。\n"
            "建议下一步按产品和目标性能继续缩小范围。"
        )

    return (
        presentation.get("headline")
        or "单位真实数据已加载。"
    )


def build_company_data_overview(
    runtime_root: str | Path,
    *,
    message: str = "",
    product_name: str | None = None,
    classification_override: dict[str, Any] | None = None,
    conversation_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo = CompanyDataRepository(runtime_root)
    try:
        manifest = repo.manifest()
    except CompanyDataValidationError as exc:
        raise CompanyDataUIError(str(exc)) from exc

    selected = None
    if product_name:
        try:
            selected = repo.product(
                product_name=product_name
            )
        except CompanyDataValidationError as exc:
            raise CompanyDataUIError(
                str(exc)
            ) from exc
    elif message:
        selected = resolve_product_from_text(
            repo, message
        )

    classification = (
        classification_override
        or classify_company_data_request(
            message,
            runtime_root=runtime_root,
        )
    )

    inspection = None
    checks = (
        classification.get("requested_checks")
        or []
    )
    if checks:
        try:
            inspection = inspect_company_data(
                repo,
                message=(
                    classification.get(
                        "classification_message"
                    )
                    or message
                ),
                selected_product=selected,
                requested_checks=checks,
            )
        except CompanyDataValidationError as exc:
            raise CompanyDataUIError(
                str(exc)
            ) from exc

    summary = manifest.get("summary") or {}
    safety = manifest.get("safety") or {}

    if selected:
        top_metrics = (
            selected.get("performance_coverage")
            or []
        )[:10]
    else:
        top_metrics = (
            manifest.get("performance_coverage")
            or []
        )[:10]

    source_payload = {
        "archive_name": (
            manifest.get("source", {})
            .get("archive_name")
        ),
        "sha256": (
            manifest.get("source", {})
            .get("sha256")
        ),
        "canonical_source": (
            manifest.get("source", {})
            .get("canonical_source")
        ),
    }

    presentation = _presentation_for(
        summary=summary,
        selected=selected,
        inspection=inspection,
        conversation_scope=conversation_scope,
    )
    answer = _answer_first(
        presentation,
        selected=selected,
        inspection=inspection,
    )

    return {
        "kind": "company_real_data",
        "status": "IMPORTED",
        "dataset_id": manifest.get("dataset_id"),
        "answer": answer,
        "summary": summary,
        "safety": safety,
        "selected_product": selected,
        "top_products": (
            manifest.get("top_products") or []
        )[:12],
        "top_metrics": top_metrics,
        "named_subsets": (
            manifest.get("named_subsets") or []
        ),
        "inspection": inspection,
        "routing": classification,
        "conversation_scope": (
            conversation_scope or {}
        ),
        "presentation": presentation,
        "source": source_payload,
        "warnings": manifest.get("warnings") or [],
    }
