from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any

from company_data import CompanyDataRepository, CompanyDataValidationError


CHECK_ORDER = (
    "sample_count",
    "product_distribution",
    "field_inventory",
    "target_missing",
    "metric_stats",
    "field_coverage",
    "formula_sparsity",
    "constant_fields",
    "duplicates",
    "outliers",
    "units",
    "process_conditions",
    "modelability",
    "quality_overview",
)

EXPLICIT_COMPANY_MARKERS = (
    "单位真实数据",
    "公司真实数据",
    "公司数据",
    "真实数据概况",
    "真实数据",
    "真实样本",
    "真实样品",
    "海科数据",
    "海科真实数据",
    "生产真实数据",
    "单位数据",
    "总库",
)

NEGATIVE_CONTEXT_MARKERS = (
    "训练集",
    "测试集",
    "验证集",
    "训练样本",
    "测试样本",
    "验证样本",
    "模型样本",
    "模拟样本",
    "模拟数据",
    "simulator",
    "synthetic",
    "附件",
    "pdf",
    "docx",
    "上传文件",
    # Historical/RAG questions belong to the knowledge router unless the user
    # explicitly says “公司真实数据/单位真实数据”. This prevents a phrase such
    # as “历史上有没有和这个类似的冲击强度异常” from being hijacked as an
    # outlier Reality Check merely because it contains “冲击强度 + 异常”.
    "历史",
    "以前",
    "过去",
    "曾经",
    "以往",
    "类似案例",
    "类似问题",
    "类似情况",
)

DATA_DOMAIN_MARKERS = (
    "样本",
    "样品",
    "数据",
    "产品",
    "配方",
    "原料",
    "材料",
    "字段",
    "特征",
    "性能",
    "指标",
    "强度",
    "密度",
    "模量",
    "温度",
    "流动",
    "收缩",
    "光泽",
    "含水",
    "燃烧",
    "平均值",
    "工艺",
    "条件",
)

CHECK_MARKERS: dict[str, tuple[str, ...]] = {
    "sample_count": (
        "样本多少", "样品多少", "多少样本", "多少样品",
        "样本量", "样品量", "样本数量", "样品数量",
        "几条数据", "数据量", "多少条数据",
    ),
    "product_distribution": (
        "有哪些产品", "产品有哪些", "产品分布", "各产品",
        "每个产品", "多少产品类型", "产品类型多少",
        "有多少产品类型", "多少个产品", "几种产品",
    ),
    "field_inventory": (
        "有哪些性能指标", "性能指标有哪些", "有哪些配方字段",
        "配方字段有哪些", "有哪些原料字段", "原料字段有哪些",
        "字段有多少", "多少字段", "多少个指标", "指标有多少",
        "多少配方字段", "多少原料字段", "多少性能指标",
        "配方字段多少", "原料字段多少", "性能指标多少",
    ),
    "target_missing": (
        "缺失多少", "缺多少", "缺失率", "缺失值",
        "空值", "为空", "null", "nan", "非空", "有效值",
        "完整率", "非数值", "不是数值",
    ),
    "metric_stats": (
        "最大值", "最小值", "最大多少", "最小多少",
        "平均值", "平均多少", "均值", "中位数",
        "数值范围", "取值范围", "性能范围", "分布统计",
    ),
    "field_coverage": (
        "覆盖率", "覆盖情况", "字段覆盖", "性能覆盖",
        "完整性", "非空率", "有效率",
    ),
    "formula_sparsity": (
        "几乎全是空", "大部分为空", "大量为空",
        "稀疏字段", "字段稀疏", "高缺失字段",
        "缺失率高", "空得多", "太空",
    ),
    "constant_fields": (
        "恒定不变", "恒定字段", "常量字段", "字段恒定",
        "没有变化", "没变化", "零方差", "只有一个值",
    ),
    "duplicates": (
        "重复样品", "重复样本", "重复数据", "重复配方",
        "重复记录", "有没有重复", "是否重复", "去重",
    ),
    "outliers": (
        "异常值", "离群值", "离群点", "极端值",
        "outlier", "异常数据", "数值异常",
    ),
    "units": (
        "单位混乱", "单位一致", "单位不一致", "单位冲突",
        "单位问题", "量纲", "unit", "单位有没有",
    ),
    "process_conditions": (
        "工艺参数有多少", "工艺数据有多少", "工艺参数缺失",
        "有没有工艺参数", "测试条件有多少", "条件数据有多少",
        "测试条件缺失", "有没有测试条件", "工艺覆盖",
        "测试条件覆盖",
    ),
    "modelability": (
        "哪些字段可以用于建模", "哪些字段适合建模",
        "可用于建模", "可建模字段", "建模字段",
        "特征可用", "能不能建模", "为什么不能建模",
        "modeling gate", "gate不通过", "gate 不通过",
        "为什么gate", "为什么 gate",
    ),
    "quality_overview": (
        "数据质量", "数据体检", "reality check", "realitycheck",
        "质量怎么样", "数据怎么样", "数据情况", "检查数据",
    ),
}

METRIC_ALIASES = (
    "冲击强度",
    "冲击",
    "流动速率",
    "流动",
    "光泽度",
    "光泽",
    "收缩率",
    "收缩",
    "拉伸",
    "弯曲",
    "燃烧",
)

COUNT_WORDS = (
    "多少", "几条", "几份", "数量", "数据量", "样本量",
    "样品量", "几个", "几种",
)


# V0.3 semantic slots.
# These are intentionally concept-level words rather than complete sentences,
# so fillers such as “现在的 / 目前 / 了 / 一下 / 帮我看看” do not break routing.
QUESTION_QUANTITY_MARKERS = (
    "多少", "几条", "几项", "几个", "几份", "几种", "多不多",
    "比例", "百分比", "率", "情况", "程度",
)
QUESTION_LIST_MARKERS = (
    "哪些", "哪个", "有什么", "有哪些", "列", "列出", "列出来",
    "罗列", "看看", "看下",
)
QUESTION_BOOLEAN_MARKERS = (
    "有没有", "是否", "是不是", "吗", "么", "不", "没",
)

MISSING_CONCEPTS = (
    "缺失", "缺少", "缺", "空值", "空的", "为空", "是空",
    "没值", "没有值", "无值", "没数据", "没有数据", "无数据",
    "不完整", "不全", "不齐", "齐不齐", "完整吗", "完整不",
    "有效值", "有效数据", "非空", "有值", "有数据", "剩多少",
    "还剩", "可用数据",
)
SPARSITY_CONCEPTS = (
    "稀疏", "太空", "很空", "空得多", "空的多", "空得特别多",
    "空的特别多", "空得比较多", "缺得多",
    "缺失高", "缺失严重", "大部分空", "大多为空", "几乎为空",
    "几乎全空", "大量为空", "大量空值",
)
CONSTANT_CONCEPTS = (
    "恒定", "常量", "不变", "没变化", "没有变化", "零方差",
    "只有一个值", "都一样", "全部一样", "一直一样",
)
DUPLICATE_CONCEPTS = (
    "重复", "重样", "重了", "重复录入", "一样的样品",
    "一样的配方", "相同配方",
)
OUTLIER_CONCEPTS = (
    "异常", "离群", "极端", "反常", "outlier", "离谱",
)
UNIT_CONCEPTS = (
    "单位", "量纲", "unit",
)
UNIT_PROBLEM_CONCEPTS = (
    "混乱", "乱", "不一致", "一致", "冲突", "问题", "统一",
    "一样", "对不对",
)
STAT_CONCEPTS = (
    "最大", "最高", "最小", "最低", "平均", "均值", "中位",
    "范围", "分布", "波动",
)
PRODUCT_CONCEPTS = (
    "产品", "牌号", "产品类型",
)
FIELD_CONCEPTS = (
    "字段", "数据列", "列名", "特征", "变量",
)
FORMULA_CONCEPTS = (
    "配方", "原料", "材料",
)
PERFORMANCE_CONCEPTS = (
    "性能", "指标", "强度", "模量", "密度", "流动", "收缩",
    "光泽", "燃烧", "含水", "冲击", "拉伸", "弯曲",
)
PROCESS_CONCEPTS = (
    "工艺", "工艺参数", "加工参数", "测试条件", "试验条件",
    "测试环境", "实验条件",
)
MODEL_CONCEPTS = (
    "建模", "模型", "modeling gate", "gate", "训练",
)
QUALITY_CONCEPTS = (
    "数据质量", "数据体检", "reality check", "质量", "数据情况",
    "数据怎么样", "检查数据", "体检一下",
)

# Polite / temporal filler phrases are normalized away only for semantic
# detection. The original user text remains available for product/metric lookup.
FILLER_PHRASES = (
    "请问", "麻烦", "麻烦你", "帮我", "帮忙", "帮我看", "帮我看看",
    "看一下", "看下", "查一下", "查下", "我想知道", "能不能看看",
    "能不能查", "现在的", "目前的", "当前的", "现有的",
    "现在", "目前", "当前", "这批", "这个", "这些", "一下",
)


def _normalize_query_text(value: Any) -> dict[str, str]:
    """Normalize user wording without changing domain identifiers.

    - Unicode NFKC: full-width punctuation/numbers become canonical.
    - lower/casefold.
    - punctuation and whitespace collapsed for concept detection.
    - common conversational fillers removed from semantic_text only.
    """
    raw = unicodedata.normalize("NFKC", str(value or "")).strip()
    lowered = raw.casefold()
    lowered = re.sub(r"[，。！？、；：,.!?;:()（）【】\\[\\]{}<>《》“”\"'`~]+", " ", lowered)
    lowered = re.sub(r"\\s+", " ", lowered).strip()
    compact = lowered.replace(" ", "")

    semantic = compact
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        semantic = semantic.replace(phrase.casefold().replace(" ", ""), "")
    # Sentence particles often appear between the semantic predicate and
    # quantity word, e.g. “缺失了多少”. Removing them makes slot matching
    # robust while not affecting the original text used for entity resolution.
    semantic = re.sub(r"(?<=[\\u4e00-\\u9fff])(了|呢|啊|呀|吧|嘛)(?=[\\u4e00-\\u9fff])", "", semantic)
    return {
        "raw": raw,
        "lowered": lowered,
        "compact": compact,
        "semantic": semantic,
    }


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(str(value).casefold().replace(" ", "") in text for value in values)


def _semantic_slots(
    normalized: dict[str, str],
    *,
    matched_products: list[str],
    matched_metrics: list[str],
) -> dict[str, bool]:
    text = normalized["semantic"]
    compact = normalized["compact"]
    has_quantity = _contains_any(text, QUESTION_QUANTITY_MARKERS)
    has_list = _contains_any(text, QUESTION_LIST_MARKERS)
    has_boolean = _contains_any(text, QUESTION_BOOLEAN_MARKERS)
    has_formula = _contains_any(text, FORMULA_CONCEPTS)
    has_performance = _contains_any(text, PERFORMANCE_CONCEPTS)
    has_field = _contains_any(text, FIELD_CONCEPTS)
    has_process = _contains_any(text, PROCESS_CONCEPTS)
    has_model = _contains_any(text, MODEL_CONCEPTS)

    return {
        "quantity": has_quantity,
        "list": has_list,
        "boolean": has_boolean,
        "missing": _contains_any(text, MISSING_CONCEPTS),
        "sparsity": _contains_any(text, SPARSITY_CONCEPTS),
        "constant": _contains_any(text, CONSTANT_CONCEPTS),
        "duplicate": _contains_any(text, DUPLICATE_CONCEPTS),
        "outlier": _contains_any(text, OUTLIER_CONCEPTS),
        "unit": _contains_any(text, UNIT_CONCEPTS),
        "unit_problem": _contains_any(text, UNIT_PROBLEM_CONCEPTS),
        "statistics": _contains_any(text, STAT_CONCEPTS),
        "product": _contains_any(text, PRODUCT_CONCEPTS),
        "field": has_field,
        "formula": has_formula,
        "performance": has_performance,
        "process": has_process,
        "model": has_model,
        "quality": _contains_any(text, QUALITY_CONCEPTS),
        "sample": _contains_any(text, ("样本", "样品", "记录")),
        "known_product": bool(matched_products),
        "known_metric": bool(matched_metrics),
        # “冲击强度” etc. may be aliases rather than exact imported metrics.
        "performance_subject": has_performance or bool(matched_metrics),
        "data_subject": _contains_any(
            compact,
            (
                "数据", "样本", "样品", "记录", "字段", "配方", "原料",
                "性能", "指标", "产品", "工艺", "条件",
            ),
        ),
    }


def _checks_from_semantics(slots: dict[str, bool]) -> list[str]:
    checks: list[str] = []

    if slots["sample"] and (slots["quantity"] or slots["list"]):
        checks.append("sample_count")

    if slots["product"] and (slots["quantity"] or slots["list"]):
        checks.append("product_distribution")

    if (slots["field"] or slots["formula"] or slots["performance"]) and slots["list"]:
        checks.append("field_inventory")

    # Missingness of an explicit performance subject = target coverage.
    # Generic “哪些字段缺失高” = field/formula coverage instead.
    if slots["missing"]:
        if slots["performance_subject"] and not (slots["field"] or slots["formula"]):
            checks.append("target_missing")
        elif slots["field"] or slots["formula"]:
            checks.extend(["field_coverage", "formula_sparsity"])
        elif slots["data_subject"]:
            checks.append("field_coverage")

    if slots["sparsity"]:
        checks.append("formula_sparsity")

    if slots["constant"]:
        checks.append("constant_fields")

    if slots["duplicate"]:
        checks.append("duplicates")

    if slots["outlier"]:
        checks.append("outliers")

    if slots["unit"] and (
        slots["unit_problem"] or slots["boolean"] or slots["quantity"]
    ):
        checks.append("units")

    if slots["statistics"] and slots["performance_subject"]:
        checks.append("metric_stats")

    if slots["process"] and (
        slots["quantity"] or slots["missing"] or slots["list"]
        or slots["boolean"]
    ):
        checks.append("process_conditions")

    if slots["model"] and (
        slots["field"] or slots["formula"] or slots["quality"]
        or slots["boolean"] or slots["missing"]
    ):
        checks.append("modelability")

    if slots["quality"]:
        checks.append("quality_overview")

    return list(dict.fromkeys(checks))


def _safe_manifest(
    runtime_root: str | Path | None,
) -> dict[str, Any] | None:
    if runtime_root is None:
        return None
    try:
        return CompanyDataRepository(runtime_root).manifest()
    except (CompanyDataValidationError, OSError, ValueError):
        return None


def _known_terms(
    manifest: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    if not manifest:
        return [], []
    products = [
        str(item.get("product_type") or "").strip()
        for item in manifest.get("products") or []
        if str(item.get("product_type") or "").strip()
    ]
    metrics = [
        str(item.get("metric") or "").strip()
        for item in manifest.get("performance_coverage") or []
        if str(item.get("metric") or "").strip()
    ]
    return products, metrics


def classify_company_data_request(
    message: str,
    *,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministic, fail-closed routing classifier for company real data.

    Strong explicit company markers win. Without an explicit company marker,
    known product/metric names and data-quality wording are required.

    The exclusions prevent phrases such as "训练样本多少" and
    "模拟数据异常值" from being silently interpreted as company history.
    """
    normalized = _normalize_query_text(message)
    text = normalized["raw"]
    lowered = normalized["compact"]
    semantic_text = normalized["semantic"]
    manifest = _safe_manifest(runtime_root)
    known_products, known_metrics = _known_terms(manifest)

    explicit_company = any(
        marker.casefold() in lowered
        for marker in EXPLICIT_COMPANY_MARKERS
    )
    negative_context = [
        marker for marker in NEGATIVE_CONTEXT_MARKERS
        if marker.casefold().replace(" ", "") in lowered
    ]

    project_match = re.search(
        r"(?:project|项目)\s*#?\s*(\d{3,})",
        text,
        re.IGNORECASE,
    )
    numeric_sample_ids = re.findall(
        r"(?<!\d)(\d{3,6})(?!\d)",
        text,
    )
    has_nonlocal_project = bool(
        project_match
        and not str(project_match.group(1)).startswith("93")
    )

    matched_products = [
        name for name in known_products
        if name.casefold() in lowered
    ]
    matched_metrics = [
        name for name in known_metrics
        if name.casefold() in lowered
    ]

    # Short unique product aliases, e.g. "FR303", are handled as a routing
    # hint only when unique in the imported product catalogue.
    alias_hits: list[str] = []
    if known_products and not matched_products:
        token_to_products: dict[str, list[str]] = defaultdict(list)
        for name in known_products:
            for token in re.findall(r"[A-Za-z]+[A-Za-z0-9-]*\d+[A-Za-z0-9-]*", name):
                if len(token) >= 4:
                    token_to_products[token.casefold()].append(name)
        for token, names in token_to_products.items():
            if token in lowered and len(set(names)) == 1:
                alias_hits.extend(names)
    matched_products = sorted(set(matched_products + alias_hits))

    requested_checks: list[str] = []
    for check in CHECK_ORDER:
        markers = CHECK_MARKERS.get(check) or ()
        if any(
            marker.casefold().replace(" ", "") in semantic_text
            for marker in markers
        ):
            requested_checks.append(check)

    semantic_slots = _semantic_slots(
        normalized,
        matched_products=matched_products,
        matched_metrics=matched_metrics,
    )
    requested_checks.extend(
        check
        for check in _checks_from_semantics(semantic_slots)
        if check not in requested_checks
    )

    has_count_question = bool(semantic_slots["quantity"])
    has_sample_word = bool(semantic_slots["sample"])
    if (
        "sample_count" not in requested_checks
        and has_count_question
        and (
            has_sample_word
            or bool(matched_products)
            or "真实" in lowered
        )
    ):
        requested_checks.append("sample_count")

    if (
        "product_distribution" not in requested_checks
        and has_count_question
        and semantic_slots["product"]
    ):
        requested_checks.append("product_distribution")

    # Generic field-missing questions are field-coverage questions, not a
    # request to invent a single target metric.
    if "target_missing" in requested_checks and any(
        token in lowered
        for token in ("字段", "配方", "原料", "性能指标")
    ):
        if "field_coverage" not in requested_checks:
            requested_checks.append("field_coverage")
        if any(token in lowered for token in ("配方", "原料", "字段")):
            if "formula_sparsity" not in requested_checks:
                requested_checks.append("formula_sparsity")
        requested_checks = [
            check for check in requested_checks
            if check != "target_missing"
        ]

    # Generic target missing / outlier questions should route when a known
    # metric or a strong materials-performance noun is present.
    has_quality_marker = any(
        check in requested_checks
        for check in (
            "target_missing",
            "metric_stats",
            "field_coverage",
            "formula_sparsity",
            "constant_fields",
            "duplicates",
            "outliers",
            "units",
            "process_conditions",
            "modelability",
            "quality_overview",
            "field_inventory",
        )
    )
    has_domain = (
        any(marker.casefold().replace(" ", "") in lowered for marker in DATA_DOMAIN_MARKERS)
        or semantic_slots["data_subject"]
        or semantic_slots["performance_subject"]
    )
    known_term_hit = bool(matched_products or matched_metrics)

    blocked_by: list[str] = []
    if negative_context and not explicit_company:
        blocked_by.append("negative_context")
    if has_nonlocal_project and not explicit_company:
        blocked_by.append("nonlocal_project_context")
    if (
        numeric_sample_ids
        and not explicit_company
        and not matched_products
        and not project_match
    ):
        # "3811样品..." is more likely a database sample lookup than a
        # company-dataset-wide Reality Check.
        blocked_by.append("specific_numeric_sample_context")

    route = False
    reason = "NO_MATCH"
    if explicit_company:
        route = True
        reason = "EXPLICIT_COMPANY_SCOPE"
    elif blocked_by:
        route = False
        reason = "EXCLUDED_AMBIGUOUS_CONTEXT"
    elif known_term_hit and (
        has_quality_marker or has_count_question
    ):
        route = True
        reason = "KNOWN_IMPORTED_TERM_PLUS_DATA_QUERY"
    elif requested_checks and has_domain:
        route = True
        reason = "DATA_QUALITY_QUERY"
    elif any(
        check in requested_checks
        for check in ("outliers", "units", "quality_overview", "duplicates")
    ):
        # These are strong standalone data-inspection questions.
        # Negative contexts above still win, so "测试集有没有异常值" and
        # "模拟数据有没有异常值" are not hijacked.
        route = True
        reason = "STANDALONE_REALITY_CHECK"
    elif (
        "sample_count" in requested_checks
        and has_sample_word
    ):
        route = True
        reason = "SAMPLE_COUNT_QUERY"

    # If a query merely mentions a product/metric but asks no data question,
    # do not hijack it.
    ordered_checks = [
        check for check in CHECK_ORDER
        if check in set(requested_checks)
    ]

    confidence = 0.0
    if explicit_company:
        confidence += 0.45
    if known_term_hit:
        confidence += 0.30
    if ordered_checks:
        confidence += 0.30
    if any(
        check in ordered_checks
        for check in ("outliers", "units", "quality_overview", "duplicates")
    ):
        confidence += 0.15
    if has_domain:
        confidence += 0.15
    if semantic_slots["quantity"] or semantic_slots["list"] or semantic_slots["boolean"]:
        confidence += 0.05
    if blocked_by:
        confidence = min(confidence, 0.25)
    confidence = min(confidence, 1.0)

    return {
        "route": route,
        "reason": reason,
        "confidence": round(confidence, 3),
        "requested_checks": ordered_checks,
        "explicit_company_scope": explicit_company,
        "matched_products": matched_products,
        "matched_metrics": matched_metrics,
        "blocked_by": blocked_by,
        "normalized_message": normalized["semantic"],
        "matched_semantics": {
            key: value
            for key, value in semantic_slots.items()
            if value
        },
    }


def resolve_product_from_text(
    repo: CompanyDataRepository,
    text: str,
) -> dict[str, Any] | None:
    direct = repo.detect_product_in_text(text)
    if direct is not None:
        return direct

    lowered = str(text or "").casefold()
    products = repo.manifest().get("products") or []
    token_to_products: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in products:
        name = str(item.get("product_type") or "")
        for token in re.findall(
            r"[A-Za-z]+[A-Za-z0-9-]*\d+[A-Za-z0-9-]*",
            name,
        ):
            if len(token) >= 4:
                token_to_products[token.casefold()].append(item)

    candidates = []
    for token, items in token_to_products.items():
        if token in lowered:
            unique = {
                str(item.get("product_type") or ""): item
                for item in items
            }
            if len(unique) == 1:
                candidates.extend(unique.values())

    names = {
        str(item.get("product_type") or ""): item
        for item in candidates
    }
    if len(names) == 1:
        return next(iter(names.values()))
    return None


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _norm_value(value: Any) -> str:
    number = _num(value)
    if number is not None:
        return f"{number:.12g}"
    return str(value or "").strip()


def _load_wide_rows(
    repo: CompanyDataRepository,
    product_type: str | None,
) -> tuple[list[str], list[dict[str, str]]]:
    manifest = repo.manifest()
    path = repo.import_dir(manifest["dataset_id"]) / "catalog_wide.csv"
    if not path.is_file():
        raise CompanyDataValidationError(
            f"catalog_wide.csv 不存在: {path}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [
            row for row in reader
            if (
                product_type is None
                or row.get("product_type") == product_type
            )
        ]
    return fields, rows


def _metric_candidates(
    message: str,
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    normalized = _normalize_query_text(message)
    text = normalized["compact"]
    metrics = [
        str(item.get("metric") or "")
        for item in manifest.get("performance_coverage") or []
        if str(item.get("metric") or "")
    ]

    exact = [
        metric for metric in metrics
        if metric.casefold().replace(" ", "") in text
    ]
    if exact:
        chosen = exact
        mode = "EXACT_NAME"
    else:
        alias = next(
            (
                token for token in METRIC_ALIASES
                if token.casefold().replace(" ", "") in text
            ),
            None,
        )
        if alias:
            chosen = [
                metric for metric in metrics
                if alias.casefold().replace(" ", "") in metric.casefold().replace(" ", "")
            ]
            mode = "ALIAS_MULTI_MATCH" if len(chosen) > 1 else "ALIAS"
        else:
            chosen = []
            mode = "NONE"

    coverage = []
    for metric in chosen:
        col = f"performance::{metric}"
        nonempty = sum(
            bool(str(row.get(col) or "").strip())
            for row in rows
        )
        numeric = sum(
            _num(row.get(col)) is not None
            for row in rows
        )
        coverage.append({
            "metric": metric,
            "nonempty_count": nonempty,
            "numeric_count": numeric,
        })

    coverage.sort(
        key=lambda item: (
            -item["nonempty_count"],
            item["metric"],
        )
    )
    return {
        "resolution_mode": mode,
        "ambiguous": len(coverage) > 1,
        "matches": coverage,
        "merged": False,
    }


def _sample_count(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    products = Counter(
        str(row.get("product_type") or "")
        for row in rows
    )
    return {
        "samples": len(rows),
        "products": len([x for x in products if x]),
    }



def _product_distribution(
    rows: list[dict[str, str]],
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    counts = Counter(
        str(row.get("product_type") or "").strip()
        for row in rows
        if str(row.get("product_type") or "").strip()
    )
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "product_type_count": len(ordered),
        "products": [
            {"product_type": name, "sample_count": count}
            for name, count in ordered[:top_n]
        ],
    }


def _field_inventory(
    fields: list[str],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    formula = [c for c in fields if c.startswith("formula::")]
    performance = [
        c for c in fields if c.startswith("performance::")
    ]
    active_formula = [
        c for c in formula
        if any(str(row.get(c) or "").strip() for row in rows)
    ]
    active_performance = [
        c for c in performance
        if any(str(row.get(c) or "").strip() for row in rows)
    ]
    return {
        "formula_fields_total": len(formula),
        "formula_fields_active": len(active_formula),
        "performance_fields_total": len(performance),
        "performance_fields_active": len(active_performance),
        "formula_label_mode": "MATERIAL_ID",
    }


def _missing_for_metrics(
    metric_resolution: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    total = len(rows)
    results = []
    for item in metric_resolution["matches"]:
        metric = item["metric"]
        col = f"performance::{metric}"
        nonempty = sum(
            bool(str(row.get(col) or "").strip())
            for row in rows
        )
        numeric = sum(
            _num(row.get(col)) is not None
            for row in rows
        )
        missing = total - nonempty
        results.append({
            "metric": metric,
            "samples": total,
            "nonempty_count": nonempty,
            "numeric_count": numeric,
            "non_numeric_count": max(nonempty - numeric, 0),
            "missing_count": missing,
            "missing_rate": (
                missing / total if total else None
            ),
        })
    return {
        "metric_resolution": metric_resolution,
        "metrics": results,
    }



def _metric_stats(
    metric_resolution: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not metric_resolution["matches"]:
        return {
            "status": "METRIC_NOT_RESOLVED",
            "message": "未确定具体性能指标，因此没有擅自汇总不同性能字段。",
        }
    items = []
    for match in metric_resolution["matches"]:
        metric = match["metric"]
        col = f"performance::{metric}"
        values = [
            _num(row.get(col))
            for row in rows
        ]
        values = [x for x in values if x is not None]
        if not values:
            items.append({
                "metric": metric,
                "numeric_count": 0,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "median": None,
            })
            continue
        items.append({
            "metric": metric,
            "numeric_count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        })
    return {
        "metric_resolution": metric_resolution,
        "metrics": items,
    }


def _field_coverage(
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    total = len(rows)
    formula = [c for c in fields if c.startswith("formula::")]
    performance = [
        c for c in fields if c.startswith("performance::")
    ]

    def rows_for(cols: list[str]) -> list[dict[str, Any]]:
        out = []
        for col in cols:
            present = sum(
                bool(str(row.get(col) or "").strip())
                for row in rows
            )
            out.append({
                "field": col,
                "present_count": present,
                "missing_count": total - present,
                "present_rate": present / total if total else None,
            })
        out.sort(
            key=lambda x: (
                x["present_rate"] if x["present_rate"] is not None else -1,
                x["field"],
            )
        )
        return out

    formula_rows = rows_for(formula)
    performance_rows = rows_for(performance)
    return {
        "samples": total,
        "formula_lowest_coverage": formula_rows[:top_n],
        "performance_lowest_coverage": performance_rows[:top_n],
        "formula_highest_coverage": list(reversed(formula_rows[-top_n:])),
        "performance_highest_coverage": list(
            reversed(performance_rows[-top_n:])
        ),
    }


def _formula_sparsity(
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    missing_rate_threshold: float = 0.80,
    top_n: int = 30,
) -> dict[str, Any]:
    total = len(rows)
    formula = [c for c in fields if c.startswith("formula::")]
    items = []
    for col in formula:
        present = sum(
            bool(str(row.get(col) or "").strip())
            for row in rows
        )
        missing = total - present
        rate = missing / total if total else 1.0
        items.append({
            "field": col,
            "material_id": col.removeprefix("formula::"),
            "present_count": present,
            "missing_count": missing,
            "missing_rate": rate,
        })

    fully_empty = [
        item for item in items if item["present_count"] == 0
    ]
    sparse_active = [
        item for item in items
        if (
            item["present_count"] > 0
            and item["missing_rate"] >= missing_rate_threshold
        )
    ]
    sparse_active.sort(
        key=lambda x: (
            -x["missing_rate"],
            x["present_count"],
            x["field"],
        )
    )
    return {
        "samples": total,
        "threshold_missing_rate": missing_rate_threshold,
        "formula_fields_total": len(formula),
        "fully_empty_count": len(fully_empty),
        "sparse_active_count": len(sparse_active),
        "fully_empty": fully_empty[:top_n],
        "sparse_active": sparse_active[:top_n],
        "display_note": (
            "当前规范化导入层仅保留 material_id，"
            "因此配方字段以 material_id 展示。"
        ),
    }


def _constant_formula_fields(
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    top_n: int = 30,
) -> dict[str, Any]:
    total = len(rows)
    min_support = max(3, math.ceil(total * 0.05))
    formula = [c for c in fields if c.startswith("formula::")]
    constants = []
    low_support_single_value = []
    for col in formula:
        values = [
            _norm_value(row.get(col))
            for row in rows
            if str(row.get(col) or "").strip()
        ]
        unique = sorted(set(values))
        if len(unique) != 1:
            continue
        item = {
            "field": col,
            "material_id": col.removeprefix("formula::"),
            "present_count": len(values),
            "present_rate": len(values) / total if total else None,
            "constant_value": unique[0],
        }
        if len(values) >= min_support:
            constants.append(item)
        elif values:
            low_support_single_value.append(item)

    constants.sort(
        key=lambda x: (-x["present_count"], x["field"])
    )
    low_support_single_value.sort(
        key=lambda x: (-x["present_count"], x["field"])
    )
    return {
        "samples": total,
        "minimum_support_for_constant": min_support,
        "constant_formula_count": len(constants),
        "constant_formula_fields": constants[:top_n],
        "single_value_low_support_count": len(
            low_support_single_value
        ),
        "low_support_note": (
            "低覆盖字段只有一个观测值时，不判定为“恒定字段”，"
            "避免把稀疏数据误判为零方差特征。"
        ),
    }


def _signature(
    row: dict[str, str],
    columns: list[str],
) -> tuple[str, ...]:
    return tuple(_norm_value(row.get(col)) for col in columns)


def _duplicates(
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    max_groups: int = 12,
) -> dict[str, Any]:
    sample_names: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        name = str(row.get("sample_name") or "").strip()
        if name:
            sample_names[name].append(row)
    duplicate_names = [
        {
            "sample_name": name,
            "count": len(group),
            "sample_keys": [
                str(row.get("sample_key") or "")
                for row in group[:20]
            ],
        }
        for name, group in sample_names.items()
        if len(group) > 1
    ]
    duplicate_names.sort(
        key=lambda x: (-x["count"], x["sample_name"])
    )

    formula_cols = [
        col for col in fields if col.startswith("formula::")
    ]
    active_formula = [
        col for col in formula_cols
        if any(str(row.get(col) or "").strip() for row in rows)
    ]
    formula_groups: dict[tuple[str, ...], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in rows:
        if active_formula:
            sig = _signature(row, active_formula)
            formula_groups[sig].append(row)

    duplicate_formula_groups = []
    for group in formula_groups.values():
        if len(group) <= 1:
            continue
        duplicate_formula_groups.append({
            "count": len(group),
            "sample_keys": [
                str(row.get("sample_key") or "")
                for row in group[:20]
            ],
            "sample_names": [
                str(row.get("sample_name") or "")
                for row in group[:20]
            ],
        })
    duplicate_formula_groups.sort(
        key=lambda x: (
            -x["count"],
            x["sample_keys"][0] if x["sample_keys"] else "",
        )
    )

    return {
        "duplicate_sample_name_group_count": len(
            duplicate_names
        ),
        "duplicate_sample_name_rows": sum(
            item["count"] for item in duplicate_names
        ),
        "duplicate_sample_name_groups": duplicate_names[:max_groups],
        "duplicate_formula_group_count": len(
            duplicate_formula_groups
        ),
        "duplicate_formula_groups": duplicate_formula_groups[
            :max_groups
        ],
        "identity_note": (
            "sample_key 是导入后的唯一身份；“重复配方”只表示"
            "规范化配方签名相同，不等同于重复录入。"
        ),
    }


def _quartiles(values: list[float]) -> tuple[float, float]:
    try:
        q = statistics.quantiles(
            values, n=4, method="inclusive"
        )
        return q[0], q[2]
    except statistics.StatisticsError:
        return min(values), max(values)


def _outlier_indices(
    values: list[float],
) -> tuple[set[int], str]:
    if len(values) < 5:
        return set(), "INSUFFICIENT_N_LT_5"
    q1, q3 = _quartiles(values)
    iqr = q3 - q1
    if iqr > 0:
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        return {
            i for i, value in enumerate(values)
            if value < low or value > high
        }, "IQR_1.5"

    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad <= 0:
        return set(), "ZERO_IQR_ZERO_MAD"
    return {
        i for i, value in enumerate(values)
        if 0.6745 * abs(value - median) / mad > 3.5
    }, "MAD_MODIFIED_Z_3.5"


def _metric_outliers(
    rows: list[dict[str, str]],
    metric: str,
    *,
    max_samples: int = 20,
) -> dict[str, Any]:
    col = f"performance::{metric}"
    numeric_rows = [
        (row, _num(row.get(col)))
        for row in rows
    ]
    numeric_rows = [
        (row, value)
        for row, value in numeric_rows
        if value is not None
    ]
    values = [value for _, value in numeric_rows]
    indices, method = _outlier_indices(values)
    samples = [
        {
            "sample_key": numeric_rows[i][0].get("sample_key"),
            "sample_name": numeric_rows[i][0].get("sample_name"),
            "value": numeric_rows[i][1],
        }
        for i in sorted(indices)
    ]
    samples.sort(key=lambda x: abs(x["value"]), reverse=True)
    return {
        "metric": metric,
        "numeric_count": len(values),
        "outlier_count": len(samples),
        "method": method,
        "samples": samples[:max_samples],
        "warning": (
            "统计异常值只是需要复核的候选点，不等于错误数据。"
        ),
    }


def _outliers(
    fields: list[str],
    rows: list[dict[str, str]],
    metric_resolution: dict[str, Any],
    *,
    top_n: int = 15,
) -> dict[str, Any]:
    if metric_resolution["matches"]:
        metrics = [
            item["metric"]
            for item in metric_resolution["matches"]
        ]
        results = [
            _metric_outliers(rows, metric)
            for metric in metrics
        ]
        return {
            "scope": "REQUESTED_METRICS",
            "metric_resolution": metric_resolution,
            "metrics": results,
        }

    metrics = [
        col.removeprefix("performance::")
        for col in fields
        if col.startswith("performance::")
    ]
    results = [
        _metric_outliers(rows, metric, max_samples=5)
        for metric in metrics
    ]
    results = [
        item for item in results
        if item["numeric_count"] >= 5
    ]
    results.sort(
        key=lambda x: (
            -x["outlier_count"],
            -x["numeric_count"],
            x["metric"],
        )
    )
    return {
        "scope": "ALL_NUMERIC_PERFORMANCE_METRICS",
        "metrics_checked": len(results),
        "metrics_with_outliers": sum(
            item["outlier_count"] > 0
            for item in results
        ),
        "top_metrics": results[:top_n],
        "warning": (
            "统计异常值只是需要复核的候选点，不等于错误数据。"
        ),
    }


def _unit_check(
    repo: CompanyDataRepository,
) -> dict[str, Any]:
    path = repo.import_dir() / "performances_long.csv"
    columns: list[str] = []
    if path.is_file():
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.reader(f)
            columns = next(reader, [])
    has_unit_column = any(
        col.casefold() in {
            "unit", "units", "单位", "量纲", "test_unit"
        }
        for col in columns
    )
    if not has_unit_column:
        return {
            "status": "NOT_AVAILABLE",
            "can_assert_consistency": False,
            "unit_column_present": False,
            "conflict_count": None,
            "reason": (
                "当前规范化 performances_long.csv 没有单位字段。"
                "因此不能可靠判断“单位是否混乱”，也不会把未知误报为 0 个冲突。"
            ),
            "required_for_future": (
                "后续导入需保留 metric + value + unit（最好再保留测试标准/条件），"
                "才能做确定性的单位一致性检查与换算。"
            ),
        }
    return {
        "status": "AVAILABLE",
        "can_assert_consistency": True,
        "unit_column_present": True,
        "conflict_count": 0,
        "reason": "单位字段已存在；当前版本尚未实现跨单位换算规则表。",
    }



def _process_conditions(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    summary = manifest.get("summary") or {}
    return {
        "workflow_metadata_rows": int(
            summary.get("workflow_metadata_rows") or 0
        ),
        "material_process_parameter_rows": int(
            summary.get("material_process_parameter_rows") or 0
        ),
        "explicit_test_condition_rows": int(
            summary.get("explicit_test_condition_rows") or 0
        ),
        "workflow_metadata_is_process_feature": False,
        "note": (
            "LOGINCATEGORY/TASKCATEGORY 是工作流元数据，"
            "不会伪装成材料工艺参数。"
        ),
    }


def _modelability(
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    manifest: dict[str, Any],
    metric_resolution: dict[str, Any],
) -> dict[str, Any]:
    total = len(rows)
    formula_cols = [
        col for col in fields if col.startswith("formula::")
    ]
    candidates = []
    for col in formula_cols:
        values = [
            _num(row.get(col))
            for row in rows
            if str(row.get(col) or "").strip()
        ]
        values = [value for value in values if value is not None]
        present = len(values)
        unique = len(set(values))
        if (
            total > 0
            and present / total >= 0.20
            and unique >= 2
        ):
            candidates.append({
                "field": col,
                "material_id": col.removeprefix("formula::"),
                "present_count": present,
                "present_rate": present / total,
                "unique_numeric_count": unique,
            })
    candidates.sort(
        key=lambda x: (
            -x["present_rate"],
            -x["unique_numeric_count"],
            x["field"],
        )
    )

    target_readiness = []
    for item in metric_resolution["matches"]:
        target_readiness.append({
            "metric": item["metric"],
            "numeric_count": item["numeric_count"],
            "numeric_rate": (
                item["numeric_count"] / total if total else None
            ),
        })

    summary = manifest.get("summary") or {}
    safety = manifest.get("safety") or {}
    blockers = []
    if int(summary.get("material_process_parameter_rows") or 0) == 0:
        blockers.append("NO_TRUE_PROCESS_PARAMETERS")
    if int(summary.get("explicit_test_condition_rows") or 0) == 0:
        blockers.append("NO_EXPLICIT_TEST_CONDITIONS")
    if not safety.get("official_model_allowed_from_import_alone", False):
        blockers.append("IMPORT_ALONE_NOT_OFFICIAL_MODEL_READY")

    return {
        "samples": total,
        "exploratory_formula_feature_count": len(candidates),
        "exploratory_formula_features": candidates[:30],
        "target_readiness": target_readiness,
        "true_process_parameter_rows": int(
            summary.get("material_process_parameter_rows") or 0
        ),
        "explicit_test_condition_rows": int(
            summary.get("explicit_test_condition_rows") or 0
        ),
        "official_model_allowed_from_import_alone": False,
        "blockers": blockers,
        "decision_note": (
            "这里是导入数据体检，不替代正式 Modeling Gate。"
            "可做探索性模型/留出验证，但正式逆向设计和闭环 BO 仍必须通过 Gate。"
        ),
    }


def inspect_company_data(
    repo: CompanyDataRepository,
    *,
    message: str,
    selected_product: dict[str, Any] | None = None,
    requested_checks: list[str] | None = None,
) -> dict[str, Any]:
    manifest = repo.manifest()
    selected_product = selected_product or resolve_product_from_text(
        repo, message
    )
    product_type = (
        str(selected_product.get("product_type"))
        if selected_product else None
    )
    fields, rows = _load_wide_rows(repo, product_type)
    classification = classify_company_data_request(
        message,
        runtime_root=repo.runtime_root,
    )
    checks = requested_checks or classification["requested_checks"]
    if not checks:
        checks = ["quality_overview"]

    metric_resolution = _metric_candidates(
        message,
        manifest=manifest,
        rows=rows,
    )

    results: dict[str, Any] = {}
    if "sample_count" in checks:
        results["sample_count"] = _sample_count(rows)
    if "product_distribution" in checks:
        results["product_distribution"] = _product_distribution(rows)
    if "field_inventory" in checks:
        results["field_inventory"] = _field_inventory(
            fields, rows
        )
    if "target_missing" in checks:
        if metric_resolution["matches"]:
            results["target_missing"] = _missing_for_metrics(
                metric_resolution, rows
            )
        else:
            results["target_missing"] = {
                "status": "METRIC_NOT_RESOLVED",
                "message": (
                    "未从问题中确定具体性能指标；未擅自把多个性能字段合并。"
                ),
            }
    if "metric_stats" in checks:
        results["metric_stats"] = _metric_stats(
            metric_resolution, rows
        )
    if "field_coverage" in checks:
        results["field_coverage"] = _field_coverage(
            fields, rows
        )
    if "formula_sparsity" in checks:
        results["formula_sparsity"] = _formula_sparsity(
            fields, rows
        )
    if "constant_fields" in checks:
        results["constant_fields"] = _constant_formula_fields(
            fields, rows
        )
    if "duplicates" in checks:
        results["duplicates"] = _duplicates(
            fields, rows
        )
    if "outliers" in checks:
        results["outliers"] = _outliers(
            fields, rows, metric_resolution
        )
    if "units" in checks:
        results["units"] = _unit_check(repo)
    if "process_conditions" in checks:
        results["process_conditions"] = _process_conditions(manifest)
    if "modelability" in checks:
        results["modelability"] = _modelability(
            fields,
            rows,
            manifest=manifest,
            metric_resolution=metric_resolution,
        )
    if "quality_overview" in checks:
        results["quality_overview"] = {
            "sample_count": _sample_count(rows),
            "field_inventory": _field_inventory(fields, rows),
            "formula_sparsity": _formula_sparsity(
                fields, rows, top_n=10
            ),
            "constant_fields": _constant_formula_fields(
                fields, rows, top_n=10
            ),
            "duplicates": _duplicates(
                fields, rows, max_groups=5
            ),
            "outliers": _outliers(
                fields, rows, metric_resolution, top_n=8
            ),
            "units": _unit_check(repo),
            "modelability": _modelability(
                fields,
                rows,
                manifest=manifest,
                metric_resolution=metric_resolution,
            ),
        }

    return {
        "scope": {
            "dataset_id": manifest.get("dataset_id"),
            "product_type": product_type,
            "samples": len(rows),
            "canonical_source": (
                manifest.get("source") or {}
            ).get("canonical_source"),
        },
        "requested_checks": checks,
        "classification": classification,
        "metric_resolution": metric_resolution,
        "results": results,
        "boundaries": {
            "no_metric_merging": True,
            "statistical_outlier_is_not_data_error": True,
            "unknown_units_are_not_reported_as_zero_conflicts": True,
            "sparse_single_observation_is_not_called_constant": True,
            "simulator_results_included": False,
        },
    }


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def render_inspection_answer(
    inspection: dict[str, Any],
) -> str:
    scope = inspection["scope"]
    results = inspection["results"]
    product = scope.get("product_type")
    prefix = (
        f"{product}（{scope['samples']} 个真实样品）"
        if product
        else f"当前真实公司数据（{scope['samples']} 个样品）"
    )
    parts: list[str] = [prefix + "："]

    if "sample_count" in results:
        item = results["sample_count"]
        if product:
            parts.append(f"样品数 {item['samples']}。")
        else:
            parts.append(
                f"样品数 {item['samples']}，覆盖 {item['products']} 个产品类型。"
            )

    if "product_distribution" in results:
        item = results["product_distribution"]
        parts.append(
            f"产品类型 {item['product_type_count']} 个。"
        )
        if not product and item["products"]:
            parts.append(
                "样本量最多的产品："
                + "、".join(
                    f"{x['product_type']}({x['sample_count']})"
                    for x in item["products"][:10]
                )
                + "。"
            )

    if "field_inventory" in results:
        item = results["field_inventory"]
        parts.append(
            f"字段：配方 {item['formula_fields_total']} 个"
            f"（当前范围实际出现 {item['formula_fields_active']} 个），"
            f"性能 {item['performance_fields_total']} 个"
            f"（实际出现 {item['performance_fields_active']} 个）。"
        )

    if "target_missing" in results:
        item = results["target_missing"]
        if item.get("status") == "METRIC_NOT_RESOLVED":
            parts.append(item["message"])
        else:
            resolution = item["metric_resolution"]
            if resolution.get("ambiguous"):
                parts.append(
                    "注意：问题中的性能名称匹配多个真实字段，"
                    "以下分别统计，不做合并："
                )
            for metric in item["metrics"]:
                parts.append(
                    f"{metric['metric']}：非空 {metric['nonempty_count']}/"
                    f"{metric['samples']}，缺失 {metric['missing_count']}"
                    f"（{_pct(metric['missing_rate'])}），"
                    f"其中数值型 {metric['numeric_count']}。"
                )

    if "metric_stats" in results:
        item = results["metric_stats"]
        if item.get("status") == "METRIC_NOT_RESOLVED":
            parts.append(item["message"])
        else:
            if item["metric_resolution"].get("ambiguous"):
                parts.append(
                    "性能名称有多个匹配，统计量分别报告，不做合并。"
                )
            for metric in item["metrics"]:
                if metric["numeric_count"] == 0:
                    parts.append(
                        f"{metric['metric']}：没有可计算数值。"
                    )
                else:
                    parts.append(
                        f"{metric['metric']}：n={metric['numeric_count']}，"
                        f"最小 {metric['minimum']:.6g}，"
                        f"最大 {metric['maximum']:.6g}，"
                        f"平均 {metric['mean']:.6g}，"
                        f"中位数 {metric['median']:.6g}。"
                    )

    if "formula_sparsity" in results:
        item = results["formula_sparsity"]
        parts.append(
            f"配方稀疏性：按缺失率 ≥ "
            f"{_pct(item['threshold_missing_rate'])} 定义“高缺失”，"
            f"完全为空 {item['fully_empty_count']} 个，"
            f"高缺失但有记录 {item['sparse_active_count']} 个。"
        )
        top = item["sparse_active"][:8]
        if top:
            parts.append(
                "最高缺失字段(material_id)："
                + "、".join(
                    f"{x['material_id']}({_pct(x['missing_rate'])})"
                    for x in top
                )
                + "。"
            )

    if "constant_fields" in results:
        item = results["constant_fields"]
        parts.append(
            f"恒定配方字段：{item['constant_formula_count']} 个；"
            f"要求至少 {item['minimum_support_for_constant']} 条非空记录"
            "且只有一个唯一值。低覆盖单值字段不会被误判为恒定。"
        )
        top = item["constant_formula_fields"][:8]
        if top:
            parts.append(
                "示例(material_id)："
                + "、".join(
                    f"{x['material_id']}={x['constant_value']}"
                    for x in top
                )
                + "。"
            )

    if "duplicates" in results:
        item = results["duplicates"]
        parts.append(
            f"重复检查：重复 sample_name 组 "
            f"{item['duplicate_sample_name_group_count']}；"
            f"相同配方签名组 {item['duplicate_formula_group_count']}。"
            "相同配方不等同于重复录入。"
        )

    if "outliers" in results:
        item = results["outliers"]
        if item["scope"] == "REQUESTED_METRICS":
            for metric in item["metrics"]:
                parts.append(
                    f"{metric}：数值样本 {metric['numeric_count']}，"
                    f"统计异常候选 {metric['outlier_count']}，"
                    f"方法 {metric['method']}。"
                )
        else:
            parts.append(
                f"异常值扫描：检查 {item['metrics_checked']} 个"
                f"可计算性能指标，其中 {item['metrics_with_outliers']} 个"
                "出现统计异常候选。"
            )
        parts.append(
            "统计异常值只表示“建议复核”，不自动判定为错误数据。"
        )

    if "units" in results:
        item = results["units"]
        if not item["can_assert_consistency"]:
            parts.append(
                "单位一致性：当前无法可靠判断。"
                + item["reason"]
            )
        else:
            parts.append(
                f"单位一致性：单位元数据可用，当前冲突数 "
                f"{item['conflict_count']}。"
            )

    if "process_conditions" in results:
        item = results["process_conditions"]
        parts.append(
            f"工艺/条件覆盖：真实材料工艺参数行 "
            f"{item['material_process_parameter_rows']}，"
            f"显式测试条件行 {item['explicit_test_condition_rows']}；"
            f"工作流元数据行 {item['workflow_metadata_rows']}，"
            "但不会被当成材料工艺特征。"
        )

    if "modelability" in results:
        item = results["modelability"]
        parts.append(
            f"探索性可用配方特征约 {item['exploratory_formula_feature_count']} 个"
            "（覆盖率≥20%且至少两个数值水平）。"
            f"真实材料工艺参数行 {item['true_process_parameter_rows']}，"
            f"显式测试条件行 {item['explicit_test_condition_rows']}。"
            "因此导入数据本身仍不能绕过 Modeling Gate 成为正式模型。"
        )

    if "field_coverage" in results:
        item = results["field_coverage"]
        parts.append(
            "字段覆盖率已计算；结构化结果中提供配方/性能字段"
            "最低与最高覆盖列表。"
        )

    if "quality_overview" in results:
        q = results["quality_overview"]
        sparse = q["formula_sparsity"]
        const = q["constant_fields"]
        dup = q["duplicates"]
        units = q["units"]
        model = q["modelability"]
        parts.append(
            "数据体检摘要："
            f"高缺失配方字段 {sparse['sparse_active_count']}，"
            f"恒定配方字段 {const['constant_formula_count']}，"
            f"重复 sample_name 组 {dup['duplicate_sample_name_group_count']}，"
            f"单位检查 {units['status']}；"
            f"工艺参数行 {model['true_process_parameter_rows']}，"
            f"测试条件行 {model['explicit_test_condition_rows']}。"
        )

    parts.append(
        "以上只统计 canonical 公司真实历史数据；"
        "V0.3 Simulator synthetic 结果不计入。"
    )
    return " ".join(parts)
