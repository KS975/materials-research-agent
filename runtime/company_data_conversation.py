from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from company_data import CompanyDataRepository, CompanyDataValidationError
from runtime.company_data_inspection import (
    classify_company_data_request,
    resolve_product_from_text,
)


GLOBAL_SCOPE_MARKERS = (
    "全库",
    "全部真实数据",
    "所有真实数据",
    "全部公司数据",
    "所有公司数据",
    "全公司",
    "整个公司",
    "公司总库",
    "所有产品",
    "全部产品",
)

HISTORY_BREAK_MARKERS = (
    "v0.3",
    "v0.2",
    "自主实验",
    "自动实验",
    "scheduler",
    "telemetry",
    "safety",
    "crash",
    "project ",
    "项目",
    "附件",
    "pdf",
    "docx",
    "历史报告",
    "历史资料",
    "历史",
    "以前",
    "过去",
    "曾经",
    "类似案例",
    "类似问题",
    "类似情况",
    "mysql",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def asks_for_global_company_scope(message: str) -> bool:
    lowered = _text(message).casefold()
    return any(
        marker.casefold() in lowered
        for marker in GLOBAL_SCOPE_MARKERS
    )


def breaks_company_product_context(message: str) -> bool:
    lowered = _text(message).casefold()
    return any(
        marker.casefold() in lowered
        for marker in HISTORY_BREAK_MARKERS
    )


def _history_items(
    history: Iterable[Any],
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = _text(item.get("role"))
            content = _text(item.get("content"))
        else:
            role = _text(getattr(item, "role", ""))
            content = _text(getattr(item, "content", ""))
        if role and content:
            items.append((role, content))
    return items


def resolve_company_conversation_scope(
    runtime_root: str | Path,
    *,
    message: str,
    history: Iterable[Any] = (),
) -> dict[str, Any]:
    """Resolve current company-data product scope deterministically."""
    repo = CompanyDataRepository(runtime_root)

    if asks_for_global_company_scope(message):
        return {
            "product_type": None,
            "source": "GLOBAL_RESET",
            "inherited": False,
            "scope_label": "全部真实数据",
        }

    try:
        current_product = resolve_product_from_text(
            repo, message
        )
    except CompanyDataValidationError:
        current_product = None

    if current_product is not None:
        return {
            "product_type": current_product["product_type"],
            "source": "CURRENT_MESSAGE",
            "inherited": False,
            "scope_label": current_product["product_type"],
        }

    for role, content in reversed(_history_items(history)):
        if role != "user":
            continue
        if asks_for_global_company_scope(content):
            break
        if breaks_company_product_context(content):
            break
        try:
            product = resolve_product_from_text(
                repo, content
            )
        except CompanyDataValidationError:
            product = None
        if product is not None:
            return {
                "product_type": product["product_type"],
                "source": "HISTORY",
                "inherited": True,
                "scope_label": product["product_type"],
            }

    return {
        "product_type": None,
        "source": "GLOBAL_DEFAULT",
        "inherited": False,
        "scope_label": "全部真实数据",
    }


def classify_company_data_turn(
    runtime_root: str | Path,
    *,
    message: str,
    history: Iterable[Any] = (),
) -> dict[str, Any]:
    """Classify the current turn, using product context only when needed."""
    direct = classify_company_data_request(
        message,
        runtime_root=runtime_root,
    )
    scope = resolve_company_conversation_scope(
        runtime_root,
        message=message,
        history=history,
    )

    if direct["route"]:
        return {
            **direct,
            "conversation_scope": scope,
            "classification_message": message,
            "contextualized": False,
        }

    product_type = scope.get("product_type")
    if product_type:
        contextual_message = (
            f"{product_type} 数据 {message}"
        )
        contextual = classify_company_data_request(
            contextual_message,
            runtime_root=runtime_root,
        )
        if contextual["route"]:
            return {
                **contextual,
                "conversation_scope": scope,
                "classification_message": contextual_message,
                "contextualized": True,
                "original_message": message,
            }

    return {
        **direct,
        "conversation_scope": scope,
        "classification_message": message,
        "contextualized": False,
    }


def company_data_has_priority(decision: dict[str, Any]) -> bool:
    """Return whether imported company data may preempt normal chat routing.

    The Haike/imported runtime is intentionally opt-in in mixed chat. It keeps
    priority for an explicit source request, a known/inherited imported product,
    or an explicit reset of such a product conversation to the full catalogue.
    Generic global matches remain available to callers but no longer preempt
    business-MySQL material intents.
    """
    if not decision.get("route"):
        return False
    if decision.get("explicit_company_scope"):
        return True
    scope = decision.get("conversation_scope") or {}
    if str(scope.get("product_type") or "").strip():
        return True
    return scope.get("source") == "GLOBAL_RESET"
