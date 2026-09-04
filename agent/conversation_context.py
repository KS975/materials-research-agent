from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CONVERSATION_CONTEXT_SCHEMA_VERSION = "2.1.2"


_PROJECT_RE = re.compile(
    # A hyphen immediately before digits is part of a negative project ID,
    # not punctuation.  This matters for history-imported projects such as
    # Project -1659.
    r"(?:project|项目)\s*[:#：]?\s*(-?\d+)",
    re.IGNORECASE,
)
_NUMERIC_ID_RE = re.compile(r"(?<![\d.])-?\d{3,}(?![\d.])")
_SAMPLE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?=[A-Za-z0-9_.-]{3,}(?![A-Za-z0-9_.-]))"
    r"(?=[A-Za-z0-9_.-]*\d)"
    r"[A-Za-z][A-Za-z0-9_.-]*"
)
_CORRECTION_RE = re.compile(
    r"(?:不是|不要)\s*([A-Za-z0-9_.-]+)\s*[，,、]?\s*"
    r"(?:是|改成|换成)\s*([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)

_METRIC_MARKERS = (
    "冲击强度",
    "拉伸强度",
    "弯曲强度",
    "断裂伸长率",
    "杨氏模量",
    "硬度",
    "密度",
    "MFR",
    "MFI",
    "熔指",
    "离子电导率",
    "室温离子电导率",
    "电化学窗口",
    "致密度",
    "孔隙率",
)

_HISTORY_MARKERS = (
    "历史",
    "以前",
    "过去",
    "曾经",
    "以往",
    "历史报告",
    "历史资料",
    "历史案例",
)
_HISTORY_SIMILARITY_MARKERS = (
    "有没有类似",
    "类似问题",
    "类似情况",
    "类似案例",
    "相似案例",
)
_REFERENTIAL_MARKERS = (
    "这个样品",
    "该样品",
    "这个",
    "那个",
    "它",
    "刚才那个",
    "前面那个",
)
_PAIR_REFERENTIAL_MARKERS = (
    "这两个样品",
    "这两个样本",
    "这两个",
    "两个样品",
    "两个样本",
    "两者",
    "二者",
    "它们",
    "刚才比较的两个",
    "前面比较的两个",
)
_SCOPE_RESET_MARKERS = (
    "全部项目",
    "所有项目",
    "全公司",
    "全部历史",
    "所有历史",
    "不限定项目",
    "不限项目",
    "不限制项目",
    "全部范围",
)


@dataclass(frozen=True, slots=True)
class ConversationHints:
    current_sample_identifiers: tuple[str, ...]
    recent_sample_identifiers: tuple[str, ...]
    current_project_ids: tuple[int, ...]
    recent_project_ids: tuple[int, ...]
    current_metrics: tuple[str, ...]
    recent_metrics: tuple[str, ...]
    action_hint: str
    correction: dict[str, str] | None
    task_refinement_intent: str | None

    # V2.1 task-level context. These are deterministic hints derived only from
    # the user turns, never hidden model state.
    active_sample_identifier: str | None
    active_comparison_identifiers: tuple[str, ...]
    current_pair_referential: bool
    active_metric: str | None
    current_history_request: bool
    current_history_referential: bool
    scope_only_followup: bool
    scope_reset_followup: bool
    active_history_task: str | None
    active_history_query: str
    active_history_sample_identifiers: tuple[str, ...]
    active_history_metric: str | None
    active_history_project_ids: tuple[int, ...]
    effective_history_query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_sample_identifiers": list(self.current_sample_identifiers),
            "recent_sample_identifiers": list(self.recent_sample_identifiers),
            "current_project_ids": list(self.current_project_ids),
            "recent_project_ids": list(self.recent_project_ids),
            "current_metrics": list(self.current_metrics),
            "recent_metrics": list(self.recent_metrics),
            "conversation_action_hint": self.action_hint,
            "correction": dict(self.correction) if self.correction else None,
            "task_refinement_intent": self.task_refinement_intent,
            "active_sample_identifier": self.active_sample_identifier,
            "active_comparison_identifiers": list(
                self.active_comparison_identifiers
            ),
            "current_pair_referential": self.current_pair_referential,
            "active_metric": self.active_metric,
            "current_history_request": self.current_history_request,
            "current_history_referential": self.current_history_referential,
            "scope_only_followup": self.scope_only_followup,
            "scope_reset_followup": self.scope_reset_followup,
            "active_history_task": self.active_history_task,
            "active_history_query": self.active_history_query,
            "active_history_sample_identifiers": list(
                self.active_history_sample_identifiers
            ),
            "active_history_metric": self.active_history_metric,
            "active_history_project_ids": list(self.active_history_project_ids),
            "effective_history_query": self.effective_history_query,
        }


def _dedupe_keep_order(values):
    seen = set()
    result = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _project_ids(text: str) -> list[int]:
    result: list[int] = []
    for match in _PROJECT_RE.finditer(text or ""):
        try:
            result.append(int(match.group(1)))
        except ValueError:
            continue
    return _dedupe_keep_order(result)


def _sample_identifiers(text: str) -> list[str]:
    text = str(text or "")
    project_spans = [m.span(1) for m in _PROJECT_RE.finditer(text)]

    def inside_project_span(start: int, end: int) -> bool:
        return any(start >= a and end <= b for a, b in project_spans)

    values: list[str] = []
    for match in _SAMPLE_TOKEN_RE.finditer(text):
        token = match.group(0).strip(".,，。；;:：")
        lowered = token.lower()
        if lowered.startswith(("v0.", "project")):
            continue
        if token:
            values.append(token)

    for match in _NUMERIC_ID_RE.finditer(text):
        if inside_project_span(*match.span()):
            continue
        token = match.group(0)
        # Common years are weak sample hints and create many false references.
        if re.fullmatch(r"20\d{2}", token):
            continue
        values.append(token)

    return _dedupe_keep_order(values)


def _metrics(text: str) -> list[str]:
    text = str(text or "")
    found = [metric for metric in _METRIC_MARKERS if metric.lower() in text.lower()]
    if "室温离子电导率" in found and "离子电导率" in found:
        found.remove("离子电导率")
    return _dedupe_keep_order(found)


def _is_history_request(text: str) -> bool:
    value = str(text or "")
    return any(marker in value for marker in _HISTORY_MARKERS) or any(
        marker in value for marker in _HISTORY_SIMILARITY_MARKERS
    )


def _has_referential_sample(text: str) -> bool:
    value = str(text or "")
    return any(marker in value for marker in _REFERENTIAL_MARKERS)


def _has_referential_pair(text: str) -> bool:
    value = str(text or "")
    return any(marker in value for marker in _PAIR_REFERENTIAL_MARKERS)


def _is_scope_reset_followup(text: str) -> bool:
    value = str(text or "").strip()
    if not any(marker in value for marker in _SCOPE_RESET_MARKERS):
        return False
    # A reset phrase may itself include “历史”, but should not replace the task.
    substantive = (
        "配方", "工艺", "性能", "冲击", "拉伸", "比较", "对比",
        "为什么", "原因", "异常原因", "样品", "样本",
    )
    return not any(marker in value for marker in substantive)


def _is_scope_only_project_followup(text: str) -> bool:
    value = str(text or "").strip()
    if not _project_ids(value):
        return False
    if _is_history_request(value):
        return False

    stripped = _PROJECT_RE.sub("", value)
    stripped = re.sub(r"[\s，,。.!！?？:：;；()（）\[\]【】]", "", stripped)
    for token in (
        "那", "那么", "呢", "只看", "只查", "就看", "就查", "限定",
        "范围", "改成", "换成", "看看", "看下", "查一下", "这个",
    ):
        stripped = stripped.replace(token, "")
    return stripped == ""


def _history_task(
    text: str,
    samples: list[str],
    active_sample: str | None,
) -> str:
    if len(samples) >= 2:
        return "joint_mysql_knowledge_analysis"
    if len(samples) == 1:
        return "sample_historical_similarity"
    if _has_referential_sample(text) and active_sample:
        return "sample_historical_similarity"
    return "search_historical_knowledge"


def _canonicalize_history_query(
    text: str,
    *,
    sample_identifier: str | None,
) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    if sample_identifier and _has_referential_sample(value):
        # Replace only the first suitable referent. Keep the user's wording so
        # the embedding query still reflects the original materials problem.
        for marker in _REFERENTIAL_MARKERS:
            if marker in value:
                value = value.replace(marker, f"样品{sample_identifier}", 1)
                break
    return value


def _action_hint(message: str) -> str:
    text = str(message or "").strip()
    if _is_scope_only_project_followup(text) or _is_scope_reset_followup(text):
        return "refine_previous"
    if _CORRECTION_RE.search(text) or any(
        marker in text for marker in ("改成", "换成", "更正", "纠正")
    ):
        return "user_correction"
    if any(marker in text for marker in ("只看", "只比较", "只要", "范围缩小", "范围扩大")):
        return "refine_previous"
    if text in {"继续", "继续吧", "接着", "接着来", "还有吗", "再来"} or text.startswith(("继续", "再找", "再给", "接着")):
        return "continue_previous"
    if any(marker in text for marker in ("这个", "那个", "它", "刚才", "前面", "上一个")):
        return "follow_up_reference"
    return "new_request"


def build_conversation_hints(
    message: str,
    history: list[dict[str, str]] | None,
) -> ConversationHints:
    current_samples = _sample_identifiers(message)
    current_projects = _project_ids(message)
    current_metrics = _metrics(message)

    recent_samples: list[str] = []
    recent_projects: list[int] = []
    recent_metrics: list[str] = []

    user_turns = [
        str(item.get("content") or "")
        for item in (history or [])
        if str(item.get("role") or "") == "user"
    ][-12:]

    # Keep the existing newest-first inventories for compatibility.
    for text in reversed(user_turns[-8:]):
        recent_samples.extend(_sample_identifiers(text))
        recent_projects.extend(_project_ids(text))
        recent_metrics.extend(_metrics(text))

    # Build a deterministic task state in chronological order. The most recent
    # explicit user sample becomes the active sample even when older turns
    # mention other samples; this fixes the fragile “len(recent)==1” behavior.
    active_sample: str | None = None
    active_comparison_samples: list[str] = []
    active_metric: str | None = None
    active_history_task: str | None = None
    active_history_query = ""
    active_history_samples: list[str] = []
    active_history_metric: str | None = None
    active_history_projects: list[int] = []

    for text in user_turns:
        samples = _sample_identifiers(text)
        metrics = _metrics(text)
        correction_match = _CORRECTION_RE.search(text)
        if correction_match:
            active_sample = correction_match.group(2)
        elif samples:
            # For ordinary follow-ups, the last explicitly named sample is the
            # best single referent. Multi-sample history tasks keep their own list.
            active_sample = samples[-1]
        if len(samples) >= 2:
            # The latest explicitly named pair is the authoritative comparison
            # target for follow-ups such as “这两个样品的工艺有什么区别”.
            active_comparison_samples = list(samples[:2])
        elif len(samples) == 1:
            # A later explicit single-sample task ends the previous pair scope;
            # never revive a stale pair after the user has changed subjects.
            active_comparison_samples = []
        if metrics:
            active_metric = metrics[-1]

        if _is_history_request(text):
            task = _history_task(text, samples, active_sample)
            history_samples = list(samples)
            if (
                task == "sample_historical_similarity"
                and not history_samples
                and active_sample
            ):
                history_samples = [active_sample]
            active_history_task = task
            active_history_samples = history_samples
            active_history_metric = metrics[-1] if metrics else active_metric
            active_history_projects = _project_ids(text)
            active_history_query = _canonicalize_history_query(
                text,
                sample_identifier=(history_samples[0] if len(history_samples) == 1 else None),
            )

    correction = None
    match = _CORRECTION_RE.search(str(message or ""))
    if match:
        correction = {"from": match.group(1), "to": match.group(2)}

    # Current explicit/corrected sample wins over prior state.
    resolved_active_sample = active_sample
    resolved_active_comparison = list(active_comparison_samples)
    if correction:
        resolved_active_sample = correction["to"]
    elif current_samples:
        resolved_active_sample = current_samples[-1]
    if len(current_samples) >= 2:
        resolved_active_comparison = list(current_samples[:2])
    elif len(current_samples) == 1:
        resolved_active_comparison = []

    resolved_metric = current_metrics[-1] if current_metrics else active_metric
    current_history_request = _is_history_request(message)
    current_history_referential = (
        current_history_request and _has_referential_sample(message)
    )
    current_pair_referential = _has_referential_pair(message)
    scope_only_followup = _is_scope_only_project_followup(message)
    scope_reset_followup = _is_scope_reset_followup(message)

    effective_history_query = ""
    if current_history_request:
        effective_history_query = _canonicalize_history_query(
            message,
            sample_identifier=(
                resolved_active_sample if current_history_referential else None
            ),
        )
    elif (scope_only_followup or scope_reset_followup) and active_history_query:
        effective_history_query = active_history_query

    current_text = str(message or "").strip()
    task_refinement_intent = None
    if any(marker in current_text for marker in ("只看", "只查看", "只想看", "看看", "看一下")):
        if "配方" in current_text:
            task_refinement_intent = "get_formula"
        elif any(marker in current_text for marker in ("工艺", "流程", "加工")):
            task_refinement_intent = "get_process"
        elif "性能" in current_text:
            task_refinement_intent = "get_performance"

    return ConversationHints(
        current_sample_identifiers=tuple(_dedupe_keep_order(current_samples)),
        recent_sample_identifiers=tuple(_dedupe_keep_order(recent_samples)),
        current_project_ids=tuple(_dedupe_keep_order(current_projects)),
        recent_project_ids=tuple(_dedupe_keep_order(recent_projects)),
        current_metrics=tuple(_dedupe_keep_order(current_metrics)),
        recent_metrics=tuple(_dedupe_keep_order(recent_metrics)),
        action_hint=_action_hint(message),
        correction=correction,
        task_refinement_intent=task_refinement_intent,
        active_sample_identifier=resolved_active_sample,
        active_comparison_identifiers=tuple(resolved_active_comparison),
        current_pair_referential=current_pair_referential,
        active_metric=resolved_metric,
        current_history_request=current_history_request,
        current_history_referential=current_history_referential,
        scope_only_followup=scope_only_followup,
        scope_reset_followup=scope_reset_followup,
        active_history_task=active_history_task,
        active_history_query=active_history_query,
        active_history_sample_identifiers=tuple(active_history_samples),
        active_history_metric=active_history_metric,
        active_history_project_ids=tuple(active_history_projects),
        effective_history_query=effective_history_query,
    )
