from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

from .campaign import (
    CampaignStore,
    _append_event,
    _touch,
    find_round,
    utc_now_iso,
)

EXPERIMENT_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "INVALID", "NOT_TESTED"}
EXPERIMENT_STATUSES = {"PLANNED", *EXPERIMENT_TERMINAL_STATUSES}


class ExperimentalResultError(RuntimeError):
    pass


class ExperimentalResultValidationError(ExperimentalResultError):
    pass


class ExperimentalResultConflictError(ExperimentalResultError):
    pass


class ExperimentalResultNotFoundError(ExperimentalResultError):
    pass


def _text(value: Any, name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ExperimentalResultValidationError(f"{name} 不能为空")
    return value


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ExperimentalResultValidationError(
            f"{name} 必须是可序列化 JSON 数据"
        ) from exc


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentalResultValidationError(f"{name} 必须是数值")
    value = float(value)
    if not math.isfinite(value):
        raise ExperimentalResultValidationError(f"{name} 必须是有限数值")
    return value


def normalize_planned_experiment(
    item: dict[str, Any],
    *,
    campaign_target_metrics: list[str],
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ExperimentalResultValidationError("planned experiment 必须是 JSON object")

    candidate_id = _text(item.get("candidate_id"), "candidate_id")
    condition = _text(
        item.get("expected_test_condition_signature"),
        "expected_test_condition_signature",
    )

    required_metrics = item.get("required_metrics", campaign_target_metrics)
    if not isinstance(required_metrics, list) or not required_metrics:
        raise ExperimentalResultValidationError(
            f"{candidate_id}.required_metrics 必须是非空 list"
        )

    clean_metrics = []
    seen = set()
    for metric in required_metrics:
        metric = _text(metric, f"{candidate_id}.required_metric")
        if metric not in campaign_target_metrics:
            raise ExperimentalResultValidationError(
                f"{candidate_id}: 指标不属于 campaign target_metrics: {metric}"
            )
        if metric not in seen:
            seen.add(metric)
            clean_metrics.append(metric)

    units = item.get("units") or {}
    if not isinstance(units, dict):
        raise ExperimentalResultValidationError(
            f"{candidate_id}.units 必须是 JSON object"
        )

    clean_units = {}
    for metric in clean_metrics:
        unit = str(units.get(metric) or "").strip()
        if unit:
            clean_units[metric] = unit

    return {
        "candidate_id": candidate_id,
        "status": "PLANNED",
        "required_metrics": clean_metrics,
        "expected_test_condition_signature": condition,
        "units": clean_units,
        "features": _json_copy(item.get("features") or {}, f"{candidate_id}.features"),
        "prediction_snapshot": _json_copy(
            item.get("prediction_snapshot") or {},
            f"{candidate_id}.prediction_snapshot",
        ),
        "registered_at": utc_now_iso(),
        "result": None,
    }


def _recompute_round_progress(round_record: dict[str, Any]) -> dict[str, int]:
    experiments = round_record.get("experiments")
    if experiments is None:
        return deepcopy(round_record.get("progress") or {})

    progress = {
        "planned": len(experiments),
        "completed": 0,
        "failed": 0,
        "invalid": 0,
        "not_tested": 0,
        "pending": 0,
        "terminal": 0,
        "training_eligible": 0,
    }

    for item in experiments:
        status = item.get("status")
        if status == "PLANNED":
            progress["pending"] += 1
            continue

        progress["terminal"] += 1
        if status == "COMPLETED":
            progress["completed"] += 1
            if item.get("result") and item["result"].get("training_eligible") is True:
                progress["training_eligible"] += 1
        elif status == "FAILED":
            progress["failed"] += 1
        elif status == "INVALID":
            progress["invalid"] += 1
        elif status == "NOT_TESTED":
            progress["not_tested"] += 1

    round_record["progress"] = progress
    return deepcopy(progress)


def register_planned_experiments(
    campaign: dict[str, Any],
    *,
    round_id: str,
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    round_record = find_round(campaign, round_id)

    if round_record.get("status") != "PLANNED":
        raise ExperimentalResultConflictError(
            "只能在 PLANNED Round 注册实验计划"
        )
    if round_record.get("experiments") is not None:
        raise ExperimentalResultConflictError(
            "该 Round 已注册实验计划，不能重复覆盖"
        )
    if not isinstance(experiments, list) or not experiments:
        raise ExperimentalResultValidationError("experiments 必须是非空 list")

    expected_count = int(round_record["plan"]["planned_experiment_count"])
    if len(experiments) != expected_count:
        raise ExperimentalResultValidationError(
            f"实验数量与 Round plan 不一致: expected={expected_count}, got={len(experiments)}"
        )

    normalized = [
        normalize_planned_experiment(
            item,
            campaign_target_metrics=list(campaign.get("target_metrics") or []),
        )
        for item in experiments
    ]
    ids = [item["candidate_id"] for item in normalized]
    if len(set(ids)) != len(ids):
        raise ExperimentalResultValidationError(
            "planned experiments 中 candidate_id 重复"
        )

    round_record["experiments"] = normalized
    _recompute_round_progress(round_record)
    _touch(campaign)
    _append_event(
        campaign,
        event_type="ROUND_EXPERIMENTS_REGISTERED",
        payload={
            "round_id": round_id,
            "candidate_ids": ids,
            "planned_experiment_count": len(ids),
        },
    )
    return deepcopy(normalized)


def find_experiment(
    campaign: dict[str, Any],
    *,
    round_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    round_record = find_round(campaign, round_id)
    candidate_id = _text(candidate_id, "candidate_id")
    experiments = round_record.get("experiments")
    if experiments is None:
        raise ExperimentalResultNotFoundError(
            "该 Round 尚未注册 planned experiments"
        )
    for item in experiments:
        if item.get("candidate_id") == candidate_id:
            return item
    raise ExperimentalResultNotFoundError(
        f"candidate_id 不属于当前 Round: {candidate_id}"
    )


def _normalize_result_payload(
    experiment: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ExperimentalResultValidationError("result payload 必须是 JSON object")

    candidate_id = _text(payload.get("candidate_id"), "candidate_id")
    if candidate_id != experiment["candidate_id"]:
        raise ExperimentalResultConflictError(
            "payload candidate_id 与 experiment 不一致"
        )

    status = str(payload.get("status") or "").strip().upper()
    if status not in EXPERIMENT_TERMINAL_STATUSES:
        raise ExperimentalResultValidationError(
            "result status 必须是 COMPLETED / FAILED / INVALID / NOT_TESTED"
        )

    condition = _text(
        payload.get("test_condition_signature"),
        "test_condition_signature",
    )
    expected_condition = experiment["expected_test_condition_signature"]
    if condition != expected_condition:
        raise ExperimentalResultConflictError(
            f"测试条件不一致: expected={expected_condition}, got={condition}"
        )

    measurements = payload.get("measurements")
    if measurements is None:
        measurements = {}
    if not isinstance(measurements, dict):
        raise ExperimentalResultValidationError("measurements 必须是 JSON object")

    units = payload.get("units")
    if units is None:
        units = {}
    if not isinstance(units, dict):
        raise ExperimentalResultValidationError("units 必须是 JSON object")

    required_metrics = list(experiment["required_metrics"])
    expected_units = dict(experiment.get("units") or {})

    clean_measurements = {}
    clean_units = {}

    if status == "COMPLETED":
        missing = [
            metric for metric in required_metrics
            if metric not in measurements or measurements.get(metric) is None
        ]
        if missing:
            raise ExperimentalResultValidationError(
                "COMPLETED 实验缺少必需指标: " + ", ".join(missing)
            )

        extras = [metric for metric in measurements if metric not in required_metrics]
        if extras:
            raise ExperimentalResultValidationError(
                "result 包含未声明指标: " + ", ".join(extras)
            )

        for metric in required_metrics:
            clean_measurements[metric] = _finite_number(
                measurements[metric],
                f"measurements[{metric}]",
            )
            expected_unit = str(expected_units.get(metric) or "").strip()
            supplied_unit = str(units.get(metric) or "").strip()
            if expected_unit and not supplied_unit:
                raise ExperimentalResultValidationError(f"缺少单位: {metric}")
            if expected_unit and supplied_unit != expected_unit:
                raise ExperimentalResultConflictError(
                    f"{metric} 单位不一致: expected={expected_unit}, got={supplied_unit}"
                )
            if supplied_unit:
                clean_units[metric] = supplied_unit
    else:
        nonempty = {
            key: value for key, value in measurements.items()
            if value is not None
        }
        if nonempty:
            raise ExperimentalResultValidationError(
                f"{status} 实验不能携带实测性能值；失败/无效/未测试不能伪装成 0 或其他数值"
            )

    failure_reason = str(payload.get("failure_reason") or "").strip()
    if status in {"FAILED", "INVALID"} and not failure_reason:
        raise ExperimentalResultValidationError(
            f"{status} 必须填写 failure_reason"
        )

    return {
        "candidate_id": candidate_id,
        "status": status,
        "test_condition_signature": condition,
        "measurements": clean_measurements,
        "units": clean_units,
        "failure_reason": failure_reason,
        "notes": str(payload.get("notes") or "").strip(),
        "training_eligible": status == "COMPLETED",
    }


def ingest_experimental_result(
    campaign: dict[str, Any],
    *,
    round_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    round_record = find_round(campaign, round_id)
    if round_record.get("status") not in {"RUNNING", "PARTIALLY_COMPLETED"}:
        raise ExperimentalResultConflictError(
            "只有 RUNNING / PARTIALLY_COMPLETED Round 可以接收实验结果"
        )

    candidate_id = _text(payload.get("candidate_id"), "candidate_id")
    experiment = find_experiment(
        campaign,
        round_id=round_id,
        candidate_id=candidate_id,
    )
    normalized = _normalize_result_payload(experiment, payload)
    existing = experiment.get("result")

    if existing is not None:
        comparable = {
            key: existing.get(key)
            for key in (
                "candidate_id",
                "status",
                "test_condition_signature",
                "measurements",
                "units",
                "failure_reason",
                "notes",
                "training_eligible",
            )
        }
        if comparable == normalized:
            return {
                "idempotent_replay": True,
                "experiment": deepcopy(experiment),
                "round_progress": _recompute_round_progress(round_record),
            }
        raise ExperimentalResultConflictError(
            f"candidate_id 已有不同实验结果，拒绝冲突覆盖: {candidate_id}"
        )

    experiment["status"] = normalized["status"]
    experiment["result"] = {
        **normalized,
        "submitted_at": utc_now_iso(),
    }
    progress = _recompute_round_progress(round_record)

    if (
        round_record.get("status") == "RUNNING"
        and progress["terminal"] > 0
        and progress["pending"] > 0
    ):
        old_status = round_record["status"]
        round_record["status"] = "PARTIALLY_COMPLETED"
        _append_event(
            campaign,
            event_type="ROUND_STATUS_CHANGED",
            payload={
                "round_id": round_id,
                "from_status": old_status,
                "to_status": "PARTIALLY_COMPLETED",
                "reason": "first experimental result ingested",
            },
        )

    _touch(campaign)
    _append_event(
        campaign,
        event_type="EXPERIMENT_RESULT_INGESTED",
        payload={
            "round_id": round_id,
            "candidate_id": candidate_id,
            "status": normalized["status"],
            "training_eligible": normalized["training_eligible"],
        },
    )

    return {
        "idempotent_replay": False,
        "experiment": deepcopy(experiment),
        "round_progress": deepcopy(progress),
    }


def round_result_summary(
    campaign: dict[str, Any],
    *,
    round_id: str,
) -> dict[str, Any]:
    round_record = find_round(campaign, round_id)
    progress = _recompute_round_progress(round_record)
    experiments = round_record.get("experiments") or []

    return {
        "round_id": round_id,
        "round_status": round_record.get("status"),
        "progress": progress,
        "training_eligible_candidate_ids": [
            item["candidate_id"]
            for item in experiments
            if item.get("result") and item["result"].get("training_eligible") is True
        ],
        "failed_candidate_ids": [
            item["candidate_id"] for item in experiments
            if item.get("status") == "FAILED"
        ],
        "invalid_candidate_ids": [
            item["candidate_id"] for item in experiments
            if item.get("status") == "INVALID"
        ],
        "not_tested_candidate_ids": [
            item["candidate_id"] for item in experiments
            if item.get("status") == "NOT_TESTED"
        ],
        "can_close_round": bool(experiments) and progress["pending"] == 0,
    }


class ExperimentalResultService:
    def __init__(self, runtime_root: str = ".runtime") -> None:
        self.store = CampaignStore(runtime_root)

    def register_planned_experiments(
        self,
        campaign_id: str,
        *,
        round_id: str,
        experiments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        campaign = self.store.load(campaign_id)
        result = register_planned_experiments(
            campaign,
            round_id=round_id,
            experiments=experiments,
        )
        self.store.save(campaign)
        return result

    def ingest(
        self,
        campaign_id: str,
        *,
        round_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        campaign = self.store.load(campaign_id)
        result = ingest_experimental_result(
            campaign,
            round_id=round_id,
            payload=payload,
        )
        self.store.save(campaign)
        return result

    def summary(
        self,
        campaign_id: str,
        *,
        round_id: str,
    ) -> dict[str, Any]:
        campaign = self.store.load(campaign_id)
        result = round_result_summary(
            campaign,
            round_id=round_id,
        )
        self.store.save(campaign)
        return result
