from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from experiments import CampaignStore, CrashResumeCoordinator, find_round
from runtime.v020_ui import (
    V020UIError,
    build_campaign_overview,
)


class V030UIError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V030UIError(f"无法读取 V0.3 运行文件: {path}: {exc}") from exc
    return data if isinstance(data, dict) else None


def _round_no(round_id: str) -> int:
    match = re.search(r"-R(\d+)$", str(round_id or ""))
    return int(match.group(1)) if match else -1


def _campaign_paths(runtime_root: Path) -> list[Path]:
    root = runtime_root / "v020" / "campaigns"
    if not root.is_dir():
        return []
    return sorted(root.glob("*/campaign.json"))


def _has_v030_runtime(runtime_root: Path, campaign_id: str) -> bool:
    roots = (
        runtime_root / "v030" / "autonomous_round" / campaign_id,
        runtime_root / "v030" / "autonomous_loop" / campaign_id,
        runtime_root / "v030" / "result_capture" / campaign_id,
        runtime_root / "v030" / "crash_resume" / campaign_id,
    )
    return any(path.exists() for path in roots)


def latest_v030_campaign_id_for_project(
    runtime_root: str | Path,
    project_id: int,
) -> str:
    root = Path(runtime_root)
    matches: list[tuple[str, float, str]] = []
    for path in _campaign_paths(root):
        data = _read_json(path)
        if not data:
            continue
        if int(data.get("project_id", -1)) != int(project_id):
            continue
        campaign_id = str(data.get("campaign_id") or "")
        if not campaign_id:
            continue
        if not (
            campaign_id.upper().startswith("V030_")
            or _has_v030_runtime(root, campaign_id)
        ):
            continue
        matches.append(
            (
                str(data.get("updated_at") or ""),
                path.stat().st_mtime,
                campaign_id,
            )
        )
    if not matches:
        raise V030UIError(
            f"Project {project_id} 尚无 V0.3 autonomous runtime"
        )
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return matches[-1][2]


def _scheduler_view(
    root: Path,
    campaign_id: str,
    round_id: str,
) -> dict[str, Any] | None:
    scheduler_root = root / "v030" / "scheduler"
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    if scheduler_root.is_dir():
        for path in scheduler_root.glob("*/scheduler.json"):
            data = _read_json(path)
            if not data:
                continue
            jobs_raw = data.get("jobs") or {}
            jobs = (
                list(jobs_raw.values())
                if isinstance(jobs_raw, dict)
                else list(jobs_raw)
                if isinstance(jobs_raw, list)
                else []
            )
            belongs = False
            for job in jobs:
                source_context = (
                    (job.get("protocol") or {}).get("source_context") or {}
                )
                if (
                    source_context.get("campaign_id") == campaign_id
                    and source_context.get("round_id") == round_id
                ):
                    belongs = True
                    break
            if not belongs and round_id not in path.parent.name:
                continue
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, path, data = candidates[-1]
    jobs_raw = data.get("jobs") or {}
    jobs = (
        list(jobs_raw.values())
        if isinstance(jobs_raw, dict)
        else list(jobs_raw)
        if isinstance(jobs_raw, list)
        else []
    )
    statuses = (
        "QUEUED",
        "DISPATCHED",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
    )
    counts = {
        status: sum(
            1 for job in jobs
            if str(job.get("status") or "").upper() == status
        )
        for status in statuses
    }
    active = next(
        (
            job for job in jobs
            if str(job.get("status") or "").upper()
            in {"RUNNING", "DISPATCHED"}
        ),
        None,
    )
    if active is None:
        active = next(
            (
                job for job in jobs
                if str(job.get("status") or "").upper() == "QUEUED"
            ),
            None,
        )
    return {
        "scheduler_id": data.get("scheduler_id"),
        "counts": counts,
        "job_count": len(jobs),
        "active_job": (
            {
                "scheduler_job_id": active.get("scheduler_job_id"),
                "adapter_job_id": active.get("adapter_job_id"),
                "candidate_id": active.get("candidate_id"),
                "device_id": active.get("device_id"),
                "status": active.get("status"),
                "elapsed_ticks": active.get("elapsed_ticks"),
                "timeout_ticks": active.get("timeout_ticks"),
                "attempts_started": active.get("attempts_started"),
            }
            if active else None
        ),
        "state_path": str(path),
    }


def _telemetry_view(
    root: Path,
    campaign_id: str,
    round_id: str,
) -> dict[str, Any] | None:
    telemetry_root = root / "v030" / "telemetry"
    sessions: list[tuple[float, Path, dict[str, Any]]] = []
    if telemetry_root.is_dir():
        for path in telemetry_root.glob("*/session.json"):
            if campaign_id not in path.parent.name or round_id not in path.parent.name:
                continue
            data = _read_json(path)
            if data:
                sessions.append((path.stat().st_mtime, path, data))
    if not sessions:
        return None
    sessions.sort(key=lambda item: item[0])
    _, latest_path, latest = sessions[-1]
    rows = list(latest.get("telemetry") or [])
    latest_row = rows[-1] if rows else None
    completed = sum(
        1
        for _, _, data in sessions
        if (data.get("phase_history") or [])[-1:] == ["COMPLETED"]
    )
    return {
        "session_count": len(sessions),
        "completed_sessions": completed,
        "latest": (
            {
                "session_id": latest.get("session_id"),
                "experiment_id": latest.get("experiment_id"),
                "device_id": latest.get("device_id"),
                "phase": (
                    latest_row.get("phase")
                    if latest_row
                    else (
                        (latest.get("phase_history") or [None])[-1]
                    )
                ),
                "progress_percent": (
                    latest_row.get("progress_percent")
                    if latest_row else None
                ),
                "elapsed_ticks": (
                    latest_row.get("elapsed_ticks")
                    if latest_row else None
                ),
                "temperature_c": (
                    latest_row.get("temperature_c")
                    if latest_row else None
                ),
                "pressure_mpa": (
                    latest_row.get("pressure_mpa")
                    if latest_row else None
                ),
                "rpm": (
                    latest_row.get("rpm")
                    if latest_row else None
                ),
                "device_status": (
                    latest_row.get("device_status")
                    if latest_row else None
                ),
                "alarm_code": (
                    latest_row.get("alarm_code")
                    if latest_row else None
                ),
                "is_real_telemetry": bool(
                    latest.get("is_real_telemetry")
                ),
            }
        ),
        "all_simulator": all(
            data.get("is_real_telemetry") is False
            for _, _, data in sessions
        ),
        "latest_session_path": str(latest_path),
    }


def _safety_view(
    root: Path,
    campaign_id: str,
    round_id: str,
) -> dict[str, Any] | None:
    safety_root = root / "v030" / "safety"
    interlocks: list[tuple[float, Path, dict[str, Any]]] = []
    if safety_root.is_dir():
        for path in safety_root.glob("*/safety.json"):
            if campaign_id not in path.parent.name or round_id not in path.parent.name:
                continue
            data = _read_json(path)
            if data:
                interlocks.append((path.stat().st_mtime, path, data))
    if not interlocks:
        return None

    stopped = [
        (mtime, path, data)
        for mtime, path, data in interlocks
        if data.get("state") == "SAFETY_STOP"
    ]
    chosen = sorted(
        stopped if stopped else interlocks,
        key=lambda item: item[0],
    )[-1]
    _, path, data = chosen
    current_trip = data.get("current_trip")
    return {
        "state": data.get("state"),
        "interlock_count": len(interlocks),
        "safety_stop_count": len(stopped),
        "current_trip": (
            {
                "trip_id": current_trip.get("trip_id"),
                "code": current_trip.get("code"),
                "recoverable_same_job": current_trip.get(
                    "recoverable_same_job"
                ),
                "acknowledged": current_trip.get("acknowledged"),
                "acknowledged_by": current_trip.get("acknowledged_by"),
                "last_recheck_safe": current_trip.get(
                    "last_recheck_safe"
                ),
            }
            if isinstance(current_trip, dict) else None
        ),
        "automatic_resume_allowed": False,
        "state_path": str(path),
    }


def _result_capture_count(
    root: Path,
    campaign_id: str,
    round_id: str,
) -> int:
    capture_dir = (
        root / "v030" / "result_capture" / campaign_id / round_id
    )
    return (
        len(list(capture_dir.glob("*.json")))
        if capture_dir.is_dir() else 0
    )


def _round_runtime(
    root: Path,
    campaign_id: str,
    round_record: dict[str, Any],
) -> dict[str, Any]:
    round_id = str(round_record.get("round_id") or "")
    autonomous_report = _read_json(
        root
        / "v030"
        / "autonomous_round"
        / campaign_id
        / round_id
        / "autonomous_round_report.json"
    )
    crash_dir = (
        root / "v030" / "crash_resume" / campaign_id / round_id
    )
    crash_checkpoint = _read_json(
        crash_dir / "crash_checkpoint.json"
    )
    recovery_report = _read_json(
        crash_dir / "recovery_report.json"
    )
    return {
        "round_id": round_id,
        "round_no": round_record.get("round_no"),
        "status": round_record.get("status"),
        "dataset_version": (
            (round_record.get("plan") or {}).get("dataset_version")
        ),
        "planned_experiments": (
            (round_record.get("plan") or {}).get(
                "planned_experiment_count"
            )
        ),
        "progress": deepcopy(round_record.get("progress") or {}),
        "scheduler": _scheduler_view(root, campaign_id, round_id),
        "telemetry": _telemetry_view(root, campaign_id, round_id),
        "safety": _safety_view(root, campaign_id, round_id),
        "capture_receipts": _result_capture_count(
            root, campaign_id, round_id
        ),
        "autonomous_report": autonomous_report,
        "crash_checkpoint": crash_checkpoint,
        "recovery_report": recovery_report,
    }


def _operator_input_dir(root: Path, project_id: int) -> Path:
    return (
        root / "v030" / "ui_inputs" / f"project_{project_id}"
    )


def resolve_operator_inputs(
    runtime_root: str | Path,
    *,
    project_id: int,
) -> dict[str, Any]:
    root = Path(runtime_root)
    folder = _operator_input_dir(root, project_id)
    files = {
        "protocol_template": folder / "protocol_template.json",
        "device_profile": folder / "device_profile.json",
        "safety_policy": folder / "safety_policy.json",
        "config": folder / "autonomy.json",
        "candidate_pool": folder / "candidate_pool.csv",
        "gate": folder / "gate.json",
    }
    base_required = (
        "protocol_template",
        "device_profile",
        "safety_policy",
    )
    missing_base = [
        key for key in base_required
        if not files[key].is_file()
    ]
    config = _read_json(files["config"]) if files["config"].is_file() else None
    resume_required = (
        "candidate_pool",
        "gate",
        "config",
    )
    missing_resume = [
        key for key in resume_required
        if (
            not files[key].is_file()
            if key != "config"
            else config is None
        )
    ]
    return {
        "folder": str(folder),
        "base_ready": not missing_base,
        "resume_ready": not missing_base and not missing_resume,
        "missing_base": missing_base,
        "missing_resume": missing_resume,
        "files": {key: str(path) for key, path in files.items()},
        "config": deepcopy(config),
    }


def build_autonomy_overview(
    runtime_root: str | Path,
    *,
    campaign_id: str | None = None,
    project_id: int | None = None,
) -> dict[str, Any]:
    root = Path(runtime_root)
    if not campaign_id:
        if project_id is None:
            raise V030UIError(
                "campaign_id / project_id 至少提供一个"
            )
        campaign_id = latest_v030_campaign_id_for_project(
            root, int(project_id)
        )

    try:
        feedback = build_campaign_overview(
            root,
            campaign_id=campaign_id,
            project_id=project_id,
        )
    except V020UIError as exc:
        raise V030UIError(str(exc)) from exc

    campaign = feedback["campaign"]
    actual_project_id = int(campaign["project_id"])
    store = CampaignStore(root)
    try:
        raw_campaign = store.load(campaign_id)
    except Exception as exc:
        raise V030UIError(str(exc)) from exc

    round_runtime = [
        _round_runtime(root, campaign_id, record)
        for record in raw_campaign.get("rounds") or []
    ]
    latest_round = round_runtime[-1] if round_runtime else None
    loop_report = _read_json(
        root
        / "v030"
        / "autonomous_loop"
        / campaign_id
        / "autonomous_loop_report.json"
    )

    total_captures = sum(
        int(item.get("capture_receipts") or 0)
        for item in round_runtime
    )
    total_safety_stops = sum(
        int((item.get("safety") or {}).get("safety_stop_count") or 0)
        for item in round_runtime
    )
    latest_scheduler = (
        latest_round.get("scheduler") if latest_round else None
    )
    latest_telemetry = (
        latest_round.get("telemetry") if latest_round else None
    )
    latest_safety = (
        latest_round.get("safety") if latest_round else None
    )
    latest_crash_checkpoint = (
        latest_round.get("crash_checkpoint")
        if latest_round else None
    )
    latest_recovery = (
        latest_round.get("recovery_report")
        if latest_round else None
    )

    operator_inputs = resolve_operator_inputs(
        root, project_id=actual_project_id
    )
    round_status = (
        latest_round.get("status") if latest_round else "NO_ROUND"
    )
    has_active_scheduler = bool(
        latest_scheduler
        and any(
            int((latest_scheduler.get("counts") or {}).get(x) or 0)
            > 0
            for x in ("QUEUED", "DISPATCHED", "RUNNING")
        )
    )
    operator_available = bool(
        latest_round
        and round_status in {"RUNNING", "PARTIALLY_COMPLETED"}
        and latest_crash_checkpoint
        and has_active_scheduler
        and operator_inputs["base_ready"]
    )

    if loop_report and loop_report.get("round_count"):
        answer = (
            f"V0.3 Campaign {campaign_id} 已记录 "
            f"{loop_report.get('round_count')} 轮 autonomous loop；"
            f"自动结果回流 {total_captures} 条。"
        )
    elif latest_round:
        phase = (
            ((latest_telemetry or {}).get("latest") or {}).get("phase")
            or "-"
        )
        answer = (
            f"V0.3 Campaign {campaign_id} 当前 "
            f"R{latest_round.get('round_no')}={round_status}，"
            f"设备阶段 {phase}，Safety="
            f"{(latest_safety or {}).get('state') or 'NO_STATE'}。"
        )
    else:
        answer = f"V0.3 Campaign {campaign_id} 尚无 Round。"

    return {
        "kind": "v030_autonomy",
        "status": (
            "SAFETY_STOP"
            if latest_safety
            and latest_safety.get("state") == "SAFETY_STOP"
            else round_status
        ),
        "answer": answer,
        "campaign": deepcopy(campaign),
        "rounds": round_runtime,
        "latest_round": latest_round,
        "autonomous_loop": deepcopy(loop_report),
        "datasets": deepcopy(feedback.get("datasets") or []),
        "latest_dataset": deepcopy(feedback.get("latest_dataset")),
        "evaluation": deepcopy(feedback.get("evaluation")),
        "model_registry": deepcopy(feedback.get("model_registry")),
        "model_promotion": deepcopy(feedback.get("model_promotion")),
        "summary": {
            "round_count": len(round_runtime),
            "automatic_capture_count": total_captures,
            "manual_result_submission_required": False,
            "safety_stop_count": total_safety_stops,
            "active_model_version": (
                feedback.get("summary") or {}
            ).get("active_model_version"),
            "promotion_decision": (
                feedback.get("summary") or {}
            ).get("promotion_decision"),
            "real_device_connected": False,
        },
        "operator": {
            "available": operator_available,
            "actions": (
                ["RESUME", "CANCEL_JOB", "ABORT_ROUND"]
                if operator_available else []
            ),
            "inputs": operator_inputs,
            "crash_checkpoint_present": bool(
                latest_crash_checkpoint
            ),
            "recovery_report_present": bool(latest_recovery),
        },
        "boundary": {
            "simulator_only": True,
            "real_device_connected": False,
            "real_device_validation_completed": False,
            "automatic_model_activation": False,
            "operator_override_cannot_bypass_safety": True,
            "automatic_result_capture": True,
            "simulator_results_are_not_real_material_measurements": True,
        },
    }


def _load_operator_bundle(
    root: Path,
    project_id: int,
) -> dict[str, Any]:
    resolved = resolve_operator_inputs(root, project_id=project_id)
    if not resolved["base_ready"]:
        raise V030UIError(
            "缺少 operator 基础配置: "
            + ", ".join(resolved["missing_base"])
        )
    files = {
        key: Path(value)
        for key, value in resolved["files"].items()
    }
    return {
        "resolved": resolved,
        "protocol_template": _read_json(files["protocol_template"]),
        "device_profile": _read_json(files["device_profile"]),
        "safety_policy": _read_json(files["safety_policy"]),
        "gate": (
            _read_json(files["gate"])
            if files["gate"].is_file() else None
        ),
        "candidate_pool_csv": files["candidate_pool"],
        "config": resolved["config"],
    }


def operator_override_for_ui(
    runtime_root: str | Path,
    *,
    campaign_id: str,
    round_id: str,
    action: str,
    operator_id: str,
    reason: str,
) -> dict[str, Any]:
    root = Path(runtime_root)
    action = str(action or "").strip().upper()
    if action not in {"RESUME", "CANCEL_JOB", "ABORT_ROUND"}:
        raise V030UIError(
            "action 必须是 RESUME / CANCEL_JOB / ABORT_ROUND"
        )
    operator_id = str(operator_id or "").strip()
    reason = str(reason or "").strip()
    if not operator_id:
        raise V030UIError("operator_id 不能为空")
    if not reason:
        raise V030UIError("reason 不能为空")

    store = CampaignStore(root)
    try:
        campaign = store.load(campaign_id)
        find_round(campaign, round_id)
    except Exception as exc:
        raise V030UIError(str(exc)) from exc

    project_id = int(campaign["project_id"])
    bundle = _load_operator_bundle(root, project_id)
    if action == "RESUME" and not bundle["resolved"]["resume_ready"]:
        raise V030UIError(
            "RESUME 缺少完整闭环配置: "
            + ", ".join(bundle["resolved"]["missing_resume"])
        )

    coordinator = CrashResumeCoordinator(root)
    kwargs: dict[str, Any] = {
        "campaign_id": campaign_id,
        "round_id": round_id,
        "protocol_template": bundle["protocol_template"],
        "device_profile": bundle["device_profile"],
        "safety_policy": bundle["safety_policy"],
        "operator_action": action,
        "operator_id": operator_id,
        "reason": reason,
    }

    if action == "RESUME":
        config = bundle["config"] or {}
        required = (
            "target_metric",
            "target_unit",
            "child_dataset_version",
            "challenger_model_version",
        )
        missing = [
            key for key in required
            if not str(config.get(key) or "").strip()
        ]
        if missing:
            raise V030UIError(
                "autonomy.json 缺少字段: " + ", ".join(missing)
            )
        kwargs.update({
            "candidate_pool_csv": bundle["candidate_pool_csv"],
            "target_metric": config["target_metric"],
            "target_unit": config["target_unit"],
            "gate": bundle["gate"],
            "child_dataset_version": config["child_dataset_version"],
            "challenger_model_version": config[
                "challenger_model_version"
            ],
            "next_batch_size": int(
                config.get("next_batch_size", 5)
            ),
        })

    try:
        action_result = coordinator.resume_after_crash(**kwargs)
    except Exception as exc:
        raise V030UIError(
            f"{action} 执行失败: {type(exc).__name__}: {exc}"
        ) from exc

    overview = build_autonomy_overview(
        root, campaign_id=campaign_id
    )
    overview["last_operator_action"] = {
        "action": action,
        "operator_id": operator_id,
        "reason": reason,
        "result": action_result,
    }
    return overview
