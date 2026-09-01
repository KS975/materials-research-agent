from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .device import (
    DeviceAdapter,
    DeviceAdapterError,
    DeviceBusyError,
    DeviceExecutionError,
    DeviceOfflineError,
    DeviceStateError,
    DeviceUnsupportedProtocolError,
    SimulatorDeviceAdapter,
)
from .protocol import (
    ExperimentProtocolValidationError,
    sha256_json,
    validate_protocol_document,
)


SCHEDULER_STAGE = "V0.3-T29_job_scheduler"
SCHEDULER_SCHEMA_VERSION = 1

JOB_STATES = {
    "QUEUED",
    "DISPATCHED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
}
ACTIVE_JOB_STATES = {"QUEUED", "DISPATCHED", "RUNNING"}
TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}


class JobSchedulerError(RuntimeError):
    code = "SCHEDULER_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class JobSchedulerValidationError(JobSchedulerError):
    code = "SCHEDULER_VALIDATION_ERROR"


class JobSchedulerConflictError(JobSchedulerError):
    code = "SCHEDULER_CONFLICT"


class JobSchedulerStateError(JobSchedulerError):
    code = "SCHEDULER_STATE_ERROR"


def deterministic_scheduler_job_id(scheduler_id: str, protocol_id: str) -> str:
    digest = sha256_json({
        "stage": SCHEDULER_STAGE,
        "scheduler_id": str(scheduler_id),
        "protocol_id": str(protocol_id),
    })
    return f"schedjob_{digest[:20]}"


def _finite_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise JobSchedulerValidationError(f"{name} 必须是整数")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise JobSchedulerValidationError(f"{name} 必须是整数") from exc
    if not math.isfinite(as_float) or abs(as_float - round(as_float)) > 1e-9:
        raise JobSchedulerValidationError(f"{name} 必须是整数")
    out = int(round(as_float))
    if out < minimum or out > maximum:
        raise JobSchedulerValidationError(
            f"{name} 超出范围 [{minimum}, {maximum}]: {out}"
        )
    return out


ATOMIC_WRITE_MAX_RETRIES = 20
ATOMIC_WRITE_RETRY_BASE_SECONDS = 0.02


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist scheduler state with Windows-friendly retries.

    Windows can transiently reject ``os.replace`` with WinError 5 when
    Defender/indexers briefly hold the destination file.  Scheduler state is
    written frequently, so retry only the atomic rename step; never fall back
    to a non-atomic overwrite.  A unique same-directory temp file also avoids
    collisions between closely-spaced saves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        last_error: PermissionError | None = None
        for attempt in range(ATOMIC_WRITE_MAX_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt + 1 >= ATOMIC_WRITE_MAX_RETRIES:
                    break
                delay = min(
                    ATOMIC_WRITE_RETRY_BASE_SECONDS * (attempt + 1),
                    0.20,
                )
                time.sleep(delay)

        raise PermissionError(
            getattr(last_error, "errno", 13),
            (
                f"Scheduler atomic replace failed after "
                f"{ATOMIC_WRITE_MAX_RETRIES} retries: {tmp} -> {path}. "
                "On Windows, check whether antivirus/indexing/backup software "
                "is holding scheduler.json open."
            ),
            str(path),
        ) from last_error
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


class JobScheduler:
    """Deterministic single-process scheduler introduced in V0.3-T29.

    T29 deliberately schedules only DeviceAdapter instances supplied by the
    caller. It does not discover or control real instruments. The acceptance
    fixture uses SimulatorDeviceAdapter exclusively.

    Crash reconciliation of an *active* physical job is intentionally deferred
    to T35. T29 persists queue/terminal audit state atomically, but a new process
    must not pretend it can infer the state of an in-flight real device.
    """

    def __init__(
        self,
        *,
        scheduler_id: str,
        devices: dict[str, DeviceAdapter],
        runtime_root: str | Path = ".runtime",
    ) -> None:
        scheduler_id = str(scheduler_id or "").strip()
        if not scheduler_id:
            raise JobSchedulerValidationError("scheduler_id 不能为空")
        if not isinstance(devices, dict) or not devices:
            raise JobSchedulerValidationError("devices 必须是非空 dict")
        if len(devices) != len(set(devices)):
            raise JobSchedulerValidationError("device id 重复")
        for device_id, adapter in devices.items():
            if not isinstance(adapter, DeviceAdapter):
                raise JobSchedulerValidationError(
                    f"{device_id} 不是 DeviceAdapter"
                )
            if str(device_id) != str(getattr(adapter, "device_id", "")):
                raise JobSchedulerValidationError(
                    f"devices key 与 adapter.device_id 不一致: {device_id}"
                )

        self.scheduler_id = scheduler_id
        self.devices = dict(devices)
        self.runtime_root = Path(runtime_root)
        self.root = (
            self.runtime_root / "v030" / "scheduler" / self.scheduler_id
        )
        self.state_path = self.root / "scheduler.json"
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._validate_loaded_state()
        else:
            self._state = {
                "stage": SCHEDULER_STAGE,
                "schema_version": SCHEDULER_SCHEMA_VERSION,
                "scheduler_id": self.scheduler_id,
                "next_enqueue_sequence": 1,
                "jobs": {},
                "events": [],
            }
            self._event("SCHEDULER_CREATED")
            self._save()

    def _validate_loaded_state(self) -> None:
        if self._state.get("stage") != SCHEDULER_STAGE:
            raise JobSchedulerValidationError("scheduler state.stage 错误")
        if self._state.get("schema_version") != SCHEDULER_SCHEMA_VERSION:
            raise JobSchedulerValidationError("scheduler schema_version 错误")
        if self._state.get("scheduler_id") != self.scheduler_id:
            raise JobSchedulerValidationError("scheduler_id 与持久化状态不一致")
        if not isinstance(self._state.get("jobs"), dict):
            raise JobSchedulerValidationError("scheduler jobs 必须是 object")
        for job in self._state["jobs"].values():
            if job.get("status") not in JOB_STATES:
                raise JobSchedulerValidationError(
                    f"未知持久化 job 状态: {job.get('status')}"
                )

    def _save(self) -> None:
        _atomic_write_json(self.state_path, self._state)

    def _event(
        self,
        event_type: str,
        *,
        scheduler_job_id: str | None = None,
        device_id: str | None = None,
        **payload: Any,
    ) -> None:
        self._state.setdefault("events", []).append({
            "sequence": len(self._state.get("events") or []) + 1,
            "event_type": event_type,
            "scheduler_job_id": scheduler_job_id,
            "device_id": device_id,
            "payload": deepcopy(payload),
        })

    def _job(self, scheduler_job_id: str) -> dict[str, Any]:
        try:
            return self._state["jobs"][scheduler_job_id]
        except KeyError as exc:
            raise JobSchedulerValidationError(
                f"未知 scheduler_job_id: {scheduler_job_id}"
            ) from exc

    def snapshot(self) -> dict[str, Any]:
        jobs = list(self._state["jobs"].values())
        counts = {state: 0 for state in sorted(JOB_STATES)}
        for job in jobs:
            counts[job["status"]] += 1
        return {
            "stage": SCHEDULER_STAGE,
            "schema_version": SCHEDULER_SCHEMA_VERSION,
            "scheduler_id": self.scheduler_id,
            "job_count": len(jobs),
            "counts": counts,
            "jobs": deepcopy(sorted(jobs, key=lambda x: x["enqueue_sequence"])),
            "events": deepcopy(self._state.get("events") or []),
            "state_path": str(self.state_path),
            "real_device_connected": False,
        }

    def submit(
        self,
        protocol: dict[str, Any],
        *,
        priority: int = 0,
        timeout_ticks: int = 100,
        max_retries: int = 0,
    ) -> dict[str, Any]:
        try:
            validate_protocol_document(protocol)
        except ExperimentProtocolValidationError as exc:
            raise JobSchedulerValidationError(
                "只能提交合法 T27 protocol",
                details={"reason": str(exc)},
            ) from exc
        if protocol.get("status") != "READY":
            raise JobSchedulerValidationError(
                "只有 T27 READY protocol 才能进入 scheduler",
                details={
                    "protocol_id": protocol.get("protocol_id"),
                    "status": protocol.get("status"),
                },
            )

        priority = _finite_int(priority, "priority", minimum=-100, maximum=100)
        timeout_ticks = _finite_int(
            timeout_ticks, "timeout_ticks", minimum=1, maximum=1_000_000
        )
        max_retries = _finite_int(
            max_retries, "max_retries", minimum=0, maximum=10
        )
        scheduler_job_id = deterministic_scheduler_job_id(
            self.scheduler_id, protocol["protocol_id"]
        )
        existing = self._state["jobs"].get(scheduler_job_id)
        config = {
            "priority": priority,
            "timeout_ticks": timeout_ticks,
            "max_retries": max_retries,
        }
        if existing is not None:
            existing_config = {
                "priority": existing["priority"],
                "timeout_ticks": existing["timeout_ticks"],
                "max_retries": existing["max_retries"],
            }
            if (
                existing.get("protocol_sha256") == protocol.get("content_sha256")
                and existing_config == config
            ):
                return {
                    "idempotent_replay": True,
                    "job": deepcopy(existing),
                }
            raise JobSchedulerConflictError(
                "同一 protocol 已以不同 scheduler 配置提交",
                details={
                    "scheduler_job_id": scheduler_job_id,
                    "existing_config": existing_config,
                    "new_config": config,
                },
            )

        seq = int(self._state["next_enqueue_sequence"])
        self._state["next_enqueue_sequence"] = seq + 1
        job = {
            "scheduler_job_id": scheduler_job_id,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol["content_sha256"],
            "candidate_id": protocol["candidate_id"],
            "protocol": deepcopy(protocol),
            "priority": priority,
            "enqueue_sequence": seq,
            "status": "QUEUED",
            "attempts_started": 0,
            "max_retries": max_retries,
            "timeout_ticks": timeout_ticks,
            "elapsed_ticks": 0,
            "device_id": None,
            "adapter_job_id": None,
            "result": None,
            "last_error": None,
        }
        self._state["jobs"][scheduler_job_id] = job
        self._event(
            "JOB_QUEUED",
            scheduler_job_id=scheduler_job_id,
            priority=priority,
            enqueue_sequence=seq,
        )
        self._save()
        return {"idempotent_replay": False, "job": deepcopy(job)}

    def queued_order(self) -> list[str]:
        jobs = [
            job for job in self._state["jobs"].values()
            if job["status"] == "QUEUED"
        ]
        jobs.sort(key=lambda x: (-int(x["priority"]), int(x["enqueue_sequence"])))
        return [job["scheduler_job_id"] for job in jobs]

    def _device_is_bound(self, device_id: str) -> bool:
        return any(
            job.get("device_id") == device_id
            and job.get("status") in {"DISPATCHED", "RUNNING"}
            for job in self._state["jobs"].values()
        )

    def _ensure_connect(self, adapter: DeviceAdapter) -> None:
        status = adapter.status()
        if status.get("state") == "DISCONNECTED" or not status.get("connected"):
            adapter.connect()

    def _cleanup_device_after_failure(self, adapter: DeviceAdapter) -> None:
        try:
            status = adapter.status()
            if status.get("state") in {"PREPARED", "SUBMITTED", "RUNNING", "PAUSED"}:
                try:
                    adapter.cancel()
                except DeviceAdapterError:
                    pass
            status = adapter.status()
            if status.get("state") != "DISCONNECTED":
                try:
                    adapter.disconnect()
                except DeviceAdapterError:
                    pass
        except DeviceAdapterError:
            pass

    def dispatch_once(self, *, max_jobs: int | None = None) -> list[dict[str, Any]]:
        if max_jobs is not None:
            max_jobs = _finite_int(max_jobs, "max_jobs", minimum=1, maximum=10_000)
        dispatched: list[dict[str, Any]] = []
        device_ids = sorted(self.devices)
        for scheduler_job_id in self.queued_order():
            if max_jobs is not None and len(dispatched) >= max_jobs:
                break
            job = self._job(scheduler_job_id)
            selected: tuple[str, DeviceAdapter, dict[str, Any]] | None = None
            for device_id in device_ids:
                if self._device_is_bound(device_id):
                    continue
                adapter = self.devices[device_id]
                try:
                    self._ensure_connect(adapter)
                    state = adapter.status().get("state")
                    if state in {"PREPARED", "SUBMITTED", "RUNNING", "PAUSED", "ERROR"}:
                        continue
                    prepared = adapter.prepare(job["protocol"])
                    selected = (device_id, adapter, prepared)
                    break
                except (
                    DeviceOfflineError,
                    DeviceBusyError,
                    DeviceUnsupportedProtocolError,
                    DeviceStateError,
                    DeviceExecutionError,
                ):
                    continue

            if selected is None:
                continue

            device_id, adapter, _prepared = selected
            try:
                submitted = adapter.submit_protocol(job["protocol"])
            except DeviceAdapterError as exc:
                self._cleanup_device_after_failure(adapter)
                job["last_error"] = exc.as_dict()
                self._event(
                    "DISPATCH_FAILED",
                    scheduler_job_id=scheduler_job_id,
                    device_id=device_id,
                    error=exc.as_dict(),
                )
                continue

            job["status"] = "DISPATCHED"
            job["device_id"] = device_id
            job["adapter_job_id"] = (submitted.get("job") or {}).get("job_id")
            job["elapsed_ticks"] = 0
            job["last_error"] = None
            self._event(
                "JOB_DISPATCHED",
                scheduler_job_id=scheduler_job_id,
                device_id=device_id,
                adapter_job_id=job["adapter_job_id"],
            )
            dispatched.append(deepcopy(job))
        self._save()
        return dispatched

    def _handle_execution_failure(
        self,
        job: dict[str, Any],
        adapter: DeviceAdapter,
        exc: DeviceAdapterError,
    ) -> None:
        scheduler_job_id = job["scheduler_job_id"]
        device_id = job.get("device_id")
        job["last_error"] = exc.as_dict()
        self._event(
            "JOB_EXECUTION_ERROR",
            scheduler_job_id=scheduler_job_id,
            device_id=device_id,
            attempts_started=job["attempts_started"],
            error=exc.as_dict(),
        )
        self._cleanup_device_after_failure(adapter)
        retries_used = max(int(job["attempts_started"]) - 1, 0)
        if retries_used < int(job["max_retries"]):
            job["status"] = "QUEUED"
            job["device_id"] = None
            job["adapter_job_id"] = None
            job["elapsed_ticks"] = 0
            self._event(
                "JOB_REQUEUED_FOR_RETRY",
                scheduler_job_id=scheduler_job_id,
                retries_used=retries_used + 1,
                max_retries=job["max_retries"],
            )
        else:
            job["status"] = "FAILED"
            self._event(
                "JOB_FAILED",
                scheduler_job_id=scheduler_job_id,
                device_id=device_id,
                attempts_started=job["attempts_started"],
            )

    def start_dispatched(self) -> list[dict[str, Any]]:
        started: list[dict[str, Any]] = []
        jobs = sorted(
            [j for j in self._state["jobs"].values() if j["status"] == "DISPATCHED"],
            key=lambda x: x["enqueue_sequence"],
        )
        for job in jobs:
            device_id = job.get("device_id")
            adapter = self.devices.get(str(device_id))
            if adapter is None:
                raise JobSchedulerStateError(
                    "已 dispatch job 的 device 不在当前 registry",
                    details={"device_id": device_id},
                )
            job["attempts_started"] = int(job["attempts_started"]) + 1
            job["elapsed_ticks"] = 0
            try:
                adapter.start()
            except DeviceAdapterError as exc:
                self._handle_execution_failure(job, adapter, exc)
                continue
            job["status"] = "RUNNING"
            self._event(
                "JOB_STARTED",
                scheduler_job_id=job["scheduler_job_id"],
                device_id=str(device_id),
                attempt=job["attempts_started"],
            )
            started.append(deepcopy(job))
        self._save()
        return started

    def advance_running(self, *, ticks: int = 1) -> list[dict[str, Any]]:
        ticks = _finite_int(ticks, "ticks", minimum=1, maximum=100_000)
        changed: list[dict[str, Any]] = []
        for _ in range(ticks):
            running_ids = [
                j["scheduler_job_id"]
                for j in sorted(
                    self._state["jobs"].values(),
                    key=lambda x: x["enqueue_sequence"],
                )
                if j["status"] == "RUNNING"
            ]
            for scheduler_job_id in running_ids:
                job = self._job(scheduler_job_id)
                device_id = str(job.get("device_id") or "")
                adapter = self.devices.get(device_id)
                if adapter is None:
                    raise JobSchedulerStateError(
                        "RUNNING job 的 device 不在当前 registry",
                        details={"device_id": device_id},
                    )
                if not isinstance(adapter, SimulatorDeviceAdapter):
                    raise JobSchedulerStateError(
                        "T29 advance_running 只允许 SimulatorDeviceAdapter；真实设备 telemetry 留到 T30"
                    )
                try:
                    adapter.tick(steps=1)
                except DeviceAdapterError as exc:
                    self._handle_execution_failure(job, adapter, exc)
                    changed.append(deepcopy(job))
                    continue

                job["elapsed_ticks"] = int(job["elapsed_ticks"]) + 1
                device_status = adapter.status()
                if device_status.get("state") == "COMPLETED":
                    try:
                        result = adapter.read_result()
                    except DeviceAdapterError as exc:
                        self._handle_execution_failure(job, adapter, exc)
                        changed.append(deepcopy(job))
                        continue
                    job["status"] = "COMPLETED"
                    job["result"] = deepcopy(result)
                    self._event(
                        "JOB_COMPLETED",
                        scheduler_job_id=scheduler_job_id,
                        device_id=device_id,
                        result_id=result.get("result_id"),
                        measurement_origin=result.get("measurement_origin"),
                    )
                    changed.append(deepcopy(job))
                    continue

                if int(job["elapsed_ticks"]) >= int(job["timeout_ticks"]):
                    try:
                        adapter.cancel()
                    except DeviceAdapterError:
                        self._cleanup_device_after_failure(adapter)
                    job["status"] = "TIMEOUT"
                    job["last_error"] = {
                        "code": "TIMEOUT",
                        "message": (
                            f"job 超过 timeout_ticks={job['timeout_ticks']}"
                        ),
                        "details": {"elapsed_ticks": job["elapsed_ticks"]},
                    }
                    self._event(
                        "JOB_TIMEOUT",
                        scheduler_job_id=scheduler_job_id,
                        device_id=device_id,
                        elapsed_ticks=job["elapsed_ticks"],
                    )
                    changed.append(deepcopy(job))
            self._save()
        return changed

    def cancel_job(self, scheduler_job_id: str) -> dict[str, Any]:
        job = self._job(scheduler_job_id)
        if job["status"] == "CANCELLED":
            return {"idempotent_replay": True, "job": deepcopy(job)}
        if job["status"] in TERMINAL_JOB_STATES:
            raise JobSchedulerStateError(
                f"终态 job 不能 cancel: {job['status']}"
            )
        device_id = job.get("device_id")
        if job["status"] in {"DISPATCHED", "RUNNING"} and device_id:
            adapter = self.devices[str(device_id)]
            try:
                adapter.cancel()
            except DeviceAdapterError as exc:
                self._cleanup_device_after_failure(adapter)
                job["last_error"] = exc.as_dict()
        job["status"] = "CANCELLED"
        self._event(
            "JOB_CANCELLED",
            scheduler_job_id=scheduler_job_id,
            device_id=str(device_id) if device_id else None,
        )
        self._save()
        return {"idempotent_replay": False, "job": deepcopy(job)}

    def run_until_terminal(self, *, max_scheduler_ticks: int = 10_000) -> dict[str, Any]:
        max_scheduler_ticks = _finite_int(
            max_scheduler_ticks,
            "max_scheduler_ticks",
            minimum=1,
            maximum=1_000_000,
        )
        for _ in range(max_scheduler_ticks):
            active = [
                j for j in self._state["jobs"].values()
                if j["status"] in ACTIVE_JOB_STATES
            ]
            if not active:
                return self.snapshot()

            before = sha256_json({
                k: {kk: vv for kk, vv in v.items() if kk != "protocol"}
                for k, v in self._state["jobs"].items()
            })
            self.dispatch_once()
            self.start_dispatched()
            self.advance_running(ticks=1)
            after = sha256_json({
                k: {kk: vv for kk, vv in v.items() if kk != "protocol"}
                for k, v in self._state["jobs"].items()
            })
            if before == after:
                # No compatible/free device can make progress. Return rather
                # than spinning forever; caller can inspect remaining QUEUED.
                break
        return self.snapshot()
