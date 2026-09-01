from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from .device import (
    DeviceAdapter,
    DeviceAdapterError,
    SimulatorDeviceAdapter,
)
from .protocol import (
    ExperimentProtocolValidationError,
    sha256_json,
    validate_protocol_document,
)


TELEMETRY_STAGE = "V0.3-T30_telemetry"
TELEMETRY_SCHEMA_VERSION = 1

CORE_EXPERIMENT_PHASES = [
    "PREPARING",
    "MATERIAL_LOADING",
    "HEATING",
    "PROCESSING",
    "COOLING",
    "MEASURING",
    "COMPLETED",
]
EXPERIMENT_PHASES = set(CORE_EXPERIMENT_PHASES) | {
    "PAUSED",
    "CANCELLED",
    "ERROR",
}
TERMINAL_EXPERIMENT_PHASES = {"COMPLETED", "CANCELLED", "ERROR"}

SIMULATOR_TIME_SOURCE = "SIMULATOR_VIRTUAL_CLOCK"
SIMULATOR_VIRTUAL_START = "2026-01-01T00:00:00+00:00"


class TelemetryError(RuntimeError):
    code = "TELEMETRY_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class TelemetryValidationError(TelemetryError):
    code = "TELEMETRY_VALIDATION_ERROR"


class TelemetryStateError(TelemetryError):
    code = "TELEMETRY_STATE_ERROR"


class TelemetryIntegrityError(TelemetryError):
    code = "TELEMETRY_INTEGRITY_ERROR"


def _text(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise TelemetryValidationError(f"{name} 不能为空")
    return out


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TelemetryValidationError(f"{name} 必须是有限数值")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(out):
        raise TelemetryValidationError(f"{name} 必须是有限数值")
    return out


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(tmp, path)


def _virtual_timestamp(sequence: int) -> str:
    start = datetime.fromisoformat(SIMULATOR_VIRTUAL_START)
    return (start + timedelta(seconds=max(sequence - 1, 0))).isoformat()


def _parameter_value(
    protocol: dict[str, Any],
    names: tuple[str, ...],
) -> float | None:
    lowered = {name.casefold() for name in names}
    for item in protocol.get("process_parameters") or []:
        name = str(item.get("name") or "").strip()
        source = str(item.get("source_feature") or "").strip()
        if name.casefold() in lowered or source.casefold() in lowered:
            value = item.get("value")
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def phase_for_device_status(
    device_status: dict[str, Any],
    *,
    last_active_phase: str | None = None,
) -> str:
    state = str(device_status.get("state") or "").strip().upper()
    job = device_status.get("job") or {}
    progress = float(job.get("progress") or 0.0)

    if state in {"PREPARED", "SUBMITTED"}:
        return "PREPARING"
    if state == "PAUSED":
        return "PAUSED"
    if state == "COMPLETED":
        return "COMPLETED"
    if state == "CANCELLED":
        return "CANCELLED"
    if state == "ERROR":
        return "ERROR"
    if state != "RUNNING":
        raise TelemetryStateError(
            f"当前 device state 不属于实验遥测状态: {state}",
            details={"device_state": state},
        )

    if progress <= 0:
        return "PREPARING"
    if progress <= 10:
        return "MATERIAL_LOADING"
    if progress <= 30:
        return "HEATING"
    if progress <= 75:
        return "PROCESSING"
    if progress <= 88:
        return "COOLING"
    if progress < 100:
        return "MEASURING"
    return "COMPLETED"


class ExperimentStateMachine:
    def __init__(self) -> None:
        self.phase: str | None = None
        self.last_active_phase: str | None = None
        self.history: list[str] = []

    def transition(self, next_phase: str) -> dict[str, Any]:
        next_phase = _text(next_phase, "next_phase").upper()
        if next_phase not in EXPERIMENT_PHASES:
            raise TelemetryStateError(f"未知 experiment phase: {next_phase}")

        previous = self.phase
        if previous in TERMINAL_EXPERIMENT_PHASES and next_phase != previous:
            raise TelemetryStateError(
                f"终态 {previous} 不能转换为 {next_phase}"
            )

        if previous == next_phase:
            return {
                "changed": False,
                "previous_phase": previous,
                "phase": next_phase,
            }

        if previous is None:
            if next_phase != "PREPARING":
                raise TelemetryStateError(
                    f"实验必须从 PREPARING 开始，不能直接进入 {next_phase}"
                )
        elif next_phase == "PAUSED":
            if previous not in {
                "PREPARING", "MATERIAL_LOADING", "HEATING",
                "PROCESSING", "COOLING", "MEASURING",
            }:
                raise TelemetryStateError(
                    f"{previous} 不能进入 PAUSED"
                )
            self.last_active_phase = previous
        elif previous == "PAUSED":
            if next_phase != self.last_active_phase:
                raise TelemetryStateError(
                    "PAUSED 后只能恢复到暂停前 phase",
                    details={
                        "last_active_phase": self.last_active_phase,
                        "requested_phase": next_phase,
                    },
                )
        elif next_phase in {"CANCELLED", "ERROR"}:
            pass
        else:
            if previous not in CORE_EXPERIMENT_PHASES:
                raise TelemetryStateError(
                    f"{previous} 不能转换为 {next_phase}"
                )
            pi = CORE_EXPERIMENT_PHASES.index(previous)
            ni = CORE_EXPERIMENT_PHASES.index(next_phase)
            if ni != pi + 1:
                raise TelemetryStateError(
                    "核心实验 phase 必须按顺序前进",
                    details={
                        "previous_phase": previous,
                        "requested_phase": next_phase,
                    },
                )

        self.phase = next_phase
        if next_phase not in {"PAUSED", "CANCELLED", "ERROR", "COMPLETED"}:
            self.last_active_phase = next_phase
        self.history.append(next_phase)
        return {
            "changed": True,
            "previous_phase": previous,
            "phase": next_phase,
        }


def _sensor_snapshot(
    protocol: dict[str, Any],
    *,
    phase: str,
    progress: float,
) -> dict[str, Any]:
    temp_target = _parameter_value(
        protocol,
        ("加工温度", "温度", "process::加工温度", "process::温度"),
    )
    rpm_target = _parameter_value(
        protocol,
        ("螺杆转速", "转速", "process::螺杆转速", "process::转速"),
    )
    pressure_target = _parameter_value(
        protocol,
        ("压力", "process::压力"),
    )

    ambient = 23.0
    active_phase = phase
    if phase == "PAUSED":
        active_phase = "PROCESSING"

    if active_phase in {"PREPARING", "MATERIAL_LOADING"}:
        temperature = ambient
        rpm = 0.0 if rpm_target is not None else None
        pressure = 0.0 if pressure_target is not None else None
    elif active_phase == "HEATING":
        frac = min(max((progress - 10.0) / 20.0, 0.0), 1.0)
        temperature = (
            ambient + (temp_target - ambient) * frac
            if temp_target is not None else None
        )
        rpm = (
            round(rpm_target * 0.25, 6)
            if rpm_target is not None else None
        )
        pressure = 0.0 if pressure_target is not None else None
    elif active_phase == "PROCESSING":
        temperature = temp_target
        rpm = rpm_target
        pressure = pressure_target
    elif active_phase == "COOLING":
        frac = min(max((progress - 75.0) / 13.0, 0.0), 1.0)
        temperature = (
            temp_target + (40.0 - temp_target) * frac
            if temp_target is not None else None
        )
        rpm = 0.0 if rpm_target is not None else None
        pressure = (
            round(pressure_target * 0.25, 6)
            if pressure_target is not None else None
        )
    else:
        temperature = ambient if temp_target is not None else None
        rpm = 0.0 if rpm_target is not None else None
        pressure = 0.0 if pressure_target is not None else None

    if phase == "PAUSED":
        rpm = 0.0 if rpm_target is not None else None

    return {
        "temperature_c": (
            round(float(temperature), 6)
            if temperature is not None else None
        ),
        "pressure_mpa": (
            round(float(pressure), 6)
            if pressure is not None else None
        ),
        "rpm": (
            round(float(rpm), 6)
            if rpm is not None else None
        ),
    }


class TelemetryRecorder:
    """Deterministic T30 telemetry/audit recorder.

    T30 records SimulatorDeviceAdapter state only. It does not discover or
    connect to real devices. All timestamps are a deterministic simulator
    virtual clock and every record explicitly says is_real_telemetry=false.
    """

    def __init__(
        self,
        *,
        session_id: str,
        scheduler_job_id: str,
        experiment_id: str,
        device_id: str,
        protocol: dict[str, Any],
        runtime_root: str | Path = ".runtime",
    ) -> None:
        self.session_id = _text(session_id, "session_id")
        self.scheduler_job_id = _text(scheduler_job_id, "scheduler_job_id")
        self.experiment_id = _text(experiment_id, "experiment_id")
        self.device_id = _text(device_id, "device_id")

        try:
            validate_protocol_document(protocol)
        except ExperimentProtocolValidationError as exc:
            raise TelemetryValidationError(
                "telemetry 只能绑定合法 T27 protocol",
                details={"reason": str(exc)},
            ) from exc
        if protocol.get("status") != "READY":
            raise TelemetryValidationError(
                "telemetry 只能绑定 READY protocol"
            )
        self.protocol = deepcopy(protocol)

        self.runtime_root = Path(runtime_root)
        self.root = (
            self.runtime_root
            / "v030"
            / "telemetry"
            / self.session_id
        )
        self.session_path = self.root / "session.json"
        self.telemetry_path = self.root / "telemetry.jsonl"
        self.events_path = self.root / "events.jsonl"

        self.machine = ExperimentStateMachine()
        if self.session_path.exists():
            self._state = json.loads(
                self.session_path.read_text(encoding="utf-8")
            )
            self._validate_loaded()
            history = self._state.get("phase_history") or []
            self.machine.phase = history[-1] if history else None
            active = [
                p for p in history
                if p in CORE_EXPERIMENT_PHASES[:-1]
            ]
            self.machine.last_active_phase = active[-1] if active else None
            self.machine.history = list(history)
        else:
            self._state = {
                "stage": TELEMETRY_STAGE,
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "session_id": self.session_id,
                "scheduler_job_id": self.scheduler_job_id,
                "experiment_id": self.experiment_id,
                "device_id": self.device_id,
                "protocol_id": self.protocol["protocol_id"],
                "protocol_sha256": self.protocol["content_sha256"],
                "adapter_type": "simulator",
                "is_simulator": True,
                "is_real_telemetry": False,
                "time_source": SIMULATOR_TIME_SOURCE,
                "virtual_start": SIMULATOR_VIRTUAL_START,
                "phase_history": [],
                "telemetry": [],
                "events": [],
            }
            self._append_event(
                "TELEMETRY_SESSION_CREATED",
                phase=None,
                payload={
                    "protocol_id": self.protocol["protocol_id"],
                    "real_device_connected": False,
                },
            )
            self._save()

    def _validate_loaded(self) -> None:
        expected = {
            "stage": TELEMETRY_STAGE,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "scheduler_job_id": self.scheduler_job_id,
            "experiment_id": self.experiment_id,
            "device_id": self.device_id,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_sha256": self.protocol["content_sha256"],
        }
        for key, value in expected.items():
            if self._state.get(key) != value:
                raise TelemetryValidationError(
                    f"持久化 telemetry {key} 与当前绑定不一致",
                    details={
                        "expected": value,
                        "actual": self._state.get(key),
                    },
                )
        self.verify_integrity()

    def _save(self) -> None:
        self._state["phase_history"] = list(self.machine.history)
        _atomic_json(self.session_path, self._state)
        _atomic_jsonl(
            self.telemetry_path,
            self._state.get("telemetry") or [],
        )
        _atomic_jsonl(
            self.events_path,
            self._state.get("events") or [],
        )

    def _append_event(
        self,
        event_type: str,
        *,
        phase: str | None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events = self._state.setdefault("events", [])
        seq = len(events) + 1
        previous_sha = (
            events[-1]["record_sha256"] if events else None
        )
        base = {
            "stage": TELEMETRY_STAGE,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "sequence": seq,
            "timestamp": _virtual_timestamp(seq),
            "time_source": SIMULATOR_TIME_SOURCE,
            "event_type": _text(event_type, "event_type"),
            "phase": phase,
            "device_id": self.device_id,
            "scheduler_job_id": self.scheduler_job_id,
            "experiment_id": self.experiment_id,
            "previous_sha256": previous_sha,
            "payload": deepcopy(payload or {}),
        }
        record = {
            **base,
            "record_sha256": sha256_json(base),
        }
        events.append(record)
        return record

    def capture(
        self,
        adapter: DeviceAdapter,
        *,
        alarm_code: str | None = None,
    ) -> dict[str, Any]:
        if str(getattr(adapter, "device_id", "")) != self.device_id:
            raise TelemetryValidationError(
                "adapter.device_id 与 telemetry session 不一致"
            )
        if not isinstance(adapter, SimulatorDeviceAdapter):
            raise TelemetryValidationError(
                "T30 acceptance 只允许 SimulatorDeviceAdapter；真实设备 telemetry 留到设备集成阶段"
            )

        status = adapter.status()
        job = status.get("job") or {}
        adapter_job_id = str(job.get("job_id") or "")
        protocol_id = str(
            status.get("prepared_protocol_id")
            or job.get("protocol_id")
            or ""
        )
        if protocol_id and protocol_id != self.protocol["protocol_id"]:
            raise TelemetryValidationError(
                "device 当前 protocol 与 telemetry session 不一致"
            )

        progress = _finite(job.get("progress", 0.0), "progress")
        if progress < 0 or progress > 100:
            raise TelemetryValidationError("progress 必须在 [0,100]")

        phase = phase_for_device_status(
            status,
            last_active_phase=self.machine.last_active_phase,
        )

        fingerprint = sha256_json({
            "device_state": status.get("state"),
            "adapter_job_id": adapter_job_id,
            "progress": progress,
            "phase": phase,
            "alarm_code": str(alarm_code or ""),
        })
        existing = self._state.get("telemetry") or []
        if existing and existing[-1].get("source_fingerprint") == fingerprint:
            return {
                "idempotent_replay": True,
                "record": deepcopy(existing[-1]),
            }

        transition = self.machine.transition(phase)
        if transition["changed"]:
            self._append_event(
                "PHASE_CHANGED",
                phase=phase,
                payload={
                    "previous_phase": transition["previous_phase"],
                    "phase": phase,
                },
            )

        seq = len(existing) + 1
        previous_sha = (
            existing[-1]["record_sha256"] if existing else None
        )
        sensors = _sensor_snapshot(
            self.protocol,
            phase=phase,
            progress=progress,
        )
        base = {
            "stage": TELEMETRY_STAGE,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "telemetry_id": (
                "tel_"
                + sha256_json(
                    {
                        "session_id": self.session_id,
                        "sequence": seq,
                        "fingerprint": fingerprint,
                    }
                )[:20]
            ),
            "sequence": seq,
            "timestamp": _virtual_timestamp(seq),
            "time_source": SIMULATOR_TIME_SOURCE,
            "device_id": self.device_id,
            "scheduler_job_id": self.scheduler_job_id,
            "adapter_job_id": adapter_job_id or None,
            "experiment_id": self.experiment_id,
            "protocol_id": self.protocol["protocol_id"],
            "phase": phase,
            "progress_percent": round(progress, 6),
            "elapsed_ticks": max(seq - 1, 0),
            "device_status": status.get("state"),
            "alarm_code": str(alarm_code).strip() if alarm_code else None,
            "temperature_c": sensors["temperature_c"],
            "pressure_mpa": sensors["pressure_mpa"],
            "rpm": sensors["rpm"],
            "measurement_origin": "SIMULATOR_FIXTURE",
            "synthetic": True,
            "is_real_telemetry": False,
            "source_fingerprint": fingerprint,
            "previous_sha256": previous_sha,
        }
        record = {
            **base,
            "record_sha256": sha256_json(base),
        }
        self._state.setdefault("telemetry", []).append(record)

        if alarm_code:
            self._append_event(
                "ALARM_RECORDED",
                phase=phase,
                payload={"alarm_code": str(alarm_code)},
            )

        self._save()
        return {
            "idempotent_replay": False,
            "record": deepcopy(record),
        }

    def verify_integrity(self) -> bool:
        for key in ("telemetry", "events"):
            previous = None
            for row in self._state.get(key) or []:
                if row.get("previous_sha256") != previous:
                    raise TelemetryIntegrityError(
                        f"{key} hash chain previous_sha256 不一致"
                    )
                base = {
                    k: v for k, v in row.items()
                    if k != "record_sha256"
                }
                expected = sha256_json(base)
                if row.get("record_sha256") != expected:
                    raise TelemetryIntegrityError(
                        f"{key} record_sha256 校验失败"
                    )
                previous = row["record_sha256"]
        return True

    def snapshot(self) -> dict[str, Any]:
        telemetry = self._state.get("telemetry") or []
        events = self._state.get("events") or []
        return {
            "stage": TELEMETRY_STAGE,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "scheduler_job_id": self.scheduler_job_id,
            "experiment_id": self.experiment_id,
            "device_id": self.device_id,
            "protocol_id": self.protocol["protocol_id"],
            "phase": self.machine.phase,
            "phase_history": list(self.machine.history),
            "telemetry_count": len(telemetry),
            "event_count": len(events),
            "telemetry": deepcopy(telemetry),
            "events": deepcopy(events),
            "session_json": str(self.session_path),
            "telemetry_jsonl": str(self.telemetry_path),
            "events_jsonl": str(self.events_path),
            "atomic_state_write": True,
            "hash_chain_valid": self.verify_integrity(),
            "adapter_type": "simulator",
            "is_simulator": True,
            "is_real_telemetry": False,
            "time_source": SIMULATOR_TIME_SOURCE,
            "real_device_connected": False,
        }
