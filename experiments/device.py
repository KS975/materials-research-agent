from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from .protocol import (
    ExperimentProtocolValidationError,
    sha256_json,
    validate_protocol_document,
)


DEVICE_STAGE = "V0.3-T28_device_adapter"
DEVICE_PROFILE_STAGE = "V0.3-T28_device_profile"
DEVICE_SCHEMA_VERSION = 1

DEVICE_STATES = {
    "DISCONNECTED",
    "IDLE",
    "PREPARED",
    "SUBMITTED",
    "RUNNING",
    "PAUSED",
    "COMPLETED",
    "CANCELLED",
    "ERROR",
}
ACTIVE_DEVICE_STATES = {"PREPARED", "SUBMITTED", "RUNNING", "PAUSED"}
TERMINAL_DEVICE_STATES = {"COMPLETED", "CANCELLED", "ERROR"}


class DeviceAdapterError(RuntimeError):
    code = "DEVICE_ADAPTER_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class DeviceOfflineError(DeviceAdapterError):
    code = "OFFLINE"


class DeviceBusyError(DeviceAdapterError):
    code = "BUSY"


class DeviceUnsupportedProtocolError(DeviceAdapterError):
    code = "UNSUPPORTED_PROTOCOL"


class DeviceExecutionError(DeviceAdapterError):
    code = "DEVICE_ERROR"


class DeviceStateError(DeviceAdapterError):
    code = "INVALID_STATE"


class DeviceCapabilityError(DeviceAdapterError):
    code = "UNSUPPORTED_OPERATION"


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DeviceAdapterError(f"{name} 不能为空")
    return text


def validate_device_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise DeviceAdapterError("device profile 必须是 object")
    if profile.get("stage") != DEVICE_PROFILE_STAGE:
        raise DeviceAdapterError(
            f"profile.stage 必须是 {DEVICE_PROFILE_STAGE!r}"
        )

    device_id = _nonempty_text(profile.get("device_id"), "device_id")
    name = _nonempty_text(profile.get("name"), "name")
    adapter_type = _nonempty_text(profile.get("adapter_type"), "adapter_type")
    if adapter_type != "simulator":
        raise DeviceAdapterError(
            "T28 只允许 adapter_type='simulator'；真实设备 Adapter 留到后续集成"
        )

    roles = profile.get("supported_roles")
    if not isinstance(roles, list) or not roles:
        raise DeviceAdapterError("supported_roles 必须是非空 list")
    roles = [str(x).strip() for x in roles]
    if any(not x for x in roles) or len(set(roles)) != len(roles):
        raise DeviceAdapterError("supported_roles 不能为空或重复")

    template_ids = profile.get("supported_template_ids")
    if template_ids is None:
        template_ids = []
    if not isinstance(template_ids, list):
        raise DeviceAdapterError("supported_template_ids 必须是 list")
    template_ids = [str(x).strip() for x in template_ids if str(x).strip()]

    capabilities = profile.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        raise DeviceAdapterError("capabilities 必须是 object")

    progress_per_tick = capabilities.get("progress_per_tick", 50)
    if isinstance(progress_per_tick, bool):
        raise DeviceAdapterError("progress_per_tick 必须是有限正数")
    try:
        progress_per_tick = float(progress_per_tick)
    except (TypeError, ValueError) as exc:
        raise DeviceAdapterError("progress_per_tick 必须是有限正数") from exc
    if not math.isfinite(progress_per_tick) or progress_per_tick <= 0:
        raise DeviceAdapterError("progress_per_tick 必须是有限正数")

    fault_injection = profile.get("fault_injection") or {}
    if not isinstance(fault_injection, dict):
        raise DeviceAdapterError("fault_injection 必须是 object")

    normalized = {
        "stage": DEVICE_PROFILE_STAGE,
        "schema_version": DEVICE_SCHEMA_VERSION,
        "device_id": device_id,
        "name": name,
        "adapter_type": adapter_type,
        "online": bool(profile.get("online", True)),
        "supported_roles": roles,
        "supported_template_ids": template_ids,
        "capabilities": {
            "pause": bool(capabilities.get("pause", True)),
            "cancel": bool(capabilities.get("cancel", True)),
            "progress_per_tick": progress_per_tick,
        },
        "fault_injection": {
            "connect": bool(fault_injection.get("connect", False)),
            "prepare": bool(fault_injection.get("prepare", False)),
            "submit": bool(fault_injection.get("submit", False)),
            "start": bool(fault_injection.get("start", False)),
            "tick": bool(fault_injection.get("tick", False)),
            "read_result": bool(fault_injection.get("read_result", False)),
        },
        "metadata": deepcopy(profile.get("metadata") or {}),
    }
    normalized["profile_sha256"] = sha256_json(normalized)
    return normalized


def protocol_device_roles(protocol: dict[str, Any]) -> list[str]:
    roles: set[str] = set()
    for step in (protocol.get("process_steps") or []):
        role = str(step.get("device_role") or "").strip()
        if role:
            roles.add(role)
    for step in (protocol.get("measurement_steps") or []):
        role = str(step.get("device_role") or "").strip()
        if role:
            roles.add(role)
    return sorted(roles)


def deterministic_job_id(device_id: str, protocol_id: str) -> str:
    digest = sha256_json({
        "stage": DEVICE_STAGE,
        "device_id": str(device_id),
        "protocol_id": str(protocol_id),
        "adapter": "simulator_v1",
    })
    return f"job_{digest[:20]}"


class DeviceAdapter(ABC):
    """Uniform device boundary introduced in V0.3-T28.

    The abstract interface intentionally accepts only T27 protocol documents.
    T28 itself never talks to a real instrument. Real-device implementations
    must be separate subclasses and must preserve the same state/safety contract.
    """

    @abstractmethod
    def connect(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_protocol(self, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def pause(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resume(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def read_result(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> dict[str, Any]:
        raise NotImplementedError


class SimulatorDeviceAdapter(DeviceAdapter):
    """Deterministic in-process simulator for adapter contract testing only.

    The simulator result is explicitly marked synthetic and must not be treated
    as a real measurement. T32 will later decide how device results are ingested.
    """

    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = validate_device_profile(profile)
        self.device_id = self.profile["device_id"]
        self.state = "DISCONNECTED"
        self.connected = False
        self._prepared_protocol: dict[str, Any] | None = None
        self._job: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None
        self._events: list[dict[str, Any]] = []
        self._event("ADAPTER_CREATED")

    def _event(self, event_type: str, **payload: Any) -> None:
        self._events.append({
            "sequence": len(self._events) + 1,
            "event_type": event_type,
            "device_state": self.state,
            "job_id": self._job.get("job_id") if self._job else None,
            "payload": deepcopy(payload),
        })

    def _set_state(self, state: str, event_type: str, **payload: Any) -> None:
        if state not in DEVICE_STATES:
            raise DeviceStateError(f"未知 device state: {state}")
        self.state = state
        if self._job is not None:
            self._job["state"] = state
        self._event(event_type, **payload)

    def _fault(self, operation: str) -> bool:
        return bool(self.profile["fault_injection"].get(operation, False))

    def _require_connected(self) -> None:
        if not self.connected or self.state == "DISCONNECTED":
            raise DeviceStateError("设备尚未连接")
        if not self.profile["online"]:
            raise DeviceOfflineError(f"设备离线: {self.device_id}")

    def _validate_protocol_compatibility(self, protocol: dict[str, Any]) -> None:
        try:
            validate_protocol_document(protocol)
        except ExperimentProtocolValidationError as exc:
            raise DeviceUnsupportedProtocolError(
                "T27 protocol 校验失败",
                details={"reason": str(exc)},
            ) from exc

        if protocol.get("status") != "READY":
            raise DeviceUnsupportedProtocolError(
                "只有 T27 READY protocol 才能进入 Device Adapter",
                details={
                    "protocol_id": protocol.get("protocol_id"),
                    "protocol_status": protocol.get("status"),
                },
            )

        supported_templates = set(self.profile["supported_template_ids"])
        if supported_templates and protocol.get("template_id") not in supported_templates:
            raise DeviceUnsupportedProtocolError(
                "设备不支持该 protocol template",
                details={
                    "template_id": protocol.get("template_id"),
                    "supported_template_ids": sorted(supported_templates),
                },
            )

        required_roles = set(protocol_device_roles(protocol))
        supported_roles = set(self.profile["supported_roles"])
        missing = sorted(required_roles - supported_roles)
        if missing:
            raise DeviceUnsupportedProtocolError(
                "设备缺少 protocol 所需 role",
                details={
                    "missing_roles": missing,
                    "required_roles": sorted(required_roles),
                    "supported_roles": sorted(supported_roles),
                },
            )

    def connect(self) -> dict[str, Any]:
        if self.connected and self.state != "DISCONNECTED":
            return {"idempotent_replay": True, **self.status()}
        if not self.profile["online"]:
            raise DeviceOfflineError(f"设备离线: {self.device_id}")
        if self._fault("connect"):
            self.state = "ERROR"
            self._event("CONNECT_ERROR")
            raise DeviceExecutionError("模拟设备连接故障")
        self.connected = True
        self._set_state("IDLE", "CONNECTED")
        return {"idempotent_replay": False, **self.status()}

    def health_check(self) -> dict[str, Any]:
        self._require_connected()
        healthy = self.state != "ERROR"
        result = {
            "device_id": self.device_id,
            "online": self.profile["online"],
            "connected": self.connected,
            "healthy": healthy,
            "state": self.state,
            "adapter_type": "simulator",
            "is_simulator": True,
        }
        self._event("HEALTH_CHECK", healthy=healthy)
        return result

    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]:
        self._require_connected()
        self._validate_protocol_compatibility(protocol)

        if self._fault("prepare"):
            self._set_state("ERROR", "PREPARE_ERROR")
            raise DeviceExecutionError("模拟设备 prepare 故障")

        protocol_id = protocol["protocol_id"]
        if self.state == "PREPARED" and self._prepared_protocol is not None:
            if self._prepared_protocol["protocol_id"] == protocol_id:
                return {"idempotent_replay": True, **self.status()}
            raise DeviceBusyError(
                "设备已有已准备 protocol",
                details={"active_protocol_id": self._prepared_protocol["protocol_id"]},
            )
        if self.state in {"SUBMITTED", "RUNNING", "PAUSED"}:
            active = self._job.get("protocol_id") if self._job else None
            raise DeviceBusyError(
                "设备已有活动 job",
                details={"active_protocol_id": active, "state": self.state},
            )
        if self.state == "ERROR":
            raise DeviceStateError("设备处于 ERROR，必须 disconnect 后重新建立会话")
        if self.state not in {"IDLE", "COMPLETED", "CANCELLED"}:
            raise DeviceStateError(f"当前状态不能 prepare: {self.state}")

        self._prepared_protocol = deepcopy(protocol)
        self._job = None
        self._result = None
        self._set_state("PREPARED", "PROTOCOL_PREPARED", protocol_id=protocol_id)
        return {"idempotent_replay": False, **self.status()}

    def submit_protocol(self, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_connected()

        if protocol is not None:
            self._validate_protocol_compatibility(protocol)

        # Replaying the same submission after SUBMITTED/RUNNING/PAUSED is safe.
        if self.state in {"SUBMITTED", "RUNNING", "PAUSED"} and self._job is not None:
            if protocol is None or protocol.get("protocol_id") == self._job["protocol_id"]:
                return {"idempotent_replay": True, **self.status()}
            raise DeviceBusyError(
                "设备已有活动 job，不能提交另一 protocol",
                details={"job_id": self._job["job_id"]},
            )

        if self.state != "PREPARED" or self._prepared_protocol is None:
            raise DeviceStateError("submit_protocol 前必须先 prepare READY protocol")

        prepared_id = self._prepared_protocol["protocol_id"]
        if protocol is not None and protocol["protocol_id"] != prepared_id:
            raise DeviceStateError(
                "submit_protocol 与已 prepare 的 protocol 不一致",
                details={
                    "prepared_protocol_id": prepared_id,
                    "submitted_protocol_id": protocol["protocol_id"],
                },
            )

        if self._fault("submit"):
            self._set_state("ERROR", "SUBMIT_ERROR")
            raise DeviceExecutionError("模拟设备 submit 故障")

        job_id = deterministic_job_id(self.device_id, prepared_id)
        self._job = {
            "job_id": job_id,
            "protocol_id": prepared_id,
            "candidate_id": self._prepared_protocol["candidate_id"],
            "state": "SUBMITTED",
            "progress": 0.0,
        }
        self._set_state("SUBMITTED", "PROTOCOL_SUBMITTED", protocol_id=prepared_id)
        return {"idempotent_replay": False, **self.status()}

    def start(self) -> dict[str, Any]:
        self._require_connected()
        if self.state != "SUBMITTED" or self._job is None:
            raise DeviceStateError(f"start 需要 SUBMITTED，当前为 {self.state}")
        if self._fault("start"):
            self._set_state("ERROR", "START_ERROR")
            raise DeviceExecutionError("模拟设备 start 故障")
        self._set_state("RUNNING", "JOB_STARTED")
        return self.status()

    def pause(self) -> dict[str, Any]:
        self._require_connected()
        if not self.profile["capabilities"]["pause"]:
            raise DeviceCapabilityError("设备不支持 pause")
        if self.state != "RUNNING":
            raise DeviceStateError(f"pause 需要 RUNNING，当前为 {self.state}")
        self._set_state("PAUSED", "JOB_PAUSED")
        return self.status()

    def resume(self) -> dict[str, Any]:
        self._require_connected()
        if self.state != "PAUSED":
            raise DeviceStateError(f"resume 需要 PAUSED，当前为 {self.state}")
        self._set_state("RUNNING", "JOB_RESUMED")
        return self.status()

    def _make_result(self) -> dict[str, Any]:
        if self._prepared_protocol is None or self._job is None:
            raise DeviceExecutionError("无法生成 simulator result：缺少 protocol/job")
        protocol = self._prepared_protocol
        measurement_by_metric = {
            str(x.get("metric")): x
            for x in protocol.get("measurement_steps") or []
        }
        outputs = []
        for index, expected in enumerate(protocol.get("expected_outputs") or []):
            metric = str(expected.get("metric") or "")
            unit = str(expected.get("unit") or "")
            seed_text = f"{protocol['protocol_id']}|{metric}|{index}"
            seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16)
            # Deterministic engineering-only simulator value in a plausible range.
            value = round(35.0 + (seed % 2000) / 100.0, 6)
            measurement = measurement_by_metric.get(metric) or {}
            outputs.append({
                "metric": metric,
                "value": value,
                "unit": unit,
                "condition_signature": str(measurement.get("condition_signature") or ""),
                "required": bool(expected.get("required", True)),
            })

        base = {
            "stage": DEVICE_STAGE,
            "schema_version": DEVICE_SCHEMA_VERSION,
            "job_id": self._job["job_id"],
            "device_id": self.device_id,
            "protocol_id": protocol["protocol_id"],
            "candidate_id": protocol["candidate_id"],
            "status": "COMPLETED",
            "measurement_origin": "SIMULATOR_FIXTURE",
            "synthetic": True,
            "is_real_measurement": False,
            "outputs": outputs,
        }
        digest = sha256_json(base)
        return {
            **base,
            "result_id": f"simres_{digest[:20]}",
            "content_sha256": digest,
        }

    def tick(self, *, steps: int = 1) -> dict[str, Any]:
        """Advance the deterministic simulator; not part of real adapter API."""
        self._require_connected()
        if self.state != "RUNNING" or self._job is None:
            raise DeviceStateError(f"tick 需要 RUNNING，当前为 {self.state}")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise DeviceStateError("steps 必须是正整数")
        if self._fault("tick"):
            self._set_state("ERROR", "TICK_ERROR")
            raise DeviceExecutionError("模拟设备运行故障")

        increment = self.profile["capabilities"]["progress_per_tick"] * steps
        self._job["progress"] = min(100.0, float(self._job["progress"]) + increment)
        self._event("SIMULATOR_TICK", progress=self._job["progress"])
        if self._job["progress"] >= 100.0:
            self._result = self._make_result()
            self._set_state("COMPLETED", "JOB_COMPLETED")
        return self.status()

    def run_to_completion(self, *, max_ticks: int = 1000) -> dict[str, Any]:
        ticks = 0
        while self.state == "RUNNING" and ticks < max_ticks:
            self.tick()
            ticks += 1
        if self.state != "COMPLETED":
            raise DeviceExecutionError(
                "simulator 未在 max_ticks 内完成",
                details={"max_ticks": max_ticks, "state": self.state},
            )
        return self.status()

    def cancel(self) -> dict[str, Any]:
        self._require_connected()
        if not self.profile["capabilities"]["cancel"]:
            raise DeviceCapabilityError("设备不支持 cancel")
        if self.state not in {"PREPARED", "SUBMITTED", "RUNNING", "PAUSED"}:
            raise DeviceStateError(
                f"cancel 只允许活动状态，当前为 {self.state}"
            )
        self._set_state("CANCELLED", "JOB_CANCELLED")
        return self.status()

    def status(self) -> dict[str, Any]:
        job = deepcopy(self._job) if self._job else None
        return {
            "stage": DEVICE_STAGE,
            "schema_version": DEVICE_SCHEMA_VERSION,
            "device_id": self.device_id,
            "name": self.profile["name"],
            "adapter_type": "simulator",
            "is_simulator": True,
            "online": bool(self.profile["online"]),
            "connected": bool(self.connected),
            "state": self.state,
            "prepared_protocol_id": (
                self._prepared_protocol.get("protocol_id")
                if self._prepared_protocol else None
            ),
            "job": job,
            "result_ready": self._result is not None and self.state == "COMPLETED",
        }

    def read_result(self) -> dict[str, Any]:
        self._require_connected()
        if self.state != "COMPLETED" or self._result is None:
            raise DeviceStateError(
                f"read_result 需要 COMPLETED，当前为 {self.state}"
            )
        if self._fault("read_result"):
            self._set_state("ERROR", "READ_RESULT_ERROR")
            raise DeviceExecutionError("模拟设备 read_result 故障")
        self._event("RESULT_READ", result_id=self._result["result_id"])
        return deepcopy(self._result)

    def disconnect(self) -> dict[str, Any]:
        if not self.connected and self.state == "DISCONNECTED":
            return {"idempotent_replay": True, **self.status()}
        if self.state in ACTIVE_DEVICE_STATES:
            raise DeviceBusyError(
                "活动 job 存在时禁止 disconnect；请先完成或 cancel",
                details={"state": self.state},
            )
        self.connected = False
        self.state = "DISCONNECTED"
        self._event("DISCONNECTED")
        return {"idempotent_replay": False, **self.status()}

    def events(self) -> list[dict[str, Any]]:
        return deepcopy(self._events)
