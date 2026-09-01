from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
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


SAFETY_STAGE = "V0.3-T31_safety_interlock"
SAFETY_POLICY_STAGE = "V0.3-T31_safety_policy"
SAFETY_SCHEMA_VERSION = 1
SAFETY_STATES = {"SAFE", "SAFETY_STOP"}
SAFETY_VIRTUAL_START = "2026-01-01T00:00:00"

RECOVERABLE_CODES = {"RUNTIME_LIMIT"}
NONRECOVERABLE_CODES = {
    "DEVICE_ALARM",
    "DEVICE_ERROR",
    "COMMUNICATION_LOSS",
    "MANUAL_EMERGENCY_STOP",
}


class SafetyInterlockError(RuntimeError):
    code = "SAFETY_INTERLOCK_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class SafetyPolicyError(SafetyInterlockError):
    code = "SAFETY_POLICY_ERROR"


class SafetyInterlockStateError(SafetyInterlockError):
    code = "SAFETY_INTERLOCK_STATE_ERROR"


class SafetyAcknowledgementRequiredError(SafetyInterlockError):
    code = "OPERATOR_ACK_REQUIRED"


class SafetyRestartRequiredError(SafetyInterlockError):
    code = "RESTART_REQUIRED"


class SafetyIntegrityError(SafetyInterlockError):
    code = "SAFETY_AUDIT_INTEGRITY_ERROR"


def _text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SafetyPolicyError(f"{name} 不能为空")
    return text


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SafetyPolicyError(f"{name} 必须是有限数值")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SafetyPolicyError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(out):
        raise SafetyPolicyError(f"{name} 必须是有限数值")
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
    start = datetime.fromisoformat(SAFETY_VIRTUAL_START)
    return (start + timedelta(seconds=max(sequence - 1, 0))).isoformat()


def validate_safety_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise SafetyPolicyError("safety policy 必须是 object")
    if policy.get("stage") != SAFETY_POLICY_STAGE:
        raise SafetyPolicyError(
            f"policy.stage 必须是 {SAFETY_POLICY_STAGE!r}"
        )

    policy_id = _text(policy.get("policy_id"), "policy_id")
    device_id = _text(policy.get("device_id"), "device_id")

    def normalize_limits(raw: Any, name: str) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict) or not raw:
            raise SafetyPolicyError(f"{name} 必须是非空 object")
        out: dict[str, dict[str, Any]] = {}
        for field, spec in raw.items():
            field_name = _text(field, f"{name}.field")
            if not isinstance(spec, dict):
                raise SafetyPolicyError(f"{name}.{field_name} 必须是 object")
            minimum = spec.get("min")
            maximum = spec.get("max")
            if minimum is None and maximum is None:
                raise SafetyPolicyError(
                    f"{name}.{field_name} 至少需要 min/max 之一"
                )
            normalized: dict[str, Any] = {}
            if minimum is not None:
                normalized["min"] = _finite(minimum, f"{name}.{field_name}.min")
            if maximum is not None:
                normalized["max"] = _finite(maximum, f"{name}.{field_name}.max")
            if "min" in normalized and "max" in normalized:
                if normalized["min"] > normalized["max"]:
                    raise SafetyPolicyError(
                        f"{name}.{field_name} min 不能大于 max"
                    )
            unit = str(spec.get("unit") or "").strip()
            if unit:
                normalized["unit"] = unit
            out[field_name] = normalized
        return out

    protocol_limits = normalize_limits(
        policy.get("protocol_limits"), "protocol_limits"
    )
    runtime_limits = normalize_limits(
        policy.get("runtime_limits"), "runtime_limits"
    )

    alarms = policy.get("blocked_alarm_codes") or []
    if not isinstance(alarms, list):
        raise SafetyPolicyError("blocked_alarm_codes 必须是 list")
    alarms = [str(x).strip().upper() for x in alarms if str(x).strip()]
    if len(set(alarms)) != len(alarms):
        raise SafetyPolicyError("blocked_alarm_codes 不能重复")

    normalized = {
        "stage": SAFETY_POLICY_STAGE,
        "schema_version": SAFETY_SCHEMA_VERSION,
        "policy_id": policy_id,
        "device_id": device_id,
        "protocol_limits": protocol_limits,
        "runtime_limits": runtime_limits,
        "blocked_alarm_codes": alarms,
        "require_operator_ack": bool(policy.get("require_operator_ack", True)),
        "metadata": deepcopy(policy.get("metadata") or {}),
    }
    normalized["policy_sha256"] = sha256_json(normalized)
    return normalized


def _protocol_parameter_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in protocol.get("process_parameters") or []:
        source = str(item.get("source_feature") or "").strip()
        name = str(item.get("name") or "").strip()
        if source:
            result[source] = item
        if name:
            result[name] = item
    return result


def _check_numeric_limit(
    *,
    field: str,
    value: Any,
    spec: dict[str, Any],
    unit: str | None = None,
    code: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        number = _finite(value, field)
    except SafetyPolicyError:
        return {
            "code": code,
            "field": field,
            "value": value,
            "reason": "NON_NUMERIC",
        }
    if spec.get("unit") and unit is not None and str(unit) != str(spec["unit"]):
        return {
            "code": code,
            "field": field,
            "value": number,
            "unit": unit,
            "expected_unit": spec["unit"],
            "reason": "UNIT_MISMATCH",
        }
    if "min" in spec and number < float(spec["min"]):
        return {
            "code": code,
            "field": field,
            "value": number,
            "limit": float(spec["min"]),
            "direction": "BELOW_MIN",
        }
    if "max" in spec and number > float(spec["max"]):
        return {
            "code": code,
            "field": field,
            "value": number,
            "limit": float(spec["max"]),
            "direction": "ABOVE_MAX",
        }
    return None


def observation_from_telemetry(
    row: dict[str, Any],
    *,
    fixture_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SafetyPolicyError("telemetry row 必须是 object")
    observation = {
        "device_id": str(row.get("device_id") or "").strip(),
        "scheduler_job_id": row.get("scheduler_job_id"),
        "experiment_id": row.get("experiment_id"),
        "protocol_id": row.get("protocol_id"),
        "phase": row.get("phase"),
        "device_status": row.get("device_status"),
        "temperature_c": row.get("temperature_c"),
        "pressure_mpa": row.get("pressure_mpa"),
        "rpm": row.get("rpm"),
        "alarm_code": row.get("alarm_code"),
        "communication_ok": True,
        "source": "T30_TELEMETRY",
        "source_telemetry_id": row.get("telemetry_id"),
        "is_real_telemetry": bool(row.get("is_real_telemetry", False)),
        "synthetic": bool(row.get("synthetic", False)),
        "fixture_override": False,
    }
    if fixture_overrides:
        if observation["is_real_telemetry"]:
            raise SafetyPolicyError(
                "真实 telemetry 禁止使用 fixture_overrides"
            )
        allowed = {
            "temperature_c",
            "pressure_mpa",
            "rpm",
            "alarm_code",
            "communication_ok",
        }
        unknown = sorted(set(fixture_overrides) - allowed)
        if unknown:
            raise SafetyPolicyError(
                "fixture_overrides 含未知字段",
                details={"unknown": unknown},
            )
        observation.update(deepcopy(fixture_overrides))
        observation["fixture_override"] = True
        observation["source"] = "T30_TELEMETRY_WITH_SIMULATOR_FAULT_INJECTION"
    return observation


class SafetyInterlock:
    """Latched deterministic safety boundary for T31.

    T31 still runs only against SimulatorDeviceAdapter. The interlock is
    deliberately fail-closed: after a trip, no code path here can resume the
    same job without explicit operator acknowledgement and a fresh safe recheck.
    """

    def __init__(
        self,
        *,
        interlock_id: str,
        policy: dict[str, Any],
        runtime_root: str | Path = ".runtime",
    ) -> None:
        self.interlock_id = _text(interlock_id, "interlock_id")
        self.policy = validate_safety_policy(policy)
        self.device_id = self.policy["device_id"]
        self.runtime_root = Path(runtime_root)
        self.root = self.runtime_root / "v030" / "safety" / self.interlock_id
        self.state_path = self.root / "safety.json"
        self.events_path = self.root / "events.jsonl"

        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._validate_loaded()
        else:
            self._state = {
                "stage": SAFETY_STAGE,
                "schema_version": SAFETY_SCHEMA_VERSION,
                "interlock_id": self.interlock_id,
                "policy_id": self.policy["policy_id"],
                "policy_sha256": self.policy["policy_sha256"],
                "device_id": self.device_id,
                "state": "SAFE",
                "current_trip": None,
                "trip_history": [],
                "events": [],
                "adapter_type": "simulator",
                "is_simulator": True,
                "real_device_connected": False,
                "automatic_resume_allowed": False,
            }
            self._event("INTERLOCK_CREATED", payload={})
            self._save()

    def _validate_loaded(self) -> None:
        expected = {
            "stage": SAFETY_STAGE,
            "schema_version": SAFETY_SCHEMA_VERSION,
            "interlock_id": self.interlock_id,
            "policy_id": self.policy["policy_id"],
            "policy_sha256": self.policy["policy_sha256"],
            "device_id": self.device_id,
        }
        for key, value in expected.items():
            if self._state.get(key) != value:
                raise SafetyPolicyError(
                    f"持久化 safety {key} 与当前配置不一致",
                    details={"expected": value, "actual": self._state.get(key)},
                )
        self.verify_integrity()

    def _event(self, event_type: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        events = self._state.setdefault("events", [])
        seq = len(events) + 1
        previous_sha = events[-1]["record_sha256"] if events else None
        base = {
            "stage": SAFETY_STAGE,
            "schema_version": SAFETY_SCHEMA_VERSION,
            "interlock_id": self.interlock_id,
            "sequence": seq,
            "timestamp": _virtual_timestamp(seq),
            "event_type": _text(event_type, "event_type"),
            "device_id": self.device_id,
            "state": self._state.get("state"),
            "previous_sha256": previous_sha,
            "payload": deepcopy(payload),
        }
        record = dict(base)
        record["record_sha256"] = sha256_json(base)
        events.append(record)
        return record

    def _save(self) -> None:
        _atomic_json(self.state_path, self._state)
        _atomic_jsonl(self.events_path, self._state.get("events") or [])

    def _adapter_check(self, adapter: DeviceAdapter) -> None:
        if not isinstance(adapter, SimulatorDeviceAdapter):
            raise SafetyPolicyError(
                "T31 acceptance 只允许 SimulatorDeviceAdapter"
            )
        if str(getattr(adapter, "device_id", "")) != self.device_id:
            raise SafetyPolicyError(
                "adapter.device_id 与 safety policy 不一致"
            )

    def _protocol_violations(self, protocol: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            validate_protocol_document(protocol)
        except ExperimentProtocolValidationError as exc:
            return [{
                "code": "PROTOCOL_INVALID",
                "reason": str(exc),
            }]
        if protocol.get("status") != "READY":
            return [{
                "code": "PROTOCOL_NOT_READY",
                "status": protocol.get("status"),
            }]

        params = _protocol_parameter_map(protocol)
        violations: list[dict[str, Any]] = []
        for field, spec in self.policy["protocol_limits"].items():
            item = params.get(field)
            if item is None:
                continue
            issue = _check_numeric_limit(
                field=field,
                value=item.get("value"),
                spec=spec,
                unit=str(item.get("unit") or ""),
                code="PROTOCOL_LIMIT",
            )
            if issue:
                violations.append(issue)
        return violations

    def _runtime_violations(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        if str(observation.get("device_id") or "") != self.device_id:
            violations.append({
                "code": "DEVICE_ID_MISMATCH",
                "observed_device_id": observation.get("device_id"),
                "expected_device_id": self.device_id,
            })
            return violations

        if not bool(observation.get("communication_ok", True)):
            violations.append({"code": "COMMUNICATION_LOSS"})

        device_status = str(observation.get("device_status") or "").upper()
        if device_status == "ERROR":
            violations.append({"code": "DEVICE_ERROR"})

        alarm_code = str(observation.get("alarm_code") or "").strip().upper()
        if alarm_code and alarm_code in set(self.policy["blocked_alarm_codes"]):
            code = "DEVICE_ERROR" if alarm_code == "DEVICE_ERROR" else "DEVICE_ALARM"
            violations.append({
                "code": code,
                "alarm_code": alarm_code,
            })

        for field, spec in self.policy["runtime_limits"].items():
            issue = _check_numeric_limit(
                field=field,
                value=observation.get(field),
                spec=spec,
                unit=None,
                code="RUNTIME_LIMIT",
            )
            if issue:
                violations.append(issue)
        return violations

    @staticmethod
    def _trip_code(violations: list[dict[str, Any]]) -> str:
        priority = [
            "MANUAL_EMERGENCY_STOP",
            "COMMUNICATION_LOSS",
            "DEVICE_ERROR",
            "DEVICE_ALARM",
            "PROTOCOL_INVALID",
            "PROTOCOL_NOT_READY",
            "PROTOCOL_LIMIT",
            "RUNTIME_LIMIT",
            "DEVICE_ID_MISMATCH",
        ]
        present = {str(v.get("code")) for v in violations}
        for code in priority:
            if code in present:
                return code
        return "SAFETY_VIOLATION"

    @staticmethod
    def _recoverable_same_job(code: str) -> bool:
        return code in RECOVERABLE_CODES

    def _stop_adapter(self, adapter: DeviceAdapter, *, recoverable: bool) -> str:
        self._adapter_check(adapter)
        status = adapter.status()
        state = str(status.get("state") or "")
        if state == "RUNNING" and recoverable:
            try:
                adapter.pause()
                return "PAUSE"
            except DeviceAdapterError:
                try:
                    adapter.cancel()
                    return "CANCEL_FALLBACK"
                except DeviceAdapterError:
                    return "NO_ACTION_DEVICE_ERROR"
        if state in {"PREPARED", "SUBMITTED", "RUNNING", "PAUSED"}:
            try:
                adapter.cancel()
                return "CANCEL"
            except DeviceAdapterError:
                return "NO_ACTION_DEVICE_ERROR"
        return "NO_ACTION"

    def _trip(
        self,
        *,
        code: str,
        violations: list[dict[str, Any]],
        observation: dict[str, Any] | None,
        adapter: DeviceAdapter | None,
        source: str,
    ) -> dict[str, Any]:
        fingerprint = sha256_json({
            "code": code,
            "violations": violations,
            "observation": observation,
            "source": source,
        })
        current = self._state.get("current_trip")
        if (
            self._state.get("state") == "SAFETY_STOP"
            and current
            and current.get("fingerprint") == fingerprint
        ):
            return {
                "idempotent_replay": True,
                "state": "SAFETY_STOP",
                "trip": deepcopy(current),
            }

        recoverable = self._recoverable_same_job(code)
        action = "NONE"
        if adapter is not None:
            action = self._stop_adapter(adapter, recoverable=recoverable)

        trip_sequence = len(self._state.get("trip_history") or []) + 1
        trip_id = "trip_" + sha256_json({
            "interlock_id": self.interlock_id,
            "trip_sequence": trip_sequence,
            "fingerprint": fingerprint,
        })[:20]
        trip = {
            "trip_id": trip_id,
            "trip_sequence": trip_sequence,
            "code": code,
            "violations": deepcopy(violations),
            "source": source,
            "observation": deepcopy(observation),
            "fingerprint": fingerprint,
            "recoverable_same_job": recoverable,
            "stop_action": action,
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledgement_note": None,
            "last_recheck_safe": False,
        }
        self._state["state"] = "SAFETY_STOP"
        self._state["current_trip"] = trip
        self._state.setdefault("trip_history", []).append(deepcopy(trip))
        self._event("SAFETY_STOP_LATCHED", payload=deepcopy(trip))
        self._save()
        return {
            "idempotent_replay": False,
            "state": "SAFETY_STOP",
            "trip": deepcopy(trip),
        }

    def check_protocol(self, protocol: dict[str, Any]) -> dict[str, Any]:
        violations = self._protocol_violations(protocol)
        if not violations:
            return {
                "allowed": True,
                "state": self._state["state"],
                "violations": [],
            }
        result = self._trip(
            code=self._trip_code(violations),
            violations=violations,
            observation={
                "protocol_id": protocol.get("protocol_id"),
                "candidate_id": protocol.get("candidate_id"),
            },
            adapter=None,
            source="PROTOCOL_PREFLIGHT",
        )
        return {
            "allowed": False,
            "state": result["state"],
            "violations": violations,
            "trip": result["trip"],
        }

    def monitor_telemetry(
        self,
        adapter: DeviceAdapter,
        row: dict[str, Any],
        *,
        fixture_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._adapter_check(adapter)
        observation = observation_from_telemetry(
            row,
            fixture_overrides=fixture_overrides,
        )
        violations = self._runtime_violations(observation)
        if not violations:
            return {
                "safe": True,
                "state": self._state["state"],
                "violations": [],
                "observation": observation,
            }
        result = self._trip(
            code=self._trip_code(violations),
            violations=violations,
            observation=observation,
            adapter=adapter,
            source="RUNTIME_TELEMETRY",
        )
        return {
            "safe": False,
            "state": result["state"],
            "violations": violations,
            "trip": result["trip"],
            "idempotent_replay": result["idempotent_replay"],
            "observation": observation,
        }

    def monitor_health(
        self,
        adapter: DeviceAdapter,
        *,
        communication_ok: bool,
    ) -> dict[str, Any]:
        self._adapter_check(adapter)
        status = adapter.status()
        observation = {
            "device_id": self.device_id,
            "device_status": status.get("state"),
            "communication_ok": bool(communication_ok),
            "temperature_c": None,
            "pressure_mpa": None,
            "rpm": None,
            "alarm_code": None,
            "source": "DEVICE_HEALTH",
            "is_real_telemetry": False,
            "synthetic": True,
            "fixture_override": True,
        }
        violations = self._runtime_violations(observation)
        if not violations:
            return {"safe": True, "state": self._state["state"], "violations": []}
        result = self._trip(
            code=self._trip_code(violations),
            violations=violations,
            observation=observation,
            adapter=adapter,
            source="DEVICE_HEALTH",
        )
        return {
            "safe": False,
            "state": result["state"],
            "violations": violations,
            "trip": result["trip"],
        }

    def emergency_stop(
        self,
        adapter: DeviceAdapter,
        *,
        operator_id: str,
        note: str = "",
    ) -> dict[str, Any]:
        operator_id = _text(operator_id, "operator_id")
        self._adapter_check(adapter)
        violations = [{
            "code": "MANUAL_EMERGENCY_STOP",
            "operator_id": operator_id,
            "note": str(note or "").strip(),
        }]
        result = self._trip(
            code="MANUAL_EMERGENCY_STOP",
            violations=violations,
            observation={
                "device_id": self.device_id,
                "device_status": adapter.status().get("state"),
            },
            adapter=adapter,
            source="OPERATOR_ESTOP",
        )
        return result

    def acknowledge(self, *, operator_id: str, note: str = "") -> dict[str, Any]:
        operator_id = _text(operator_id, "operator_id")
        if self._state.get("state") != "SAFETY_STOP" or not self._state.get("current_trip"):
            raise SafetyInterlockStateError("当前没有 SAFETY_STOP 可确认")
        trip = self._state["current_trip"]
        if trip.get("acknowledged"):
            if trip.get("acknowledged_by") == operator_id:
                return {
                    "idempotent_replay": True,
                    "state": "SAFETY_STOP",
                    "trip": deepcopy(trip),
                }
            raise SafetyInterlockStateError(
                "该 SAFETY_STOP 已由另一 operator 确认"
            )
        trip["acknowledged"] = True
        trip["acknowledged_by"] = operator_id
        trip["acknowledgement_note"] = str(note or "").strip()
        if self._state.get("trip_history"):
            self._state["trip_history"][-1] = deepcopy(trip)
        self._event(
            "OPERATOR_ACKNOWLEDGED",
            payload={
                "trip_id": trip["trip_id"],
                "operator_id": operator_id,
                "note": trip["acknowledgement_note"],
            },
        )
        self._save()
        return {
            "idempotent_replay": False,
            "state": "SAFETY_STOP",
            "trip": deepcopy(trip),
        }

    def recheck_telemetry(
        self,
        row: dict[str, Any],
        *,
        fixture_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._state.get("state") != "SAFETY_STOP" or not self._state.get("current_trip"):
            raise SafetyInterlockStateError("recheck 需要已锁存 SAFETY_STOP")
        trip = self._state["current_trip"]
        if self.policy["require_operator_ack"] and not trip.get("acknowledged"):
            raise SafetyAcknowledgementRequiredError(
                "必须先由 operator acknowledgement 才能 safety recheck"
            )
        observation = observation_from_telemetry(
            row,
            fixture_overrides=fixture_overrides,
        )
        violations = self._runtime_violations(observation)
        safe = not violations
        trip["last_recheck_safe"] = safe
        trip["last_recheck_violations"] = deepcopy(violations)
        if self._state.get("trip_history"):
            self._state["trip_history"][-1] = deepcopy(trip)
        self._event(
            "SAFETY_RECHECK",
            payload={
                "trip_id": trip["trip_id"],
                "safe": safe,
                "violations": violations,
            },
        )
        self._save()
        return {
            "safe": safe,
            "state": "SAFETY_STOP",
            "violations": violations,
        }

    def automatic_resume(self, adapter: DeviceAdapter) -> None:
        self._adapter_check(adapter)
        if self._state.get("state") == "SAFETY_STOP":
            raise SafetyAcknowledgementRequiredError(
                "SAFETY_STOP 禁止自动 resume；必须 operator acknowledgement + recheck"
            )
        raise SafetyInterlockStateError("没有需要恢复的 SAFETY_STOP")

    def resume_after_ack(
        self,
        adapter: DeviceAdapter,
        row: dict[str, Any],
        *,
        operator_id: str,
        fixture_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operator_id = _text(operator_id, "operator_id")
        self._adapter_check(adapter)
        if self._state.get("state") != "SAFETY_STOP" or not self._state.get("current_trip"):
            raise SafetyInterlockStateError("当前没有 SAFETY_STOP")
        trip = self._state["current_trip"]
        if not trip.get("acknowledged"):
            raise SafetyAcknowledgementRequiredError("operator acknowledgement 缺失")
        if trip.get("acknowledged_by") != operator_id:
            raise SafetyAcknowledgementRequiredError(
                "resume operator 必须与 acknowledgement operator 一致"
            )
        if not trip.get("recoverable_same_job"):
            raise SafetyRestartRequiredError(
                "该安全事件不允许恢复同一 job，必须人工检查后创建新任务",
                details={"trip_code": trip.get("code")},
            )

        recheck = self.recheck_telemetry(
            row,
            fixture_overrides=fixture_overrides,
        )
        if not recheck["safe"]:
            raise SafetyInterlockStateError(
                "safety recheck 仍不安全，禁止 resume",
                details={"violations": recheck["violations"]},
            )
        status = adapter.status()
        if status.get("state") != "PAUSED":
            raise SafetyInterlockStateError(
                "可恢复 SAFETY_STOP 要求设备处于 PAUSED",
                details={"device_state": status.get("state")},
            )
        adapter.resume()
        completed_trip = deepcopy(trip)
        completed_trip["cleared_by"] = operator_id
        completed_trip["cleared_via"] = "SAFE_RECHECK_AND_RESUME"
        self._state["trip_history"][-1] = completed_trip
        self._state["current_trip"] = None
        self._state["state"] = "SAFE"
        self._event(
            "SAFETY_STOP_CLEARED_AND_RESUMED",
            payload={
                "trip_id": completed_trip["trip_id"],
                "operator_id": operator_id,
            },
        )
        self._save()
        return {
            "state": "SAFE",
            "device_state": adapter.status().get("state"),
            "resumed": True,
        }

    def clear_terminal_stop(
        self,
        adapter: DeviceAdapter,
        *,
        operator_id: str,
    ) -> dict[str, Any]:
        operator_id = _text(operator_id, "operator_id")
        self._adapter_check(adapter)
        if self._state.get("state") != "SAFETY_STOP" or not self._state.get("current_trip"):
            raise SafetyInterlockStateError("当前没有 SAFETY_STOP")
        trip = self._state["current_trip"]
        if not trip.get("acknowledged") or trip.get("acknowledged_by") != operator_id:
            raise SafetyAcknowledgementRequiredError("operator acknowledgement 缺失")
        if trip.get("recoverable_same_job"):
            raise SafetyInterlockStateError(
                "可恢复事件应使用 resume_after_ack，而不是 clear_terminal_stop"
            )
        device_state = str(adapter.status().get("state") or "")
        if device_state not in {"CANCELLED", "ERROR", "IDLE", "DISCONNECTED", "COMPLETED"}:
            raise SafetyInterlockStateError(
                "nonrecoverable stop 只有在设备非活动状态才能清除",
                details={"device_state": device_state},
            )
        completed_trip = deepcopy(trip)
        completed_trip["cleared_by"] = operator_id
        completed_trip["cleared_via"] = "TERMINAL_ACK_NEW_JOB_REQUIRED"
        self._state["trip_history"][-1] = completed_trip
        self._state["current_trip"] = None
        self._state["state"] = "SAFE"
        self._event(
            "SAFETY_STOP_CLEARED_RESTART_REQUIRED",
            payload={
                "trip_id": completed_trip["trip_id"],
                "operator_id": operator_id,
                "device_state": device_state,
            },
        )
        self._save()
        return {
            "state": "SAFE",
            "same_job_resumed": False,
            "new_job_required": True,
            "device_state": device_state,
        }

    def verify_integrity(self) -> bool:
        previous = None
        for row in self._state.get("events") or []:
            if row.get("previous_sha256") != previous:
                raise SafetyIntegrityError(
                    "safety event hash chain previous_sha256 不一致"
                )
            base = {k: v for k, v in row.items() if k != "record_sha256"}
            expected = sha256_json(base)
            if row.get("record_sha256") != expected:
                raise SafetyIntegrityError("safety event record_sha256 校验失败")
            previous = row["record_sha256"]
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "stage": SAFETY_STAGE,
            "schema_version": SAFETY_SCHEMA_VERSION,
            "interlock_id": self.interlock_id,
            "policy_id": self.policy["policy_id"],
            "device_id": self.device_id,
            "state": self._state["state"],
            "current_trip": deepcopy(self._state.get("current_trip")),
            "trip_history": deepcopy(self._state.get("trip_history") or []),
            "event_count": len(self._state.get("events") or []),
            "events": deepcopy(self._state.get("events") or []),
            "state_json": str(self.state_path),
            "events_jsonl": str(self.events_path),
            "hash_chain_valid": self.verify_integrity(),
            "atomic_state_write": True,
            "automatic_resume_allowed": False,
            "adapter_type": "simulator",
            "is_simulator": True,
            "real_device_connected": False,
        }
