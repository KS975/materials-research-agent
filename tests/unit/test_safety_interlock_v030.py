from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.device import DeviceAdapterError, SimulatorDeviceAdapter
from experiments.safety import (
    SafetyAcknowledgementRequiredError,
    SafetyIntegrityError,
    SafetyInterlock,
    SafetyInterlockStateError,
    SafetyPolicyError,
    SafetyRestartRequiredError,
    observation_from_telemetry,
    validate_safety_policy,
)
from experiments.telemetry import TelemetryRecorder
from scripts.build_v030_t31_fixture import (
    device_profile,
    normal_protocol,
    overlimit_protocol,
    safety_policy,
)


def _running(device_id: str, protocol: dict):
    adapter = SimulatorDeviceAdapter(device_profile(device_id))
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    return adapter


def _processing(tmp_path, device_id="SIM_T31_MAIN"):
    protocol = normal_protocol()
    adapter = _running(device_id, protocol)
    recorder = TelemetryRecorder(
        session_id="TEL_" + device_id,
        scheduler_job_id="sched_" + device_id,
        experiment_id=protocol["candidate_id"],
        device_id=device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    row = recorder.capture(adapter)["record"]
    while row["phase"] != "PROCESSING":
        adapter.tick()
        row = recorder.capture(adapter)["record"]
    return protocol, adapter, recorder, row


def test_policy_validation():
    policy = validate_safety_policy(safety_policy())
    assert policy["policy_id"] == "V030_T31_POLICY_V1"
    assert policy["runtime_limits"]["temperature_c"]["max"] == 245.0


def test_normal_protocol_passes_preflight(tmp_path):
    interlock = SafetyInterlock(
        interlock_id="PREFLIGHT_SAFE",
        policy=safety_policy(),
        runtime_root=tmp_path,
    )
    result = interlock.check_protocol(normal_protocol())
    assert result["allowed"] is True
    assert interlock.snapshot()["state"] == "SAFE"


def test_t27_ready_can_still_be_blocked_by_stricter_t31_policy(tmp_path):
    protocol = overlimit_protocol()
    assert protocol["status"] == "READY"
    interlock = SafetyInterlock(
        interlock_id="PREFLIGHT_BLOCK",
        policy=safety_policy(),
        runtime_root=tmp_path,
    )
    result = interlock.check_protocol(protocol)
    assert result["allowed"] is False
    assert result["state"] == "SAFETY_STOP"
    assert result["trip"]["code"] == "PROTOCOL_LIMIT"


def test_runtime_overtemperature_pauses_device(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="TEMP",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    result = interlock.monitor_telemetry(
        adapter,
        row,
        fixture_overrides={"temperature_c": 250.0},
    )
    assert result["state"] == "SAFETY_STOP"
    assert result["trip"]["code"] == "RUNTIME_LIMIT"
    assert adapter.status()["state"] == "PAUSED"
    assert result["trip"]["recoverable_same_job"] is True


def test_runtime_overpressure_pauses_device(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path, "SIM_T31_PRESSURE")
    interlock = SafetyInterlock(
        interlock_id="PRESSURE",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    result = interlock.monitor_telemetry(
        adapter,
        row,
        fixture_overrides={"pressure_mpa": 13.0},
    )
    assert result["trip"]["code"] == "RUNTIME_LIMIT"
    assert adapter.status()["state"] == "PAUSED"


def test_alarm_cancels_same_job(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path, "SIM_T31_ALARM")
    interlock = SafetyInterlock(
        interlock_id="ALARM",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    result = interlock.monitor_telemetry(
        adapter,
        row,
        fixture_overrides={"alarm_code": "SENSOR_FAULT"},
    )
    assert result["trip"]["code"] == "DEVICE_ALARM"
    assert result["trip"]["recoverable_same_job"] is False
    assert adapter.status()["state"] == "CANCELLED"


def test_communication_loss_cancels_job(tmp_path):
    protocol = normal_protocol()
    adapter = _running("SIM_T31_COMM", protocol)
    interlock = SafetyInterlock(
        interlock_id="COMM",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    result = interlock.monitor_health(adapter, communication_ok=False)
    assert result["trip"]["code"] == "COMMUNICATION_LOSS"
    assert adapter.status()["state"] == "CANCELLED"


def test_manual_estop_is_latched_and_nonrecoverable(tmp_path):
    protocol = normal_protocol()
    adapter = _running("SIM_T31_ESTOP", protocol)
    interlock = SafetyInterlock(
        interlock_id="ESTOP",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    result = interlock.emergency_stop(
        adapter,
        operator_id="op1",
        note="test",
    )
    assert result["state"] == "SAFETY_STOP"
    assert result["trip"]["code"] == "MANUAL_EMERGENCY_STOP"
    assert adapter.status()["state"] == "CANCELLED"


def test_automatic_resume_is_always_blocked_when_latched(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="NO_AUTO",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    with pytest.raises(SafetyAcknowledgementRequiredError):
        interlock.automatic_resume(adapter)
    assert adapter.status()["state"] == "PAUSED"


def test_acknowledgement_alone_does_not_resume(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="ACK_ONLY",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    interlock.acknowledge(operator_id="op1")
    assert interlock.snapshot()["state"] == "SAFETY_STOP"
    assert adapter.status()["state"] == "PAUSED"


def test_unsafe_recheck_blocks_resume(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="UNSAFE_RECHECK",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    interlock.acknowledge(operator_id="op1")
    result = interlock.recheck_telemetry(
        row, fixture_overrides={"temperature_c": 250.0}
    )
    assert result["safe"] is False
    with pytest.raises(SafetyInterlockStateError):
        interlock.resume_after_ack(
            adapter,
            row,
            operator_id="op1",
            fixture_overrides={"temperature_c": 250.0},
        )


def test_safe_recheck_after_ack_can_resume_recoverable_trip(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="RESUME",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    interlock.acknowledge(operator_id="op1")
    safe_paused_row = recorder.capture(adapter)["record"]
    result = interlock.resume_after_ack(
        adapter,
        safe_paused_row,
        operator_id="op1",
    )
    assert result["resumed"] is True
    assert result["device_state"] == "RUNNING"
    assert interlock.snapshot()["state"] == "SAFE"


def test_nonrecoverable_trip_requires_new_job(tmp_path):
    protocol = normal_protocol()
    adapter = _running("SIM_T31_ESTOP2", protocol)
    interlock = SafetyInterlock(
        interlock_id="ESTOP2",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.emergency_stop(adapter, operator_id="op2")
    interlock.acknowledge(operator_id="op2")
    row = {
        "device_id": adapter.device_id,
        "scheduler_job_id": "x",
        "experiment_id": protocol["candidate_id"],
        "protocol_id": protocol["protocol_id"],
        "phase": "CANCELLED",
        "device_status": "CANCELLED",
        "temperature_c": 23,
        "pressure_mpa": 0,
        "rpm": 0,
        "alarm_code": None,
        "is_real_telemetry": False,
        "synthetic": True,
    }
    with pytest.raises(SafetyRestartRequiredError):
        interlock.resume_after_ack(adapter, row, operator_id="op2")
    cleared = interlock.clear_terminal_stop(adapter, operator_id="op2")
    assert cleared["new_job_required"] is True
    assert interlock.snapshot()["state"] == "SAFE"


def test_trip_replay_is_idempotent(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="IDEMP",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    first = interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    second = interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert len(interlock.snapshot()["trip_history"]) == 1


def test_device_error_latches_safety_stop(tmp_path):
    protocol = normal_protocol()
    adapter = SimulatorDeviceAdapter(device_profile("SIM_T31_FAULT_TEST", fault=True))
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    recorder = TelemetryRecorder(
        session_id="FAULT_TEST",
        scheduler_job_id="sched_fault_test",
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    recorder.capture(adapter)
    with pytest.raises(DeviceAdapterError) as caught:
        adapter.tick()
    row = recorder.capture(adapter, alarm_code=caught.value.code)["record"]
    interlock = SafetyInterlock(
        interlock_id="FAULT_TEST",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    result = interlock.monitor_telemetry(adapter, row)
    assert result["trip"]["code"] == "DEVICE_ERROR"
    assert result["state"] == "SAFETY_STOP"
    assert adapter.status()["state"] == "ERROR"


def test_real_telemetry_cannot_be_fixture_overridden():
    row = {
        "device_id": "D1",
        "is_real_telemetry": True,
        "temperature_c": 20,
    }
    with pytest.raises(SafetyPolicyError):
        observation_from_telemetry(
            row,
            fixture_overrides={"temperature_c": 999},
        )


def test_audit_hash_chain_and_reload(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="AUDIT",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    interlock.acknowledge(operator_id="op1")
    assert interlock.verify_integrity() is True
    reloaded = SafetyInterlock(
        interlock_id="AUDIT",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    assert reloaded.verify_integrity() is True
    assert reloaded.snapshot()["state"] == "SAFETY_STOP"


def test_audit_tampering_is_detected(tmp_path):
    protocol, adapter, recorder, row = _processing(tmp_path)
    interlock = SafetyInterlock(
        interlock_id="TAMPER",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    interlock.monitor_telemetry(
        adapter, row, fixture_overrides={"temperature_c": 250.0}
    )
    path = Path(interlock.snapshot()["state_json"])
    state = json.loads(path.read_text(encoding="utf-8"))
    state["events"][-1]["payload"]["trip_id"] = "tampered"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(SafetyIntegrityError):
        SafetyInterlock(
            interlock_id="TAMPER",
            policy=safety_policy(adapter.device_id),
            runtime_root=tmp_path,
        )
