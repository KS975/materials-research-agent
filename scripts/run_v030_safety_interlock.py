from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from experiments.device import DeviceAdapterError, SimulatorDeviceAdapter
from experiments.safety import (
    SafetyAcknowledgementRequiredError,
    SafetyInterlock,
    SafetyInterlockStateError,
    SafetyRestartRequiredError,
)
from experiments.telemetry import TelemetryRecorder
from scripts.build_v030_t31_fixture import (
    device_profile,
    normal_protocol,
    overlimit_protocol,
    safety_policy,
)


def _running_adapter(device_id: str, protocol: dict):
    adapter = SimulatorDeviceAdapter(device_profile(device_id))
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    return adapter


def _processing_row(root: Path, session_id: str, adapter, protocol):
    recorder = TelemetryRecorder(
        session_id=session_id,
        scheduler_job_id="sched_" + session_id.lower(),
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=root,
    )
    row = recorder.capture(adapter)["record"]
    while row["phase"] != "PROCESSING":
        adapter.tick()
        row = recorder.capture(adapter)["record"]
    return recorder, row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(args.runtime_root)

    if args.reset:
        safety_root = root / "v030" / "safety"
        if safety_root.exists():
            shutil.rmtree(safety_root)
        telemetry_root = root / "v030" / "telemetry"
        for name in [
            "V030_T31_RUNTIME",
            "V030_T31_PRESSURE",
            "V030_T31_ALARM",
            "V030_T31_ESTOP",
            "V030_T31_DEVICE_ERROR",
        ]:
            target = telemetry_root / name
            if target.exists():
                shutil.rmtree(target)

    protocol = normal_protocol()

    print("V0.3-T31 SAFETY INTERLOCK")
    print()
    print("BOUNDARY")
    print("device_adapter: SimulatorDeviceAdapter only")
    print("real_device_connected: false")
    print("safety_state: SAFE / SAFETY_STOP")
    print("automatic_resume_allowed: false")
    print()

    # 1. Independent protocol preflight safety policy.
    preflight = SafetyInterlock(
        interlock_id="V030_T31_PREFLIGHT",
        policy=safety_policy("SIM_T31_MAIN"),
        runtime_root=root,
    )
    safe_preflight = preflight.check_protocol(protocol)
    blocked_preflight = SafetyInterlock(
        interlock_id="V030_T31_PREFLIGHT_BLOCK",
        policy=safety_policy("SIM_T31_MAIN"),
        runtime_root=root,
    ).check_protocol(overlimit_protocol())

    print("PROTOCOL PREFLIGHT")
    print(f"normal_protocol_allowed: {str(safe_preflight['allowed']).lower()}")
    print(f"overlimit_protocol_state: {blocked_preflight['state']}")
    print(f"overlimit_code: {blocked_preflight['trip']['code']}")
    print(f"overlimit_value_c: {blocked_preflight['violations'][0]['value']:.6f}")
    print(f"overlimit_policy_max_c: {blocked_preflight['violations'][0]['limit']:.6f}")
    print()

    # 2. Recoverable runtime temperature violation.
    adapter = _running_adapter("SIM_T31_MAIN", protocol)
    recorder, normal_row = _processing_row(root, "V030_T31_RUNTIME", adapter, protocol)
    runtime = SafetyInterlock(
        interlock_id="V030_T31_RUNTIME",
        policy=safety_policy(adapter.device_id),
        runtime_root=root,
    )
    before = runtime.monitor_telemetry(adapter, normal_row)
    trip = runtime.monitor_telemetry(
        adapter,
        normal_row,
        fixture_overrides={"temperature_c": 250.0},
    )
    replay = runtime.monitor_telemetry(
        adapter,
        normal_row,
        fixture_overrides={"temperature_c": 250.0},
    )

    auto_resume_code = None
    try:
        runtime.automatic_resume(adapter)
    except SafetyAcknowledgementRequiredError as exc:
        auto_resume_code = exc.code

    runtime.acknowledge(
        operator_id="operator_001",
        note="已检查模拟设备，准备复核传感器值",
    )
    unsafe_recheck = runtime.recheck_telemetry(
        normal_row,
        fixture_overrides={"temperature_c": 250.0},
    )
    paused_row = recorder.capture(adapter)["record"]
    resumed = runtime.resume_after_ack(
        adapter,
        paused_row,
        operator_id="operator_001",
    )

    print("RUNTIME OVERTEMPERATURE")
    print(f"normal_observation_safe: {str(before['safe']).lower()}")
    print(f"trip_state: {trip['state']}")
    print(f"trip_code: {trip['trip']['code']}")
    print(f"device_after_trip: {trip['trip']['stop_action']} -> {paused_row['device_status']}")
    print(f"trip_replay_idempotent: {str(replay['idempotent_replay']).lower()}")
    print(f"automatic_resume_blocked: {auto_resume_code}")
    print(f"unsafe_recheck_safe: {str(unsafe_recheck['safe']).lower()}")
    print(f"operator_acknowledged: true")
    print(f"safe_recheck_resume: {str(resumed['resumed']).lower()}")
    print(f"device_after_resume: {resumed['device_state']}")
    print(f"interlock_after_resume: {runtime.snapshot()['state']}")
    print()

    # 3. Runtime overpressure.
    pressure_adapter = _running_adapter("SIM_T31_PRESSURE", protocol)
    _, pressure_row = _processing_row(root, "V030_T31_PRESSURE", pressure_adapter, protocol)
    pressure_interlock = SafetyInterlock(
        interlock_id="V030_T31_PRESSURE",
        policy=safety_policy(pressure_adapter.device_id),
        runtime_root=root,
    )
    pressure_trip = pressure_interlock.monitor_telemetry(
        pressure_adapter,
        pressure_row,
        fixture_overrides={"pressure_mpa": 13.0},
    )
    print("RUNTIME OVERPRESSURE")
    print(f"trip_state: {pressure_trip['state']}")
    print(f"trip_field: {pressure_trip['violations'][0]['field']}")
    print(f"observed_pressure_mpa: {pressure_trip['violations'][0]['value']:.6f}")
    print(f"policy_max_mpa: {pressure_trip['violations'][0]['limit']:.6f}")
    print()

    # 4. Alarm is nonrecoverable for the same job.
    alarm_adapter = _running_adapter("SIM_T31_ALARM", protocol)
    _, alarm_row = _processing_row(root, "V030_T31_ALARM", alarm_adapter, protocol)
    alarm_interlock = SafetyInterlock(
        interlock_id="V030_T31_ALARM",
        policy=safety_policy(alarm_adapter.device_id),
        runtime_root=root,
    )
    alarm_trip = alarm_interlock.monitor_telemetry(
        alarm_adapter,
        alarm_row,
        fixture_overrides={"alarm_code": "SENSOR_FAULT"},
    )
    print("DEVICE ALARM")
    print(f"alarm_trip_code: {alarm_trip['trip']['code']}")
    print(f"alarm_device_state: {alarm_adapter.status()['state']}")
    print(f"same_job_recoverable: {str(alarm_trip['trip']['recoverable_same_job']).lower()}")
    print()

    # 5. Communication loss.
    comm_adapter = _running_adapter("SIM_T31_COMM", protocol)
    comm_interlock = SafetyInterlock(
        interlock_id="V030_T31_COMM",
        policy=safety_policy(comm_adapter.device_id),
        runtime_root=root,
    )
    comm_trip = comm_interlock.monitor_health(
        comm_adapter,
        communication_ok=False,
    )
    print("COMMUNICATION LOSS")
    print(f"comm_trip_code: {comm_trip['trip']['code']}")
    print(f"comm_state: {comm_trip['state']}")
    print(f"comm_device_state: {comm_adapter.status()['state']}")
    print()

    # 6. Manual e-stop must not resume the same job.
    estop_adapter = _running_adapter("SIM_T31_ESTOP", protocol)
    estop_interlock = SafetyInterlock(
        interlock_id="V030_T31_ESTOP",
        policy=safety_policy(estop_adapter.device_id),
        runtime_root=root,
    )
    estop = estop_interlock.emergency_stop(
        estop_adapter,
        operator_id="operator_999",
        note="fixture emergency stop",
    )
    estop_interlock.acknowledge(
        operator_id="operator_999",
        note="现场确认",
    )
    restart_code = None
    dummy_row = deepcopy(normal_row)
    dummy_row["device_id"] = estop_adapter.device_id
    try:
        estop_interlock.resume_after_ack(
            estop_adapter,
            dummy_row,
            operator_id="operator_999",
        )
    except SafetyRestartRequiredError as exc:
        restart_code = exc.code
    cleared = estop_interlock.clear_terminal_stop(
        estop_adapter,
        operator_id="operator_999",
    )
    print("MANUAL EMERGENCY STOP")
    print(f"estop_state: {estop['state']}")
    print(f"estop_device_state: {estop_adapter.status()['state']}")
    print(f"same_job_resume_blocked: {restart_code}")
    print(f"new_job_required: {str(cleared['new_job_required']).lower()}")
    print()

    # 7. Device execution error.
    fault_protocol = normal_protocol()
    fault_adapter = SimulatorDeviceAdapter(device_profile("SIM_T31_FAULT", fault=True))
    fault_adapter.connect()
    fault_adapter.prepare(fault_protocol)
    fault_adapter.submit_protocol(fault_protocol)
    fault_adapter.start()
    fault_recorder = TelemetryRecorder(
        session_id="V030_T31_DEVICE_ERROR",
        scheduler_job_id="sched_v030_t31_device_error",
        experiment_id=fault_protocol["candidate_id"],
        device_id=fault_adapter.device_id,
        protocol=fault_protocol,
        runtime_root=root,
    )
    fault_recorder.capture(fault_adapter)
    fault_code = None
    try:
        fault_adapter.tick()
    except DeviceAdapterError as exc:
        fault_code = exc.code
        fault_row = fault_recorder.capture(
            fault_adapter,
            alarm_code=exc.code,
        )["record"]
    fault_interlock = SafetyInterlock(
        interlock_id="V030_T31_DEVICE_ERROR",
        policy=safety_policy(fault_adapter.device_id),
        runtime_root=root,
    )
    fault_trip = fault_interlock.monitor_telemetry(fault_adapter, fault_row)
    print("DEVICE ERROR")
    print(f"device_error_code: {fault_code}")
    print(f"safety_trip_code: {fault_trip['trip']['code']}")
    print(f"safety_state: {fault_trip['state']}")
    print(f"device_state: {fault_adapter.status()['state']}")
    print()

    snap = runtime.snapshot()
    print("AUDIT")
    print(f"event_count: {snap['event_count']}")
    print(f"hash_chain_valid: {str(snap['hash_chain_valid']).lower()}")
    print(f"atomic_state_write: {str(snap['atomic_state_write']).lower()}")
    print(f"automatic_resume_allowed: {str(snap['automatic_resume_allowed']).lower()}")
    print(f"safety_json: {snap['state_json']}")
    print(f"events_jsonl: {snap['events_jsonl']}")
    print()
    print("EXECUTION BOUNDARY")
    print("T31 uses deterministic simulator safety observations only.")
    print("A SAFETY_STOP is latched and cannot auto-resume.")
    print("Recoverable trips require operator acknowledgement + fresh safe recheck.")
    print("Nonrecoverable trips require a new job after operator clearance.")
    print("T31 does not connect to or control a real device.")

    if not safe_preflight["allowed"]:
        raise SystemExit("ERROR: normal protocol should pass safety preflight")
    if blocked_preflight["state"] != "SAFETY_STOP":
        raise SystemExit("ERROR: overlimit protocol not blocked")
    if trip["trip"]["code"] != "RUNTIME_LIMIT":
        raise SystemExit("ERROR: overtemperature did not trip RUNTIME_LIMIT")
    if paused_row["device_status"] != "PAUSED":
        raise SystemExit("ERROR: recoverable safety trip must pause device")
    if auto_resume_code != "OPERATOR_ACK_REQUIRED":
        raise SystemExit("ERROR: automatic resume was not blocked")
    if unsafe_recheck["safe"]:
        raise SystemExit("ERROR: unsafe recheck unexpectedly passed")
    if resumed["device_state"] != "RUNNING" or runtime.snapshot()["state"] != "SAFE":
        raise SystemExit("ERROR: safe operator resume failed")
    if pressure_trip["state"] != "SAFETY_STOP":
        raise SystemExit("ERROR: overpressure not stopped")
    if alarm_trip["trip"]["code"] != "DEVICE_ALARM":
        raise SystemExit("ERROR: alarm did not trip")
    if comm_trip["trip"]["code"] != "COMMUNICATION_LOSS":
        raise SystemExit("ERROR: communication loss did not trip")
    if estop_adapter.status()["state"] != "CANCELLED":
        raise SystemExit("ERROR: e-stop did not cancel device")
    if restart_code != "RESTART_REQUIRED":
        raise SystemExit("ERROR: e-stop same-job resume was not blocked")
    if fault_trip["trip"]["code"] != "DEVICE_ERROR":
        raise SystemExit("ERROR: device error did not latch safety stop")
    if not snap["hash_chain_valid"]:
        raise SystemExit("ERROR: safety audit hash chain invalid")

    print()
    print("V0.3-T31 SAFETY INTERLOCK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
