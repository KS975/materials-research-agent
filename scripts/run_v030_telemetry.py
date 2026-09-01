from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from experiments.device import (
    DeviceAdapterError,
    SimulatorDeviceAdapter,
)
from experiments.scheduler import JobScheduler
from experiments.telemetry import TelemetryRecorder
from scripts.build_v030_t30_fixture import (
    device_profiles,
    protocol_document,
)


def _find_job(snapshot: dict, scheduler_job_id: str) -> dict:
    return next(
        j for j in snapshot["jobs"]
        if j["scheduler_job_id"] == scheduler_job_id
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(args.runtime_root)

    if args.reset:
        for relative in (
            Path("v030") / "telemetry",
            Path("v030") / "scheduler" / "V030_T30_MAIN",
        ):
            target = root / relative
            if target.exists():
                shutil.rmtree(target)

    protocol = protocol_document()
    profiles = device_profiles()
    adapter = SimulatorDeviceAdapter(profiles["normal"])
    scheduler = JobScheduler(
        scheduler_id="V030_T30_MAIN",
        devices={adapter.device_id: adapter},
        runtime_root=root,
    )
    submitted = scheduler.submit(
        protocol,
        priority=10,
        timeout_ticks=30,
    )["job"]
    scheduler.dispatch_once()
    scheduler.start_dispatched()
    running = _find_job(
        scheduler.snapshot(),
        submitted["scheduler_job_id"],
    )

    recorder = TelemetryRecorder(
        session_id="V030_T30_MAIN_SESSION",
        scheduler_job_id=submitted["scheduler_job_id"],
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=root,
    )

    first = recorder.capture(adapter)
    replay = recorder.capture(adapter)

    while scheduler.snapshot()["counts"]["RUNNING"]:
        scheduler.advance_running(ticks=1)
        recorder.capture(adapter)

    snap = recorder.snapshot()
    job_final = _find_job(
        scheduler.snapshot(),
        submitted["scheduler_job_id"],
    )

    unique_phases = []
    for p in snap["phase_history"]:
        if not unique_phases or unique_phases[-1] != p:
            unique_phases.append(p)

    processing = next(
        x for x in snap["telemetry"]
        if x["phase"] == "PROCESSING"
    )

    # Pause/resume is exercised directly at the adapter boundary. Scheduler-level
    # operator orchestration remains T35.
    pause_adapter = SimulatorDeviceAdapter(profiles["normal"])
    pause_adapter.connect()
    pause_adapter.prepare(protocol)
    pause_adapter.submit_protocol(protocol)
    pause_adapter.start()
    pause_recorder = TelemetryRecorder(
        session_id="V030_T30_PAUSE_SESSION",
        scheduler_job_id="schedjob_t30_pause_demo",
        experiment_id=protocol["candidate_id"],
        device_id=pause_adapter.device_id,
        protocol=protocol,
        runtime_root=root,
    )
    pause_recorder.capture(pause_adapter)
    for _ in range(4):
        pause_adapter.tick()
        pause_recorder.capture(pause_adapter)
    before_pause_phase = pause_recorder.snapshot()["phase"]
    pause_adapter.pause()
    paused = pause_recorder.capture(pause_adapter)["record"]
    pause_adapter.resume()
    resumed = pause_recorder.capture(pause_adapter)["record"]

    # Deterministic device fault -> ERROR telemetry + alarm code.
    fault_adapter = SimulatorDeviceAdapter(profiles["fault"])
    fault_adapter.connect()
    fault_adapter.prepare(protocol)
    fault_adapter.submit_protocol(protocol)
    fault_adapter.start()
    fault_recorder = TelemetryRecorder(
        session_id="V030_T30_FAULT_SESSION",
        scheduler_job_id="schedjob_t30_fault_demo",
        experiment_id=protocol["candidate_id"],
        device_id=fault_adapter.device_id,
        protocol=protocol,
        runtime_root=root,
    )
    fault_recorder.capture(fault_adapter)
    fault_code = None
    try:
        fault_adapter.tick()
    except DeviceAdapterError as exc:
        fault_code = exc.code
        fault_recorder.capture(
            fault_adapter,
            alarm_code=exc.code,
        )
    fault_snap = fault_recorder.snapshot()

    print("V0.3-T30 TELEMETRY + EXPERIMENT STATE MACHINE")
    print()
    print("BOUNDARY")
    print("device_adapter: SimulatorDeviceAdapter only")
    print("real_device_connected: false")
    print("time_source: SIMULATOR_VIRTUAL_CLOCK")
    print("is_real_telemetry: false")
    print()
    print("TRACEABILITY")
    print(f"experiment_id: {protocol['candidate_id']}")
    print(f"scheduler_job_id: {submitted['scheduler_job_id']}")
    print(f"adapter_job_id: {running['adapter_job_id']}")
    print(f"device_id: {adapter.device_id}")
    print()
    print("STATE MACHINE")
    print(f"phase_sequence: {unique_phases}")
    print(f"final_phase: {snap['phase']}")
    print(f"scheduler_final_state: {job_final['status']}")
    print()
    print("PROCESSING TELEMETRY")
    print(f"progress_percent: {processing['progress_percent']:.6f}")
    print(f"temperature_c: {processing['temperature_c']:.6f}")
    print(f"pressure_mpa: {processing['pressure_mpa']:.6f}")
    print(f"rpm: {processing['rpm']:.6f}")
    print(f"device_status: {processing['device_status']}")
    print(f"alarm_code: {processing['alarm_code']}")
    print()
    print("IDEMPOTENCY + AUDIT")
    print(f"duplicate_capture_idempotent: {str(replay['idempotent_replay']).lower()}")
    print(f"telemetry_count: {snap['telemetry_count']}")
    print(f"event_count: {snap['event_count']}")
    print(f"hash_chain_valid: {str(snap['hash_chain_valid']).lower()}")
    print(f"atomic_state_write: {str(snap['atomic_state_write']).lower()}")
    print(f"session_json: {snap['session_json']}")
    print(f"telemetry_jsonl: {snap['telemetry_jsonl']}")
    print(f"events_jsonl: {snap['events_jsonl']}")
    print()
    print("PAUSE / RESUME")
    print(f"before_pause_phase: {before_pause_phase}")
    print(f"paused_phase: {paused['phase']}")
    print(f"paused_rpm: {paused['rpm']:.6f}")
    print(f"resumed_phase: {resumed['phase']}")
    print()
    print("FAULT TELEMETRY")
    print(f"fault_code: {fault_code}")
    print(f"fault_phase: {fault_snap['phase']}")
    print(f"fault_alarm_code: {fault_snap['telemetry'][-1]['alarm_code']}")
    print()
    print("EXECUTION BOUNDARY")
    print("T30 records deterministic simulator telemetry and experiment phases only.")
    print("T31 will own safety interlocks; T35 will own active-device crash reconciliation.")
    print("Synthetic telemetry must never be treated as real instrument telemetry.")

    expected_phases = [
        "PREPARING",
        "MATERIAL_LOADING",
        "HEATING",
        "PROCESSING",
        "COOLING",
        "MEASURING",
        "COMPLETED",
    ]
    if unique_phases != expected_phases:
        raise SystemExit(
            f"ERROR: phase sequence mismatch: {unique_phases}"
        )
    if job_final["status"] != "COMPLETED":
        raise SystemExit("ERROR: scheduler job not completed")
    if processing["temperature_c"] != 230.0:
        raise SystemExit("ERROR: processing temperature mismatch")
    if processing["pressure_mpa"] != 8.5:
        raise SystemExit("ERROR: processing pressure mismatch")
    if processing["rpm"] != 320.0:
        raise SystemExit("ERROR: processing rpm mismatch")
    if not replay["idempotent_replay"]:
        raise SystemExit("ERROR: duplicate telemetry capture not idempotent")
    if not snap["hash_chain_valid"]:
        raise SystemExit("ERROR: telemetry hash chain invalid")
    if paused["phase"] != "PAUSED" or paused["rpm"] != 0.0:
        raise SystemExit("ERROR: pause telemetry invalid")
    if resumed["phase"] != before_pause_phase:
        raise SystemExit("ERROR: resume did not return to prior phase")
    if fault_code != "DEVICE_ERROR":
        raise SystemExit("ERROR: expected DEVICE_ERROR")
    if fault_snap["phase"] != "ERROR":
        raise SystemExit("ERROR: fault phase expected ERROR")
    if fault_snap["telemetry"][-1]["alarm_code"] != "DEVICE_ERROR":
        raise SystemExit("ERROR: fault alarm code missing")

    print()
    print("V0.3-T30 TELEMETRY + STATE MACHINE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
