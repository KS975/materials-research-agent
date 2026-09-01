from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.device import (
    DeviceAdapterError,
    SimulatorDeviceAdapter,
)
from experiments.scheduler import JobScheduler
from experiments.telemetry import (
    ExperimentStateMachine,
    SIMULATOR_TIME_SOURCE,
    TelemetryIntegrityError,
    TelemetryRecorder,
    TelemetryStateError,
    TelemetryValidationError,
)
from scripts.build_v030_t30_fixture import (
    device_profiles,
    protocol_document,
)


def _running(tmp_path):
    protocol = protocol_document()
    profile = device_profiles()["normal"]
    adapter = SimulatorDeviceAdapter(profile)
    scheduler = JobScheduler(
        scheduler_id="TEST_T30",
        devices={adapter.device_id: adapter},
        runtime_root=tmp_path,
    )
    job = scheduler.submit(protocol, timeout_ticks=30)["job"]
    scheduler.dispatch_once()
    scheduler.start_dispatched()
    recorder = TelemetryRecorder(
        session_id="TEST_T30_SESSION",
        scheduler_job_id=job["scheduler_job_id"],
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    return protocol, adapter, scheduler, job, recorder


def test_full_phase_sequence_with_scheduler(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    recorder.capture(adapter)
    while scheduler.snapshot()["counts"]["RUNNING"]:
        scheduler.advance_running(ticks=1)
        recorder.capture(adapter)
    phases = recorder.snapshot()["phase_history"]
    assert phases == [
        "PREPARING",
        "MATERIAL_LOADING",
        "HEATING",
        "PROCESSING",
        "COOLING",
        "MEASURING",
        "COMPLETED",
    ]


def test_traceability_fields_present(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    row = recorder.capture(adapter)["record"]
    assert row["device_id"] == adapter.device_id
    assert row["scheduler_job_id"] == job["scheduler_job_id"]
    assert row["experiment_id"] == protocol["candidate_id"]
    assert row["protocol_id"] == protocol["protocol_id"]
    assert row["timestamp"]
    assert row["time_source"] == SIMULATOR_TIME_SOURCE


def test_processing_sensors_follow_protocol(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    recorder.capture(adapter)
    found = None
    for _ in range(10):
        scheduler.advance_running(ticks=1)
        row = recorder.capture(adapter)["record"]
        if row["phase"] == "PROCESSING" and found is None:
            found = row
    assert found is not None
    assert found["temperature_c"] == 230.0
    assert found["pressure_mpa"] == 8.5
    assert found["rpm"] == 320.0


def test_duplicate_capture_is_idempotent(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    first = recorder.capture(adapter)
    second = recorder.capture(adapter)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert recorder.snapshot()["telemetry_count"] == 1


def test_pause_resume_returns_to_prior_phase(tmp_path):
    protocol = protocol_document()
    adapter = SimulatorDeviceAdapter(device_profiles()["normal"])
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    recorder = TelemetryRecorder(
        session_id="PAUSE",
        scheduler_job_id="sched_pause",
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    recorder.capture(adapter)
    for _ in range(4):
        adapter.tick()
        recorder.capture(adapter)
    prior = recorder.snapshot()["phase"]
    assert prior == "PROCESSING"
    adapter.pause()
    paused = recorder.capture(adapter)["record"]
    assert paused["phase"] == "PAUSED"
    assert paused["rpm"] == 0.0
    adapter.resume()
    resumed = recorder.capture(adapter)["record"]
    assert resumed["phase"] == prior


def test_fault_becomes_error_with_alarm(tmp_path):
    protocol = protocol_document()
    adapter = SimulatorDeviceAdapter(device_profiles()["fault"])
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    recorder = TelemetryRecorder(
        session_id="FAULT",
        scheduler_job_id="sched_fault",
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    recorder.capture(adapter)
    with pytest.raises(DeviceAdapterError) as caught:
        adapter.tick()
    row = recorder.capture(
        adapter,
        alarm_code=caught.value.code,
    )["record"]
    assert row["phase"] == "ERROR"
    assert row["alarm_code"] == "DEVICE_ERROR"


def test_cancel_becomes_cancelled(tmp_path):
    protocol = protocol_document()
    adapter = SimulatorDeviceAdapter(device_profiles()["normal"])
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    recorder = TelemetryRecorder(
        session_id="CANCEL",
        scheduler_job_id="sched_cancel",
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    recorder.capture(adapter)
    adapter.cancel()
    row = recorder.capture(adapter)["record"]
    assert row["phase"] == "CANCELLED"


def test_hash_chain_and_persistence_reload(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    recorder.capture(adapter)
    scheduler.advance_running(ticks=1)
    recorder.capture(adapter)
    assert recorder.verify_integrity() is True

    reloaded = TelemetryRecorder(
        session_id="TEST_T30_SESSION",
        scheduler_job_id=job["scheduler_job_id"],
        experiment_id=protocol["candidate_id"],
        device_id=adapter.device_id,
        protocol=protocol,
        runtime_root=tmp_path,
    )
    assert reloaded.snapshot()["telemetry_count"] == 2
    assert reloaded.verify_integrity() is True


def test_tamper_detected(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    recorder.capture(adapter)
    path = Path(recorder.snapshot()["session_json"])
    state = json.loads(path.read_text(encoding="utf-8"))
    state["telemetry"][0]["progress_percent"] = 99.0
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(TelemetryIntegrityError):
        TelemetryRecorder(
            session_id="TEST_T30_SESSION",
            scheduler_job_id=job["scheduler_job_id"],
            experiment_id=protocol["candidate_id"],
            device_id=adapter.device_id,
            protocol=protocol,
            runtime_root=tmp_path,
        )


def test_wrong_device_rejected(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    other_profile = device_profiles()["normal"].copy()
    other_profile["device_id"] = "SIM_T30_OTHER"
    other = SimulatorDeviceAdapter(other_profile)
    with pytest.raises(TelemetryValidationError):
        recorder.capture(other)


def test_state_machine_rejects_phase_skip():
    machine = ExperimentStateMachine()
    machine.transition("PREPARING")
    with pytest.raises(TelemetryStateError):
        machine.transition("PROCESSING")


def test_terminal_phase_cannot_restart():
    machine = ExperimentStateMachine()
    for phase in (
        "PREPARING", "MATERIAL_LOADING", "HEATING",
        "PROCESSING", "COOLING", "MEASURING", "COMPLETED",
    ):
        machine.transition(phase)
    with pytest.raises(TelemetryStateError):
        machine.transition("PREPARING")


def test_simulator_boundary_is_explicit(tmp_path):
    protocol, adapter, scheduler, job, recorder = _running(tmp_path)
    row = recorder.capture(adapter)["record"]
    snap = recorder.snapshot()
    assert row["measurement_origin"] == "SIMULATOR_FIXTURE"
    assert row["synthetic"] is True
    assert row["is_real_telemetry"] is False
    assert snap["real_device_connected"] is False
    assert snap["time_source"] == SIMULATOR_TIME_SOURCE
