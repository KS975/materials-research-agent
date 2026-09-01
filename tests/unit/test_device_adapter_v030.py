from copy import deepcopy

import pytest

from experiments.device import (
    DeviceBusyError,
    DeviceExecutionError,
    DeviceOfflineError,
    DeviceStateError,
    DeviceUnsupportedProtocolError,
    SimulatorDeviceAdapter,
    deterministic_job_id,
    protocol_device_roles,
    validate_device_profile,
)
from experiments.protocol import ExperimentProtocolBuilder
from scripts.build_v030_t27_fixture import candidates, fixture_template
from scripts.build_v030_t28_fixture import device_profiles


def docs():
    builder = ExperimentProtocolBuilder(fixture_template())
    c = candidates()
    return (
        builder.build(c["valid"]),
        builder.build(c["optional_missing"]),
        builder.build(c["unsafe"]),
    )


def test_happy_path_connect_prepare_submit_run_and_read_result():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    assert device.status()["state"] == "DISCONNECTED"
    assert device.connect()["state"] == "IDLE"
    assert device.health_check()["healthy"] is True
    assert device.prepare(ready)["state"] == "PREPARED"
    submitted = device.submit_protocol(ready)
    assert submitted["state"] == "SUBMITTED"
    assert device.start()["state"] == "RUNNING"
    device.run_to_completion()
    assert device.status()["state"] == "COMPLETED"
    result = device.read_result()
    assert result["synthetic"] is True
    assert result["is_real_measurement"] is False
    assert result["measurement_origin"] == "SIMULATOR_FIXTURE"
    assert result["outputs"][0]["condition_signature"] == "ISO180_23C_NOTCHED"
    assert device.disconnect()["state"] == "DISCONNECTED"


def test_submission_is_idempotent_and_job_id_is_deterministic():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    device.prepare(ready)
    first = device.submit_protocol(ready)
    second = device.submit_protocol(ready)
    expected = deterministic_job_id(device.device_id, ready["protocol_id"])
    assert first["job"]["job_id"] == expected
    assert second["job"]["job_id"] == expected
    assert second["idempotent_replay"] is True


def test_blocked_t27_protocol_is_rejected_at_device_boundary():
    _, _, blocked = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    with pytest.raises(DeviceUnsupportedProtocolError) as exc:
        device.prepare(blocked)
    assert exc.value.code == "UNSUPPORTED_PROTOCOL"


def test_tampered_protocol_is_rejected():
    ready, _, _ = docs()
    tampered = deepcopy(ready)
    tampered["process_parameters"][0]["value"] += 1
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    with pytest.raises(DeviceUnsupportedProtocolError):
        device.prepare(tampered)


def test_offline_device_cannot_connect():
    device = SimulatorDeviceAdapter(device_profiles()["offline"])
    with pytest.raises(DeviceOfflineError) as exc:
        device.connect()
    assert exc.value.code == "OFFLINE"
    assert device.status()["state"] == "DISCONNECTED"


def test_missing_required_role_is_unsupported_protocol():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["limited"])
    device.connect()
    assert set(protocol_device_roles(ready)) == {
        "material_dispenser", "compounder", "impact_tester"
    }
    with pytest.raises(DeviceUnsupportedProtocolError) as exc:
        device.prepare(ready)
    assert "impact_tester" in exc.value.details["missing_roles"]


def test_busy_device_rejects_different_protocol():
    ready, second, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    device.prepare(ready)
    device.submit_protocol(ready)
    with pytest.raises(DeviceBusyError) as exc:
        device.prepare(second)
    assert exc.value.code == "BUSY"


def test_invalid_state_operations_are_rejected():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    with pytest.raises(DeviceStateError):
        device.start()
    device.prepare(ready)
    with pytest.raises(DeviceStateError):
        device.pause()
    device.submit_protocol(ready)
    with pytest.raises(DeviceStateError):
        device.read_result()


def test_pause_resume_and_cancel_are_explicit_state_transitions():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    device.prepare(ready)
    device.submit_protocol(ready)
    device.start()
    assert device.pause()["state"] == "PAUSED"
    assert device.resume()["state"] == "RUNNING"
    assert device.cancel()["state"] == "CANCELLED"
    assert device.disconnect()["state"] == "DISCONNECTED"


def test_active_job_prevents_disconnect():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["main"])
    device.connect()
    device.prepare(ready)
    device.submit_protocol(ready)
    with pytest.raises(DeviceBusyError):
        device.disconnect()


def test_fault_injection_surfaces_device_error_and_error_state():
    ready, _, _ = docs()
    device = SimulatorDeviceAdapter(device_profiles()["fault"])
    device.connect()
    device.prepare(ready)
    device.submit_protocol(ready)
    with pytest.raises(DeviceExecutionError) as exc:
        device.start()
    assert exc.value.code == "DEVICE_ERROR"
    assert device.status()["state"] == "ERROR"
    # ERROR may be disconnected explicitly so a later controller can reconcile.
    assert device.disconnect()["state"] == "DISCONNECTED"


def test_result_is_deterministic_but_explicitly_non_real():
    ready, _, _ = docs()
    results = []
    for _ in range(2):
        device = SimulatorDeviceAdapter(device_profiles()["main"])
        device.connect()
        device.prepare(ready)
        device.submit_protocol(ready)
        device.start()
        device.run_to_completion()
        results.append(device.read_result())
    assert results[0] == results[1]
    assert results[0]["is_real_measurement"] is False


def test_profile_rejects_real_adapter_type_in_t28():
    profile = deepcopy(device_profiles()["main"])
    profile["adapter_type"] = "serial_real_device"
    with pytest.raises(Exception, match="simulator"):
        validate_device_profile(profile)
