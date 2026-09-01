from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.device import (
    DeviceBusyError,
    DeviceExecutionError,
    DeviceOfflineError,
    DeviceStateError,
    DeviceUnsupportedProtocolError,
    SimulatorDeviceAdapter,
)


def code_of(fn) -> str:
    try:
        fn()
    except Exception as exc:  # runner deliberately demonstrates typed failures
        return getattr(exc, "code", type(exc).__name__)
    raise AssertionError("expected failure did not occur")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    args = parser.parse_args()

    fixture = Path(args.runtime_root) / "v030" / "fixtures" / "t28"
    profiles_path = fixture / "device_profiles.json"
    protocols_path = fixture / "protocols.json"
    if not profiles_path.exists() or not protocols_path.exists():
        raise SystemExit(
            "ERROR: 请先运行 python -m scripts.build_v030_t28_fixture --reset"
        )

    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    protocols = json.loads(protocols_path.read_text(encoding="utf-8"))
    ready = protocols["ready"]
    second_ready = protocols["second_ready"]
    blocked = protocols["blocked"]

    print("V0.3-T28 DEVICE ADAPTER")
    print()
    print("BOUNDARY")
    print("adapter_type: simulator")
    print("real_device_connected: false")
    print("only_T27_READY_protocol_allowed: true")

    device = SimulatorDeviceAdapter(profiles["main"])
    print()
    print("HAPPY PATH")
    print(f"initial_state: {device.status()['state']}")
    print(f"connect_state: {device.connect()['state']}")
    health = device.health_check()
    print(f"health: {'HEALTHY' if health['healthy'] else 'UNHEALTHY'}")
    print(f"prepare_state: {device.prepare(ready)['state']}")
    submitted = device.submit_protocol(ready)
    print(f"submit_state: {submitted['state']}")
    print(f"job_id: {submitted['job']['job_id']}")
    replay = device.submit_protocol(ready)
    print(f"submit_replay_idempotent: {str(replay['idempotent_replay']).lower()}")
    print(f"start_state: {device.start()['state']}")
    print(f"pause_state: {device.pause()['state']}")
    print(f"resume_state: {device.resume()['state']}")
    device.run_to_completion()
    print(f"completed_state: {device.status()['state']}")
    result = device.read_result()
    first_output = result["outputs"][0]
    print(f"result_id: {result['result_id']}")
    print(f"result_origin: {result['measurement_origin']}")
    print(f"is_real_measurement: {str(result['is_real_measurement']).lower()}")
    print(
        "result: "
        f"{first_output['metric']}={first_output['value']:.6f} {first_output['unit']}"
    )
    print(f"disconnect_state: {device.disconnect()['state']}")

    print()
    print("BOUNDARY REJECTIONS")
    blocked_device = SimulatorDeviceAdapter(profiles["main"])
    blocked_device.connect()
    blocked_code = code_of(lambda: blocked_device.prepare(blocked))
    print(f"blocked_protocol: {blocked_code}")

    offline = SimulatorDeviceAdapter(profiles["offline"])
    offline_code = code_of(offline.connect)
    print(f"offline_device: {offline_code}")

    limited = SimulatorDeviceAdapter(profiles["limited"])
    limited.connect()
    unsupported_code = code_of(lambda: limited.prepare(ready))
    print(f"unsupported_protocol: {unsupported_code}")

    busy = SimulatorDeviceAdapter(profiles["main"])
    busy.connect()
    busy.prepare(ready)
    busy.submit_protocol(ready)
    busy_code = code_of(lambda: busy.prepare(second_ready))
    print(f"busy_device: {busy_code}")

    invalid = SimulatorDeviceAdapter(profiles["main"])
    invalid.connect()
    invalid_code = code_of(invalid.start)
    print(f"invalid_state: {invalid_code}")

    fault = SimulatorDeviceAdapter(profiles["fault"])
    fault.connect()
    fault.prepare(ready)
    fault.submit_protocol(ready)
    fault_code = code_of(fault.start)
    print(f"device_error: {fault_code}")
    print(f"fault_state: {fault.status()['state']}")

    cancel_device = SimulatorDeviceAdapter(profiles["main"])
    cancel_device.connect()
    cancel_device.prepare(ready)
    cancel_device.submit_protocol(ready)
    cancel_device.start()
    print(f"cancel_state: {cancel_device.cancel()['state']}")

    assert device.state == "DISCONNECTED"
    assert health["healthy"] is True
    assert replay["idempotent_replay"] is True
    assert result["synthetic"] is True
    assert result["is_real_measurement"] is False
    assert blocked_code == "UNSUPPORTED_PROTOCOL"
    assert offline_code == "OFFLINE"
    assert unsupported_code == "UNSUPPORTED_PROTOCOL"
    assert busy_code == "BUSY"
    assert invalid_code == "INVALID_STATE"
    assert fault_code == "DEVICE_ERROR"
    assert fault.state == "ERROR"
    assert cancel_device.state == "CANCELLED"

    print()
    print("EXECUTION BOUNDARY")
    print("T28 validates the adapter contract with a deterministic simulator only.")
    print("Synthetic simulator output must never be treated as a real measurement.")
    print("Real device drivers are not connected in T28.")
    print()
    print("V0.3-T28 DEVICE ADAPTER PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
