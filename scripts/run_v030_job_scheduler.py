from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from experiments.device import SimulatorDeviceAdapter
from experiments.scheduler import JobScheduler
from scripts.build_v030_t29_fixture import device_profiles, protocol_documents


def _scheduler(root: Path, scheduler_id: str, profile_keys: list[str]) -> JobScheduler:
    profiles = device_profiles()
    devices = {
        profiles[key]["device_id"]: SimulatorDeviceAdapter(profiles[key])
        for key in profile_keys
    }
    return JobScheduler(
        scheduler_id=scheduler_id,
        devices=devices,
        runtime_root=root,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(args.runtime_root)
    if args.reset:
        target = root / "v030" / "scheduler"
        if target.exists():
            shutil.rmtree(target)

    protocols = protocol_documents()

    print("V0.3-T29 JOB SCHEDULER")
    print()
    print("BOUNDARY")
    print("device_adapter: SimulatorDeviceAdapter only")
    print("real_device_connected: false")
    print("only_T27_READY_protocol_allowed: true")

    # Priority + FIFO on one finite device.
    scheduler = _scheduler(root, "V030_T29_PRIORITY", ["fast"])
    priorities = [10, 30, 30, 5, 20]
    queued = []
    for protocol, priority in zip(protocols[:5], priorities):
        queued.append(
            scheduler.submit(protocol, priority=priority, timeout_ticks=20)["job"]
        )
    replay = scheduler.submit(protocols[0], priority=10, timeout_ticks=20)
    expected_order = [
        protocols[1]["candidate_id"],
        protocols[2]["candidate_id"],
        protocols[4]["candidate_id"],
        protocols[0]["candidate_id"],
        protocols[3]["candidate_id"],
    ]
    actual_dispatch_order = []
    while scheduler.snapshot()["counts"]["QUEUED"]:
        dispatched = scheduler.dispatch_once(max_jobs=1)
        if not dispatched:
            raise SystemExit("ERROR: priority scheduler could not dispatch")
        actual_dispatch_order.append(dispatched[0]["candidate_id"])
        scheduler.start_dispatched()
        while scheduler.snapshot()["counts"]["RUNNING"]:
            scheduler.advance_running(ticks=1)

    print()
    print("PRIORITY + FIFO")
    print(f"submitted_jobs: {len(queued)}")
    print(f"priority_sequence: {priorities}")
    print(f"expected_dispatch_order: {expected_order}")
    print(f"actual_dispatch_order: {actual_dispatch_order}")
    print(f"priority_fifo_correct: {str(actual_dispatch_order == expected_order).lower()}")
    print(f"duplicate_submit_idempotent: {str(replay['idempotent_replay']).lower()}")
    print(f"job_count_after_replay: {scheduler.snapshot()['job_count']}")

    # Finite resource / device conflict.
    conflict = _scheduler(root, "V030_T29_CONFLICT", ["fast"])
    conflict.submit(protocols[0], priority=0)
    conflict.submit(protocols[1], priority=0)
    first_dispatch = conflict.dispatch_once()
    conflict_counts = conflict.snapshot()["counts"]
    print()
    print("FINITE DEVICE CAPACITY")
    print(f"queued_jobs: 2")
    print(f"dispatched_in_first_pass: {len(first_dispatch)}")
    print(f"remaining_queued: {conflict_counts['QUEUED']}")
    print("same_device_double_booked: false")

    # Cancel a queued job.
    cancel_sched = _scheduler(root, "V030_T29_CANCEL", ["fast"])
    cancel_job = cancel_sched.submit(protocols[2])["job"]
    cancelled = cancel_sched.cancel_job(cancel_job["scheduler_job_id"])
    cancel_replay = cancel_sched.cancel_job(cancel_job["scheduler_job_id"])
    print()
    print("CANCEL")
    print(f"queued_cancel_state: {cancelled['job']['status']}")
    print(f"cancel_replay_idempotent: {str(cancel_replay['idempotent_replay']).lower()}")

    # Timeout on a slow simulator.
    timeout_sched = _scheduler(root, "V030_T29_TIMEOUT", ["slow"])
    timeout_job = timeout_sched.submit(
        protocols[3], timeout_ticks=2, max_retries=0
    )["job"]
    timeout_sched.dispatch_once()
    timeout_sched.start_dispatched()
    timeout_sched.advance_running(ticks=2)
    timeout_state = next(
        j for j in timeout_sched.snapshot()["jobs"]
        if j["scheduler_job_id"] == timeout_job["scheduler_job_id"]
    )
    print()
    print("TIMEOUT")
    print(f"timeout_ticks: {timeout_state['timeout_ticks']}")
    print(f"elapsed_ticks: {timeout_state['elapsed_ticks']}")
    print(f"timeout_state: {timeout_state['status']}")

    # Retry after a single injected device execution fault.
    retry_sched = _scheduler(root, "V030_T29_RETRY", ["flaky"])
    retry_job = retry_sched.submit(
        protocols[4], timeout_ticks=20, max_retries=1
    )["job"]
    retry_sched.dispatch_once()
    retry_sched.start_dispatched()
    retry_sched.advance_running(ticks=1)
    after_failure = next(
        j for j in retry_sched.snapshot()["jobs"]
        if j["scheduler_job_id"] == retry_job["scheduler_job_id"]
    )
    flaky = next(iter(retry_sched.devices.values()))
    flaky.profile["fault_injection"]["tick"] = False
    retry_sched.run_until_terminal(max_scheduler_ticks=20)
    after_retry = next(
        j for j in retry_sched.snapshot()["jobs"]
        if j["scheduler_job_id"] == retry_job["scheduler_job_id"]
    )
    print()
    print("FAILURE RETRY")
    print(f"after_first_failure: {after_failure['status']}")
    print(f"attempts_after_failure: {after_failure['attempts_started']}")
    print(f"final_state: {after_retry['status']}")
    print(f"attempts_total: {after_retry['attempts_started']}")
    print(f"result_origin: {(after_retry.get('result') or {}).get('measurement_origin')}")
    print(f"is_real_measurement: {str(bool((after_retry.get('result') or {}).get('is_real_measurement'))).lower()}")

    # Two-device throughput: at most one job per device, two can dispatch together.
    multi = _scheduler(root, "V030_T29_MULTI", ["fast", "aux"])
    for p in protocols[:5]:
        multi.submit(p, priority=0, timeout_ticks=20)
    first_multi = multi.dispatch_once()
    multi.start_dispatched()
    multi.run_until_terminal(max_scheduler_ticks=100)
    multi_snapshot = multi.snapshot()
    print()
    print("MULTI-DEVICE")
    print(f"registered_devices: {len(multi.devices)}")
    print(f"first_pass_dispatched: {len(first_multi)}")
    print(f"completed_jobs: {multi_snapshot['counts']['COMPLETED']}")
    print(f"failed_jobs: {multi_snapshot['counts']['FAILED']}")
    print(f"timeout_jobs: {multi_snapshot['counts']['TIMEOUT']}")

    if actual_dispatch_order != expected_order:
        raise SystemExit("ERROR: priority/FIFO order mismatch")
    if not replay["idempotent_replay"] or scheduler.snapshot()["job_count"] != 5:
        raise SystemExit("ERROR: duplicate submit idempotency failed")
    if len(first_dispatch) != 1 or conflict_counts["QUEUED"] != 1:
        raise SystemExit("ERROR: finite device capacity failed")
    if cancelled["job"]["status"] != "CANCELLED" or not cancel_replay["idempotent_replay"]:
        raise SystemExit("ERROR: cancel behavior failed")
    if timeout_state["status"] != "TIMEOUT":
        raise SystemExit("ERROR: timeout behavior failed")
    if after_failure["status"] != "QUEUED" or after_retry["status"] != "COMPLETED":
        raise SystemExit("ERROR: retry behavior failed")
    if after_retry["attempts_started"] != 2:
        raise SystemExit("ERROR: retry attempts expected 2")
    if (after_retry.get("result") or {}).get("is_real_measurement") is not False:
        raise SystemExit("ERROR: simulator result boundary failed")
    if len(first_multi) != 2 or multi_snapshot["counts"]["COMPLETED"] != 5:
        raise SystemExit("ERROR: multi-device scheduling failed")

    print()
    print("PERSISTENCE")
    print(f"scheduler_json: {multi_snapshot['state_path']}")
    print("atomic_state_write: true")
    print()
    print("EXECUTION BOUNDARY")
    print("T29 schedules deterministic simulator jobs only.")
    print("T30 will own telemetry/state streaming; T35 will own active-job crash reconciliation.")
    print("Synthetic scheduler/device output must never be treated as a real measurement.")
    print()
    print("V0.3-T29 JOB SCHEDULER PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
