from pathlib import Path

import pytest

from experiments.device import SimulatorDeviceAdapter
from experiments.scheduler import (
    JobScheduler,
    JobSchedulerConflictError,
    JobSchedulerStateError,
    JobSchedulerValidationError,
)
from scripts.build_v030_t29_fixture import device_profiles, protocol_documents


def make_scheduler(tmp_path: Path, scheduler_id="S", keys=("fast",)):
    profiles = device_profiles()
    devices = {
        profiles[k]["device_id"]: SimulatorDeviceAdapter(profiles[k])
        for k in keys
    }
    return JobScheduler(
        scheduler_id=scheduler_id,
        devices=devices,
        runtime_root=tmp_path,
    )


def test_priority_then_fifo(tmp_path):
    p = protocol_documents()
    s = make_scheduler(tmp_path)
    for proto, priority in zip(p[:5], [10, 30, 30, 5, 20]):
        s.submit(proto, priority=priority)
    order = []
    while s.snapshot()["counts"]["QUEUED"]:
        d = s.dispatch_once(max_jobs=1)
        order.append(d[0]["candidate_id"])
        s.start_dispatched()
        while s.snapshot()["counts"]["RUNNING"]:
            s.advance_running(ticks=1)
    assert order == [
        "V030_T29_EXP02", "V030_T29_EXP03", "V030_T29_EXP05",
        "V030_T29_EXP01", "V030_T29_EXP04",
    ]


def test_same_priority_is_fifo(tmp_path):
    p = protocol_documents()
    s = make_scheduler(tmp_path)
    for proto in p[:3]:
        s.submit(proto, priority=7)
    assert [s._job(x)["candidate_id"] for x in s.queued_order()] == [
        "V030_T29_EXP01", "V030_T29_EXP02", "V030_T29_EXP03"
    ]


def test_duplicate_submit_is_idempotent(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path)
    first = s.submit(p, priority=10, timeout_ticks=5, max_retries=1)
    second = s.submit(p, priority=10, timeout_ticks=5, max_retries=1)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert s.snapshot()["job_count"] == 1


def test_duplicate_submit_with_different_config_is_conflict(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path)
    s.submit(p, priority=10)
    with pytest.raises(JobSchedulerConflictError):
        s.submit(p, priority=11)


def test_only_one_job_dispatches_to_one_device(tmp_path):
    p = protocol_documents()
    s = make_scheduler(tmp_path)
    s.submit(p[0]); s.submit(p[1])
    dispatched = s.dispatch_once()
    snap = s.snapshot()
    assert len(dispatched) == 1
    assert snap["counts"]["DISPATCHED"] == 1
    assert snap["counts"]["QUEUED"] == 1


def test_two_devices_dispatch_two_jobs_without_double_booking(tmp_path):
    p = protocol_documents()
    s = make_scheduler(tmp_path, keys=("fast", "aux"))
    for proto in p[:3]: s.submit(proto)
    dispatched = s.dispatch_once()
    assert len(dispatched) == 2
    assert len({j["device_id"] for j in dispatched}) == 2


def test_device_release_allows_next_job(tmp_path):
    p = protocol_documents()
    s = make_scheduler(tmp_path)
    s.submit(p[0]); s.submit(p[1])
    s.dispatch_once(); s.start_dispatched(); s.run_until_terminal(max_scheduler_ticks=20)
    assert s.snapshot()["counts"]["COMPLETED"] == 2


def test_queued_cancel_is_terminal_and_idempotent(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path)
    job = s.submit(p)["job"]
    first = s.cancel_job(job["scheduler_job_id"])
    second = s.cancel_job(job["scheduler_job_id"])
    assert first["job"]["status"] == "CANCELLED"
    assert second["idempotent_replay"] is True


def test_running_cancel_releases_device(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path)
    job = s.submit(p)["job"]
    s.dispatch_once(); s.start_dispatched()
    cancelled = s.cancel_job(job["scheduler_job_id"])
    assert cancelled["job"]["status"] == "CANCELLED"
    assert next(iter(s.devices.values())).status()["state"] == "CANCELLED"


def test_timeout_marks_job_timeout(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path, keys=("slow",))
    job = s.submit(p, timeout_ticks=2)["job"]
    s.dispatch_once(); s.start_dispatched(); s.advance_running(ticks=2)
    state = s._job(job["scheduler_job_id"])
    assert state["status"] == "TIMEOUT"
    assert state["elapsed_ticks"] == 2


def test_transient_device_failure_retries_once(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path, keys=("flaky",))
    job = s.submit(p, max_retries=1, timeout_ticks=20)["job"]
    s.dispatch_once(); s.start_dispatched(); s.advance_running(ticks=1)
    assert s._job(job["scheduler_job_id"])["status"] == "QUEUED"
    adapter = next(iter(s.devices.values()))
    adapter.profile["fault_injection"]["tick"] = False
    s.run_until_terminal(max_scheduler_ticks=20)
    final = s._job(job["scheduler_job_id"])
    assert final["status"] == "COMPLETED"
    assert final["attempts_started"] == 2
    assert final["result"]["is_real_measurement"] is False


def test_failure_exhausts_retry_budget(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path, keys=("flaky",))
    job = s.submit(p, max_retries=0)["job"]
    s.dispatch_once(); s.start_dispatched(); s.advance_running(ticks=1)
    assert s._job(job["scheduler_job_id"])["status"] == "FAILED"


def test_blocked_t27_protocol_rejected(tmp_path):
    from experiments.protocol import ExperimentProtocolBuilder
    from scripts.build_v030_t27_fixture import candidates, fixture_template
    blocked = ExperimentProtocolBuilder(fixture_template()).build(candidates()["unsafe"])
    s = make_scheduler(tmp_path)
    with pytest.raises(JobSchedulerValidationError):
        s.submit(blocked)


def test_completed_result_remains_explicitly_synthetic(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path)
    job = s.submit(p)["job"]
    s.run_until_terminal(max_scheduler_ticks=20)
    final = s._job(job["scheduler_job_id"])
    assert final["status"] == "COMPLETED"
    assert final["result"]["measurement_origin"] == "SIMULATOR_FIXTURE"
    assert final["result"]["is_real_measurement"] is False


def test_scheduler_state_is_atomically_persisted_and_terminal_reload_is_safe(tmp_path):
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path, scheduler_id="PERSIST")
    s.submit(p)
    s.run_until_terminal(max_scheduler_ticks=20)
    path = Path(s.snapshot()["state_path"])
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    # T29 guarantees persisted queue/terminal audit state. Active-device crash
    # reconciliation remains a T35 responsibility.
    reloaded = make_scheduler(tmp_path, scheduler_id="PERSIST")
    assert reloaded.snapshot()["counts"]["COMPLETED"] == 1


def test_scheduler_atomic_write_retries_transient_permission_error(tmp_path, monkeypatch):
    import experiments.scheduler as scheduler_module

    real_replace = scheduler_module.os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(scheduler_module.os, "replace", flaky_replace)
    p = protocol_documents()[0]
    s = make_scheduler(tmp_path, scheduler_id="WIN_ATOMIC_RETRY")
    s.submit(p)

    assert calls["count"] >= 3
    state_path = Path(s.snapshot()["state_path"])
    assert state_path.exists()
    assert list(state_path.parent.glob(state_path.name + ".*.tmp")) == []
