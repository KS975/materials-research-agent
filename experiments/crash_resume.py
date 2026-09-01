from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from .autonomous_round import AutonomousRoundController
from .campaign import CampaignStore, find_round
from .closed_loop_bo import ClosedLoopBOService
from .dataset_versioning import DatasetVersionStore
from .device import SimulatorDeviceAdapter
from .evaluation import PredictionEvaluationService
from .model_promotion import ModelPromotionService
from .protocol import ExperimentProtocolBuilder, sha256_json
from .result_capture import AutomaticResultCaptureService
from .results import ExperimentalResultService
from .safety import SafetyInterlock, validate_safety_policy
from .scheduler import JobScheduler
from .telemetry import TelemetryRecorder


CRASH_RESUME_STAGE = "V0.3-T35_crash_resume_operator_override"
CRASH_RESUME_SCHEMA_VERSION = 1
OPERATOR_ACTIONS = {"RESUME", "CANCEL_JOB", "ABORT_ROUND"}


class CrashResumeError(RuntimeError):
    code = "CRASH_RESUME_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class CrashResumeValidationError(CrashResumeError):
    code = "CRASH_RESUME_VALIDATION_ERROR"


class CrashResumeConflictError(CrashResumeError):
    code = "CRASH_RESUME_CONFLICT"


class OperatorOverrideRequiredError(CrashResumeError):
    code = "OPERATOR_OVERRIDE_REQUIRED"


class OperatorOverrideBlockedError(CrashResumeError):
    code = "OPERATOR_OVERRIDE_BLOCKED"


class RecoveryIntegrityError(CrashResumeError):
    code = "RECOVERY_AUDIT_INTEGRITY_ERROR"


def _text(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise CrashResumeValidationError(f"{name} 不能为空")
    return out


def _safe_component(value: Any) -> str:
    text = _text(value, "path component")
    return "".join(
        ch if (ch.isalnum() or ch in "._-") else "_"
        for ch in text
    )[:140]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class CrashResumeCoordinator:
    """T35 crash/restart reconciliation for deterministic simulator jobs.

    The simulator process itself has no durable device memory. After a simulated
    process crash, T35 reconstructs the *same deterministic simulator job* from
    the persisted T29 scheduler job and its elapsed_ticks, then validates that
    adapter_job_id/progress match before execution can continue.

    This is intentionally NOT a claim that a real instrument can be restored by
    replaying commands. A real device adapter must query its physical controller
    by durable job identity in a future hardware integration.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.campaigns = CampaignStore(self.runtime_root)
        self.results = ExperimentalResultService(str(self.runtime_root))
        self.capture = AutomaticResultCaptureService(self.runtime_root)
        self.datasets = DatasetVersionStore(self.runtime_root)
        self.evaluations = PredictionEvaluationService(self.runtime_root)
        self.models = ModelPromotionService(self.runtime_root)
        self.bo = ClosedLoopBOService(self.runtime_root)

    def recovery_dir(self, campaign_id: str, round_id: str) -> Path:
        return (
            self.runtime_root
            / "v030"
            / "crash_resume"
            / _safe_component(campaign_id)
            / _safe_component(round_id)
        )

    def checkpoint_path(self, campaign_id: str, round_id: str) -> Path:
        return self.recovery_dir(campaign_id, round_id) / "crash_checkpoint.json"

    def state_path(self, campaign_id: str, round_id: str) -> Path:
        return self.recovery_dir(campaign_id, round_id) / "recovery_state.json"

    def report_path(self, campaign_id: str, round_id: str) -> Path:
        return self.recovery_dir(campaign_id, round_id) / "recovery_report.json"

    @staticmethod
    def scheduler_id(campaign_id: str, round_id: str) -> str:
        return f"{campaign_id}_{round_id}_T35"

    @staticmethod
    def telemetry_session_id(
        campaign_id: str,
        round_id: str,
        candidate_id: str,
    ) -> str:
        return f"{campaign_id}_{round_id}_{candidate_id}_T35"

    @staticmethod
    def safety_interlock_id(
        campaign_id: str,
        round_id: str,
        candidate_id: str,
    ) -> str:
        return f"{campaign_id}_{round_id}_{candidate_id}_T35"

    def _load_recovery_state(
        self,
        campaign_id: str,
        round_id: str,
    ) -> dict[str, Any]:
        path = self.state_path(campaign_id, round_id)
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            self._verify_events(state.get("events") or [])
            return state
        state = {
            "stage": CRASH_RESUME_STAGE,
            "schema_version": CRASH_RESUME_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "round_id": round_id,
            "events": [],
        }
        self._append_event(
            state,
            "RECOVERY_STATE_CREATED",
            payload={},
        )
        _atomic_json(path, state)
        return state

    def _save_recovery_state(
        self,
        campaign_id: str,
        round_id: str,
        state: dict[str, Any],
    ) -> None:
        _atomic_json(self.state_path(campaign_id, round_id), state)

    def _append_event(
        self,
        state: dict[str, Any],
        event_type: str,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        events = state.setdefault("events", [])
        seq = len(events) + 1
        previous_sha = events[-1]["record_sha256"] if events else None
        base = {
            "stage": CRASH_RESUME_STAGE,
            "schema_version": CRASH_RESUME_SCHEMA_VERSION,
            "sequence": seq,
            "event_type": _text(event_type, "event_type"),
            "previous_sha256": previous_sha,
            "payload": deepcopy(payload),
        }
        row = dict(base)
        row["record_sha256"] = sha256_json(base)
        events.append(row)
        return row

    @staticmethod
    def _verify_events(events: list[dict[str, Any]]) -> bool:
        previous = None
        for event in events:
            if event.get("previous_sha256") != previous:
                raise RecoveryIntegrityError(
                    "recovery audit previous_sha256 不一致"
                )
            base = {
                key: value
                for key, value in event.items()
                if key != "record_sha256"
            }
            expected = sha256_json(base)
            if event.get("record_sha256") != expected:
                raise RecoveryIntegrityError(
                    "recovery audit record_sha256 校验失败"
                )
            previous = event["record_sha256"]
        return True

    def verify_integrity(
        self,
        campaign_id: str,
        round_id: str,
    ) -> bool:
        state = self._load_recovery_state(campaign_id, round_id)
        return self._verify_events(state.get("events") or [])

    def _build_protocols(
        self,
        campaign_id: str,
        round_id: str,
        protocol_template: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        campaign = self.campaigns.load(campaign_id)
        round_record = find_round(campaign, round_id)
        experiments = round_record.get("experiments") or []
        if not experiments:
            raise CrashResumeValidationError(
                "Round 尚未注册 planned experiments"
            )
        builder = ExperimentProtocolBuilder(protocol_template)
        protocols = {}
        for experiment in experiments:
            candidate_id = experiment["candidate_id"]
            protocol = builder.build({
                "candidate_id": candidate_id,
                "source_context": {
                    "campaign_id": campaign_id,
                    "round_id": round_id,
                    "recovery_stage": "T35",
                },
                "features": deepcopy(experiment.get("features") or {}),
            })
            if protocol.get("status") != "READY":
                raise CrashResumeConflictError(
                    f"T27 protocol 未 READY: {candidate_id}",
                    details={"protocol": protocol},
                )
            protocols[candidate_id] = protocol
        return protocols

    def _safety_map(
        self,
        *,
        campaign_id: str,
        round_id: str,
        protocols: dict[str, dict[str, Any]],
        safety_policy: dict[str, Any],
        preflight_missing: bool,
    ) -> dict[str, SafetyInterlock]:
        normalized = validate_safety_policy(safety_policy)
        result = {}
        for candidate_id, protocol in protocols.items():
            interlock = SafetyInterlock(
                interlock_id=self.safety_interlock_id(
                    campaign_id, round_id, candidate_id
                ),
                policy=normalized,
                runtime_root=self.runtime_root,
            )
            if preflight_missing:
                snap = interlock.snapshot()
                if (
                    snap["state"] == "SAFE"
                    and len(snap["trip_history"]) == 0
                    and snap["event_count"] == 1
                ):
                    checked = interlock.check_protocol(protocol)
                    if not checked.get("allowed"):
                        raise CrashResumeConflictError(
                            f"T31 preflight 阻断: {candidate_id}",
                            details=checked,
                        )
            result[candidate_id] = interlock
        return result

    def _recorder(
        self,
        *,
        campaign_id: str,
        round_id: str,
        candidate_id: str,
        scheduler_job_id: str,
        device_id: str,
        protocol: dict[str, Any],
    ) -> TelemetryRecorder:
        return TelemetryRecorder(
            session_id=self.telemetry_session_id(
                campaign_id, round_id, candidate_id
            ),
            scheduler_job_id=scheduler_job_id,
            experiment_id=candidate_id,
            device_id=device_id,
            protocol=protocol,
            runtime_root=self.runtime_root,
        )

    def _receipt_count(self, campaign_id: str, round_id: str) -> int:
        path = (
            self.runtime_root
            / "v030"
            / "result_capture"
            / campaign_id
            / round_id
        )
        return len(list(path.glob("*.json"))) if path.exists() else 0

    def stage_simulated_crash(
        self,
        *,
        campaign_id: str,
        round_id: str,
        protocol_template: dict[str, Any],
        device_profile: dict[str, Any],
        safety_policy: dict[str, Any],
        completed_before_crash: int = 2,
        active_elapsed_ticks: int = 4,
        scheduler_timeout_ticks: int = 30,
    ) -> dict[str, Any]:
        """Create a deterministic mid-round crash fixture.

        This method intentionally stops while one scheduler job is still
        RUNNING. Returning from the method represents loss of the in-process
        adapter object. Persisted Scheduler/Telemetry/Safety/T20 state remains.
        """
        if completed_before_crash < 0:
            raise CrashResumeValidationError(
                "completed_before_crash 不能小于 0"
            )
        if active_elapsed_ticks <= 0:
            raise CrashResumeValidationError(
                "active_elapsed_ticks 必须 > 0"
            )
        checkpoint = self.checkpoint_path(campaign_id, round_id)
        if checkpoint.exists():
            return json.loads(checkpoint.read_text(encoding="utf-8"))

        campaign = self.campaigns.load(campaign_id)
        round_record = find_round(campaign, round_id)
        if round_record.get("status") != "PLANNED":
            raise CrashResumeConflictError(
                "stage_simulated_crash 要求 Round=PLANNED"
            )

        protocols = self._build_protocols(
            campaign_id, round_id, protocol_template
        )
        experiments = round_record.get("experiments") or []
        if completed_before_crash >= len(experiments):
            raise CrashResumeValidationError(
                "必须至少留下一个 active experiment"
            )

        adapter = SimulatorDeviceAdapter(device_profile)
        policy = validate_safety_policy(safety_policy)
        if policy["device_id"] != adapter.device_id:
            raise CrashResumeValidationError(
                "safety policy device_id 与 simulator 不一致"
            )
        safety = self._safety_map(
            campaign_id=campaign_id,
            round_id=round_id,
            protocols=protocols,
            safety_policy=policy,
            preflight_missing=True,
        )

        self.campaigns.transition_round(
            campaign_id,
            round_id=round_id,
            new_status="RUNNING",
            reason="T35 simulated pre-crash execution",
        )

        scheduler = JobScheduler(
            scheduler_id=self.scheduler_id(campaign_id, round_id),
            devices={adapter.device_id: adapter},
            runtime_root=self.runtime_root,
        )
        for index, experiment in enumerate(experiments, start=1):
            scheduler.submit(
                protocols[experiment["candidate_id"]],
                priority=len(experiments) - index,
                timeout_ticks=scheduler_timeout_ticks,
                max_retries=0,
            )

        completed_candidates: set[str] = set()
        active_candidate = None
        active_progress = None
        active_scheduler_job_id = None

        while True:
            scheduler.dispatch_once(max_jobs=1)
            started = scheduler.start_dispatched()
            for job in started:
                candidate_id = job["candidate_id"]
                recorder = self._recorder(
                    campaign_id=campaign_id,
                    round_id=round_id,
                    candidate_id=candidate_id,
                    scheduler_job_id=job["scheduler_job_id"],
                    device_id=adapter.device_id,
                    protocol=protocols[candidate_id],
                )
                row = recorder.capture(adapter)["record"]
                safe = safety[candidate_id].monitor_telemetry(adapter, row)
                if not safe.get("safe"):
                    raise CrashResumeConflictError(
                        "T35 pre-crash fixture unexpectedly hit SAFETY_STOP"
                    )

            running = [
                job for job in scheduler.snapshot()["jobs"]
                if job["status"] == "RUNNING"
            ]
            if not running:
                continue
            job = running[0]
            candidate_id = job["candidate_id"]

            if len(completed_candidates) >= completed_before_crash:
                while True:
                    current = next(
                        x for x in scheduler.snapshot()["jobs"]
                        if x["scheduler_job_id"] == job["scheduler_job_id"]
                    )
                    if int(current["elapsed_ticks"]) >= active_elapsed_ticks:
                        active_candidate = candidate_id
                        active_progress = float(
                            (adapter.status().get("job") or {}).get(
                                "progress", 0.0
                            )
                        )
                        active_scheduler_job_id = job["scheduler_job_id"]
                        break
                    scheduler.advance_running(ticks=1)
                    recorder = self._recorder(
                        campaign_id=campaign_id,
                        round_id=round_id,
                        candidate_id=candidate_id,
                        scheduler_job_id=job["scheduler_job_id"],
                        device_id=adapter.device_id,
                        protocol=protocols[candidate_id],
                    )
                    row = recorder.capture(adapter)["record"]
                    safe = safety[candidate_id].monitor_telemetry(
                        adapter, row
                    )
                    if not safe.get("safe"):
                        raise CrashResumeConflictError(
                            "T35 crash point unexpectedly hit SAFETY_STOP"
                        )
                break

            scheduler.advance_running(ticks=1)
            recorder = self._recorder(
                campaign_id=campaign_id,
                round_id=round_id,
                candidate_id=candidate_id,
                scheduler_job_id=job["scheduler_job_id"],
                device_id=adapter.device_id,
                protocol=protocols[candidate_id],
            )
            row = recorder.capture(adapter)["record"]
            safe = safety[candidate_id].monitor_telemetry(adapter, row)
            if not safe.get("safe"):
                raise CrashResumeConflictError(
                    "T35 pre-crash run unexpectedly hit SAFETY_STOP"
                )
            latest = next(
                x for x in scheduler.snapshot()["jobs"]
                if x["scheduler_job_id"] == job["scheduler_job_id"]
            )
            if latest["status"] == "COMPLETED":
                self.capture.capture(
                    campaign_id,
                    round_id=round_id,
                    adapter=adapter,
                    protocol=protocols[candidate_id],
                    safety_interlock=safety[candidate_id],
                    evaluate=True,
                )
                completed_candidates.add(candidate_id)

        scheduler_snapshot = scheduler.snapshot()
        t20 = self.results.summary(campaign_id, round_id=round_id)
        active_job = next(
            job for job in scheduler_snapshot["jobs"]
            if job["scheduler_job_id"] == active_scheduler_job_id
        )
        recorder = self._recorder(
            campaign_id=campaign_id,
            round_id=round_id,
            candidate_id=active_candidate,
            scheduler_job_id=active_scheduler_job_id,
            device_id=adapter.device_id,
            protocol=protocols[active_candidate],
        )
        telemetry_snapshot = recorder.snapshot()

        payload = {
            "stage": CRASH_RESUME_STAGE,
            "schema_version": CRASH_RESUME_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "round_id": round_id,
            "scheduler_id": scheduler.scheduler_id,
            "simulated_crash": True,
            "real_process_crash": False,
            "completed_results_before_crash": t20["progress"]["completed"],
            "pending_results_before_crash": t20["progress"]["pending"],
            "capture_receipts_before_crash": self._receipt_count(
                campaign_id, round_id
            ),
            "active_candidate_id": active_candidate,
            "active_scheduler_job_id": active_scheduler_job_id,
            "active_adapter_job_id": active_job["adapter_job_id"],
            "scheduler_elapsed_ticks": active_job["elapsed_ticks"],
            "device_progress_percent": active_progress,
            "telemetry_phase": telemetry_snapshot["phase"],
            "telemetry_count": telemetry_snapshot["telemetry_count"],
            "scheduler_state_path": scheduler_snapshot["state_path"],
            "round_status": find_round(
                self.campaigns.load(campaign_id), round_id
            )["status"],
            "is_real_measurement": False,
            "real_device_connected": False,
        }
        _atomic_json(checkpoint, payload)

        state = self._load_recovery_state(campaign_id, round_id)
        self._append_event(
            state,
            "SIMULATED_PROCESS_CRASH",
            payload={
                "active_candidate_id": active_candidate,
                "scheduler_elapsed_ticks": active_job["elapsed_ticks"],
                "device_progress_percent": active_progress,
                "completed_results": t20["progress"]["completed"],
            },
        )
        self._save_recovery_state(campaign_id, round_id, state)
        return deepcopy(payload)

    def _reconstruct_active_simulator(
        self,
        *,
        scheduler: JobScheduler,
        adapter: SimulatorDeviceAdapter,
        active_job: dict[str, Any],
    ) -> dict[str, Any]:
        if active_job["status"] not in {"DISPATCHED", "RUNNING"}:
            raise CrashResumeConflictError(
                "只能重建 DISPATCHED/RUNNING simulator job"
            )
        protocol = active_job["protocol"]
        adapter.connect()
        adapter.prepare(protocol)
        submitted = adapter.submit_protocol(protocol)
        rebuilt_job_id = (submitted.get("job") or {}).get("job_id")
        if rebuilt_job_id != active_job.get("adapter_job_id"):
            raise CrashResumeConflictError(
                "重建 simulator adapter_job_id 与 scheduler 不一致",
                details={
                    "expected": active_job.get("adapter_job_id"),
                    "actual": rebuilt_job_id,
                },
            )

        if active_job["status"] == "DISPATCHED":
            return {
                "device_state": adapter.status()["state"],
                "replayed_ticks": 0,
                "progress_percent": 0.0,
            }

        adapter.start()
        replayed_ticks = int(active_job.get("elapsed_ticks") or 0)
        if replayed_ticks > 0:
            adapter.tick(steps=replayed_ticks)
        status = adapter.status()
        if status["state"] != "RUNNING":
            raise CrashResumeConflictError(
                "重建后的 simulator 不再是 RUNNING；"
                "scheduler active state 可能与持久化信息冲突",
                details={
                    "scheduler_elapsed_ticks": replayed_ticks,
                    "device_state": status["state"],
                },
            )
        return {
            "device_state": status["state"],
            "replayed_ticks": replayed_ticks,
            "progress_percent": float(
                (status.get("job") or {}).get("progress", 0.0)
            ),
        }

    def _record_operator_action(
        self,
        *,
        campaign_id: str,
        round_id: str,
        operator_id: str,
        action: str,
        reason: str,
        candidate_id: str | None,
    ) -> None:
        state = self._load_recovery_state(campaign_id, round_id)
        self._append_event(
            state,
            "OPERATOR_OVERRIDE",
            payload={
                "operator_id": operator_id,
                "action": action,
                "reason": reason,
                "candidate_id": candidate_id,
            },
        )
        self._save_recovery_state(campaign_id, round_id, state)

    def _finalize_completed_round(
        self,
        *,
        campaign_id: str,
        round_id: str,
        target_metric: str,
        target_unit: str,
        candidate_pool_csv: str | Path,
        gate: dict[str, Any],
        child_dataset_version: str,
        challenger_model_version: str,
        next_batch_size: int,
    ) -> dict[str, Any]:
        summary = self.results.summary(campaign_id, round_id=round_id)
        if not summary["can_close_round"]:
            raise CrashResumeConflictError(
                "恢复后 T20 尚不能关闭 Round",
                details=summary,
            )

        round_record = find_round(
            self.campaigns.load(campaign_id), round_id
        )
        if round_record["status"] != "COMPLETED":
            self.campaigns.transition_round(
                campaign_id,
                round_id=round_id,
                new_status="COMPLETED",
                reason="T35 crash recovery finished all experiments",
            )

        evaluation = self.evaluations.evaluate(
            campaign_id,
            round_id=round_id,
            metric=target_metric,
            persist=True,
        )

        campaign = self.campaigns.load(campaign_id)
        round_record = find_round(campaign, round_id)
        parent_dataset_version = round_record["plan"]["dataset_version"]

        dataset_update = self.datasets.update_from_round(
            campaign_store=self.campaigns,
            campaign_id=campaign_id,
            round_id=round_id,
            new_dataset_version=child_dataset_version,
        )
        manifest = dataset_update["manifest"]

        incumbent_model_version = str(
            (round_record["plan"].get("model_versions") or {}).get(
                target_metric, "model_v001"
            )
        )
        promotion = self.models.compare_and_register(
            project_id=int(campaign["project_id"]),
            target_metric=target_metric,
            parent_dataset_version=parent_dataset_version,
            child_dataset_version=child_dataset_version,
            incumbent_model_version=incumbent_model_version,
            challenger_model_version=challenger_model_version,
            model_family="ExtraTreesRegressor",
            gate=gate,
            folds=5,
            random_state=35,
            holdout_fraction=0.25,
        )
        registry = self.models.registry.load_registry(
            int(campaign["project_id"]), target_metric
        )

        bo_report = self.bo.generate_next_round(
            campaign_id=campaign_id,
            source_round_id=round_id,
            latest_dataset_version=child_dataset_version,
            candidate_pool_csv=candidate_pool_csv,
            target_metric=target_metric,
            target_unit=target_unit,
            gate=gate,
            batch_size=next_batch_size,
            acquisition="EI",
            direction="maximize",
            xi=0.01,
            kappa=2.0,
            min_batch_distance=0.20,
            soft_penalty_weight=0.10,
            allow_borderline_for_exploration=True,
            random_state=35,
        )
        next_round = find_round(
            self.campaigns.load(campaign_id),
            bo_report["next_round_id"],
        )
        return {
            "evaluation": evaluation,
            "dataset": {
                "parent_dataset_version": parent_dataset_version,
                "child_dataset_version": child_dataset_version,
                "row_count_before": manifest["row_count_before"],
                "added_row_count": manifest["added_row_count"],
                "row_count_after": manifest["row_count_after"],
            },
            "model_governance": {
                "decision": promotion["decision"],
                "active_model_version": registry.get("active_model_version"),
                "challenger_model_version": challenger_model_version,
                "automatic_activation": False,
            },
            "next_round": {
                "round_id": next_round["round_id"],
                "status": next_round["status"],
                "dataset_version": next_round["plan"]["dataset_version"],
                "planned_experiments": len(
                    next_round.get("experiments") or []
                ),
            },
        }

    def resume_after_crash(
        self,
        *,
        campaign_id: str,
        round_id: str,
        protocol_template: dict[str, Any],
        device_profile: dict[str, Any],
        safety_policy: dict[str, Any],
        operator_action: str,
        operator_id: str,
        reason: str,
        candidate_pool_csv: str | Path | None = None,
        target_metric: str | None = None,
        target_unit: str | None = None,
        gate: dict[str, Any] | None = None,
        child_dataset_version: str | None = None,
        challenger_model_version: str | None = None,
        next_batch_size: int = 5,
        max_scheduler_ticks: int = 1000,
    ) -> dict[str, Any]:
        operator_action = _text(
            operator_action, "operator_action"
        ).upper()
        if operator_action not in OPERATOR_ACTIONS:
            raise CrashResumeValidationError(
                "operator_action 必须是 RESUME / CANCEL_JOB / ABORT_ROUND"
            )
        operator_id = _text(operator_id, "operator_id")
        reason = _text(reason, "reason")

        if operator_action == "RESUME":
            report_path = self.report_path(campaign_id, round_id)
            if report_path.exists():
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                report["idempotent_replay"] = True
                report["report_json"] = str(report_path)
                return report

        checkpoint_path = self.checkpoint_path(campaign_id, round_id)
        if not checkpoint_path.exists():
            raise CrashResumeConflictError(
                "缺少 T35 crash_checkpoint.json"
            )
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        campaign = self.campaigns.load(campaign_id)
        round_record = find_round(campaign, round_id)
        if round_record["status"] not in {
            "RUNNING", "PARTIALLY_COMPLETED"
        }:
            raise CrashResumeConflictError(
                "crash recovery 要求 Round 为 RUNNING / PARTIALLY_COMPLETED"
            )

        protocols = self._build_protocols(
            campaign_id, round_id, protocol_template
        )
        adapter = SimulatorDeviceAdapter(device_profile)
        normalized_policy = validate_safety_policy(safety_policy)
        if normalized_policy["device_id"] != adapter.device_id:
            raise CrashResumeValidationError(
                "safety policy device_id 与 simulator 不一致"
            )
        safety = self._safety_map(
            campaign_id=campaign_id,
            round_id=round_id,
            protocols=protocols,
            safety_policy=normalized_policy,
            preflight_missing=True,
        )
        scheduler = JobScheduler(
            scheduler_id=self.scheduler_id(campaign_id, round_id),
            devices={adapter.device_id: adapter},
            runtime_root=self.runtime_root,
        )
        scheduler_snapshot = scheduler.snapshot()
        active_jobs = [
            job for job in scheduler_snapshot["jobs"]
            if job["status"] in {"DISPATCHED", "RUNNING"}
        ]
        if len(active_jobs) != 1:
            raise CrashResumeConflictError(
                "T35 simulator acceptance 要求恰好一个 active job",
                details={
                    "active_count": len(active_jobs),
                    "counts": scheduler_snapshot["counts"],
                },
            )
        active_job = active_jobs[0]
        candidate_id = active_job["candidate_id"]

        if operator_action == "RESUME":
            safety_snapshot = safety[candidate_id].snapshot()
            if safety_snapshot["state"] != "SAFE":
                raise OperatorOverrideBlockedError(
                    "T31 SAFETY_STOP 未清除，operator RESUME 不能绕过安全锁",
                    details={
                        "candidate_id": candidate_id,
                        "safety_state": safety_snapshot["state"],
                    },
                )

        rebuild = self._reconstruct_active_simulator(
            scheduler=scheduler,
            adapter=adapter,
            active_job=active_job,
        )

        # Checkpoint is evidence, but persisted scheduler is the recovery source
        # of truth. This flags stale checkpoint claims instead of following them.
        checkpoint_stale = (
            int(checkpoint.get("scheduler_elapsed_ticks") or 0)
            != int(active_job.get("elapsed_ticks") or 0)
            or abs(
                float(checkpoint.get("device_progress_percent") or 0.0)
                - float(rebuild["progress_percent"])
            ) > 1e-9
        )

        recorder = self._recorder(
            campaign_id=campaign_id,
            round_id=round_id,
            candidate_id=candidate_id,
            scheduler_job_id=active_job["scheduler_job_id"],
            device_id=adapter.device_id,
            protocol=protocols[candidate_id],
        )
        replay_capture = recorder.capture(adapter)

        self._record_operator_action(
            campaign_id=campaign_id,
            round_id=round_id,
            operator_id=operator_id,
            action=operator_action,
            reason=reason,
            candidate_id=candidate_id,
        )

        if operator_action == "CANCEL_JOB":
            cancelled = scheduler.cancel_job(
                active_job["scheduler_job_id"]
            )
            result = {
                "stage": CRASH_RESUME_STAGE,
                "schema_version": CRASH_RESUME_SCHEMA_VERSION,
                "operator_action": "CANCEL_JOB",
                "operator_id": operator_id,
                "candidate_id": candidate_id,
                "scheduler_job_status": cancelled["job"]["status"],
                "round_status": find_round(
                    self.campaigns.load(campaign_id), round_id
                )["status"],
                "automatic_continuation": False,
                "dataset_updated": False,
                "next_round_created": False,
                "checkpoint_stale_detected": checkpoint_stale,
                "recovery_audit_valid": self.verify_integrity(
                    campaign_id, round_id
                ),
            }
            return result

        if operator_action == "ABORT_ROUND":
            for job in scheduler.snapshot()["jobs"]:
                if job["status"] in {
                    "QUEUED", "DISPATCHED", "RUNNING"
                }:
                    scheduler.cancel_job(job["scheduler_job_id"])
            self.campaigns.transition_round(
                campaign_id,
                round_id=round_id,
                new_status="CANCELLED",
                reason=(
                    f"T35 operator {operator_id} ABORT_ROUND: {reason}"
                ),
            )
            result = {
                "stage": CRASH_RESUME_STAGE,
                "schema_version": CRASH_RESUME_SCHEMA_VERSION,
                "operator_action": "ABORT_ROUND",
                "operator_id": operator_id,
                "round_status": "CANCELLED",
                "active_jobs_after_abort": 0,
                "dataset_updated": False,
                "model_governance_run": False,
                "next_round_created": False,
                "checkpoint_stale_detected": checkpoint_stale,
                "recovery_audit_valid": self.verify_integrity(
                    campaign_id, round_id
                ),
            }
            return result

        if candidate_pool_csv is None:
            raise OperatorOverrideRequiredError(
                "RESUME 完整闭环需要 candidate_pool_csv"
            )
        target_metric = _text(target_metric, "target_metric")
        target_unit = _text(target_unit, "target_unit")
        child_dataset_version = _text(
            child_dataset_version, "child_dataset_version"
        )
        challenger_model_version = _text(
            challenger_model_version,
            "challenger_model_version",
        )
        if not isinstance(gate, dict):
            raise CrashResumeValidationError("gate 必须是 object")

        captured_candidates = {
            experiment["candidate_id"]
            for experiment in (
                find_round(
                    self.campaigns.load(campaign_id), round_id
                ).get("experiments") or []
            )
            if experiment.get("result") is not None
        }
        completed_before_resume = len(captured_candidates)
        receipt_count_before_resume = self._receipt_count(
            campaign_id, round_id
        )
        scheduler_ticks_after_resume = 0

        # If the crash occurred in DISPATCHED state, the reconstructed adapter is
        # SUBMITTED and the persisted scheduler should perform the normal start.
        if active_job["status"] == "DISPATCHED":
            scheduler.start_dispatched()

        while scheduler_ticks_after_resume < max_scheduler_ticks:
            snap = scheduler.snapshot()
            active_count = (
                snap["counts"]["QUEUED"]
                + snap["counts"]["DISPATCHED"]
                + snap["counts"]["RUNNING"]
            )
            if active_count == 0:
                break

            scheduler.dispatch_once(max_jobs=1)
            started = scheduler.start_dispatched()
            for job in started:
                cid = job["candidate_id"]
                rec = self._recorder(
                    campaign_id=campaign_id,
                    round_id=round_id,
                    candidate_id=cid,
                    scheduler_job_id=job["scheduler_job_id"],
                    device_id=adapter.device_id,
                    protocol=protocols[cid],
                )
                row = rec.capture(adapter)["record"]
                safe = safety[cid].monitor_telemetry(adapter, row)
                if not safe.get("safe"):
                    raise OperatorOverrideBlockedError(
                        f"T31 runtime SAFETY_STOP: {cid}",
                        details=safe,
                    )

            changed = scheduler.advance_running(ticks=1)
            scheduler_ticks_after_resume += 1
            current_jobs = {
                job["scheduler_job_id"]: job
                for job in scheduler.snapshot()["jobs"]
            }
            relevant = {
                job["scheduler_job_id"]
                for job in current_jobs.values()
                if job["status"] == "RUNNING"
            } | {
                job["scheduler_job_id"]
                for job in changed
                if job["status"] in {
                    "COMPLETED", "FAILED", "TIMEOUT"
                }
            }
            for scheduler_job_id in sorted(relevant):
                job = current_jobs[scheduler_job_id]
                cid = job["candidate_id"]
                rec = self._recorder(
                    campaign_id=campaign_id,
                    round_id=round_id,
                    candidate_id=cid,
                    scheduler_job_id=scheduler_job_id,
                    device_id=adapter.device_id,
                    protocol=protocols[cid],
                )
                row = rec.capture(adapter)["record"]
                safe = safety[cid].monitor_telemetry(adapter, row)
                if not safe.get("safe"):
                    raise OperatorOverrideBlockedError(
                        f"T31 runtime SAFETY_STOP: {cid}",
                        details=safe,
                    )
                if (
                    job["status"] == "COMPLETED"
                    and cid not in captured_candidates
                ):
                    self.capture.capture(
                        campaign_id,
                        round_id=round_id,
                        adapter=adapter,
                        protocol=protocols[cid],
                        safety_interlock=safety[cid],
                        evaluate=True,
                    )
                    captured_candidates.add(cid)

            bad = [
                job for job in scheduler.snapshot()["jobs"]
                if job["status"] in {
                    "FAILED", "TIMEOUT", "CANCELLED"
                }
            ]
            if bad:
                raise CrashResumeConflictError(
                    "RESUME 后 scheduler 出现异常终态",
                    details={"jobs": bad},
                )
        else:
            raise CrashResumeConflictError(
                "T35 RESUME 超过 max_scheduler_ticks"
            )

        final_scheduler = scheduler.snapshot()
        if final_scheduler["counts"]["COMPLETED"] != len(protocols):
            raise CrashResumeConflictError(
                "恢复后 scheduler 未全部完成",
                details={"counts": final_scheduler["counts"]},
            )

        finalization = self._finalize_completed_round(
            campaign_id=campaign_id,
            round_id=round_id,
            target_metric=target_metric,
            target_unit=target_unit,
            candidate_pool_csv=candidate_pool_csv,
            gate=gate,
            child_dataset_version=child_dataset_version,
            challenger_model_version=challenger_model_version,
            next_batch_size=next_batch_size,
        )
        final_campaign = self.campaigns.load(campaign_id)
        t20 = self.results.summary(campaign_id, round_id=round_id)

        report = {
            "stage": CRASH_RESUME_STAGE,
            "schema_version": CRASH_RESUME_SCHEMA_VERSION,
            "idempotent_replay": False,
            "campaign_id": campaign_id,
            "round_id": round_id,
            "operator": {
                "operator_id": operator_id,
                "action": "RESUME",
                "reason": reason,
            },
            "crash_point": {
                "completed_results_before_restart": checkpoint[
                    "completed_results_before_crash"
                ],
                "pending_results_before_restart": checkpoint[
                    "pending_results_before_crash"
                ],
                "active_candidate_id": checkpoint[
                    "active_candidate_id"
                ],
                "checkpoint_progress_percent": checkpoint[
                    "device_progress_percent"
                ],
                "scheduler_elapsed_ticks": active_job["elapsed_ticks"],
            },
            "reconciliation": {
                "source_of_truth": "PERSISTED_T29_SCHEDULER",
                "simulator_replayed_ticks": rebuild["replayed_ticks"],
                "reconstructed_progress_percent": rebuild[
                    "progress_percent"
                ],
                "adapter_job_id_match": True,
                "telemetry_replay_idempotent": bool(
                    replay_capture["idempotent_replay"]
                ),
                "checkpoint_stale_detected": checkpoint_stale,
                "automatic_resume_used": False,
                "explicit_operator_resume_required": True,
            },
            "results": {
                "completed_before_resume": completed_before_resume,
                "completed_after_resume": t20["progress"]["completed"],
                "pending_after_resume": t20["progress"]["pending"],
                "capture_receipts_before_resume": receipt_count_before_resume,
                "capture_receipts_after_resume": self._receipt_count(
                    campaign_id, round_id
                ),
                "duplicate_completed_result_writes": 0,
                "measurement_origin": "SIMULATOR_FIXTURE",
                "is_real_measurement": False,
            },
            "scheduler": {
                "ticks_after_resume": scheduler_ticks_after_resume,
                "counts": final_scheduler["counts"],
            },
            "dataset": finalization["dataset"],
            "model_governance": finalization[
                "model_governance"
            ],
            "next_round": finalization["next_round"],
            "round_count": len(final_campaign["rounds"]),
            "recovery_audit_valid": self.verify_integrity(
                campaign_id, round_id
            ),
            "boundary": {
                "real_device_connected": False,
                "simulator_reconstruction_only": True,
                "real_device_recovery_requires_physical_job_query": True,
                "operator_override_cannot_bypass_safety": True,
                "model_promotion_auto_approved": False,
            },
        }
        report_path = self.report_path(campaign_id, round_id)
        _atomic_json(report_path, report)
        report["report_json"] = str(report_path)
        return deepcopy(report)
