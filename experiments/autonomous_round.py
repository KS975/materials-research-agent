from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from .campaign import CampaignStore, find_round
from .closed_loop_bo import ClosedLoopBOService
from .dataset_versioning import DatasetVersionStore
from .device import SimulatorDeviceAdapter
from .evaluation import PredictionEvaluationService
from .model_promotion import ModelPromotionService
from .protocol import ExperimentProtocolBuilder
from .result_capture import AutomaticResultCaptureService
from .results import ExperimentalResultService
from .safety import SafetyInterlock, validate_safety_policy
from .scheduler import JobScheduler
from .telemetry import TelemetryRecorder


AUTONOMOUS_ROUND_STAGE = "V0.3-T33_autonomous_round_controller"
AUTONOMOUS_ROUND_SCHEMA_VERSION = 1


class AutonomousRoundError(RuntimeError):
    code = "AUTONOMOUS_ROUND_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class AutonomousRoundValidationError(AutonomousRoundError):
    code = "AUTONOMOUS_ROUND_VALIDATION_ERROR"


class AutonomousRoundConflictError(AutonomousRoundError):
    code = "AUTONOMOUS_ROUND_CONFLICT"


def _text(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise AutonomousRoundValidationError(f"{name} 不能为空")
    return out


def _safe_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise AutonomousRoundValidationError("路径标识不能为空")
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


class AutonomousRoundController:
    """T33: run exactly one autonomous simulator feedback round.

    This controller deliberately orchestrates existing frozen components rather
    than reimplementing their logic:
      T27 protocol
      T28 simulator adapter
      T29 scheduler
      T30 telemetry
      T31 safety
      T32 result capture -> T20/T21
      T22 dataset versioning
      T23 model governance
      T24 next-round BO

    T33 does not run an infinite loop. It executes one source Round and creates
    exactly one PLANNED next Round.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.campaigns = CampaignStore(self.runtime_root)
        self.results = ExperimentalResultService(str(self.runtime_root))
        self.datasets = DatasetVersionStore(self.runtime_root)
        self.evaluations = PredictionEvaluationService(self.runtime_root)
        self.models = ModelPromotionService(self.runtime_root)
        self.bo = ClosedLoopBOService(self.runtime_root)
        self.capture = AutomaticResultCaptureService(self.runtime_root)

    def _report_path(self, campaign_id: str, round_id: str) -> Path:
        return (
            self.runtime_root
            / "v030"
            / "autonomous_round"
            / _safe_component(campaign_id)
            / _safe_component(round_id)
            / "autonomous_round_report.json"
        )

    def _load_existing(self, campaign_id: str, round_id: str) -> dict[str, Any] | None:
        path = self._report_path(campaign_id, round_id)
        if not path.exists():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        report["idempotent_replay"] = True
        report["report_json"] = str(path)
        return report

    def _build_protocols(
        self,
        *,
        campaign_id: str,
        round_id: str,
        template: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        campaign = self.campaigns.load(campaign_id)
        round_record = find_round(campaign, round_id)
        experiments = round_record.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise AutonomousRoundValidationError(
                "Round 必须先注册 planned experiments"
            )

        builder = ExperimentProtocolBuilder(template)
        protocols: dict[str, dict[str, Any]] = {}
        for experiment in experiments:
            candidate_id = _text(experiment.get("candidate_id"), "candidate_id")
            protocol = builder.build({
                "candidate_id": candidate_id,
                "source_context": {
                    "campaign_id": campaign_id,
                    "round_id": round_id,
                },
                "features": deepcopy(experiment.get("features") or {}),
            })
            if protocol.get("status") != "READY":
                raise AutonomousRoundConflictError(
                    f"T27 protocol 未 READY: {candidate_id}",
                    details={
                        "candidate_id": candidate_id,
                        "status": protocol.get("status"),
                        "issues": protocol.get("issues"),
                    },
                )
            protocols[candidate_id] = protocol
        return protocols

    def run_one_round(
        self,
        *,
        campaign_id: str,
        round_id: str,
        protocol_template: dict[str, Any],
        device_profile: dict[str, Any],
        safety_policy: dict[str, Any],
        candidate_pool_csv: str | Path,
        target_metric: str,
        target_unit: str,
        gate: dict[str, Any],
        child_dataset_version: str,
        incumbent_model_version: str = "model_v001",
        challenger_model_version: str = "model_v002",
        next_batch_size: int = 5,
        scheduler_timeout_ticks: int = 40,
        max_scheduler_ticks: int = 1000,
        create_next_round: bool = True,
    ) -> dict[str, Any]:
        existing = self._load_existing(campaign_id, round_id)
        if existing is not None:
            return existing

        target_metric = _text(target_metric, "target_metric")
        target_unit = _text(target_unit, "target_unit")
        child_dataset_version = _text(
            child_dataset_version, "child_dataset_version"
        )
        if gate.get("training_allowed") is not True:
            raise AutonomousRoundConflictError(
                "Modeling Gate training_allowed=false"
            )
        if gate.get("official_model_allowed") is not True:
            raise AutonomousRoundConflictError(
                "Modeling Gate official_model_allowed=false"
            )

        campaign = self.campaigns.load(campaign_id)
        source_round = find_round(campaign, round_id)
        if source_round.get("status") != "PLANNED":
            raise AutonomousRoundConflictError(
                "T33 fresh run 要求 source Round=PLANNED；"
                "已完成重放必须存在 autonomous_round_report.json"
            )

        parent_dataset_version = _text(
            (source_round.get("plan") or {}).get("dataset_version"),
            "round.plan.dataset_version",
        )
        planned_count = int(
            (source_round.get("plan") or {}).get(
                "planned_experiment_count", 0
            )
        )
        experiments = source_round.get("experiments") or []
        if len(experiments) != planned_count or planned_count <= 0:
            raise AutonomousRoundValidationError(
                "Round planned_experiment_count 与 experiments 数量不一致"
            )

        adapter = SimulatorDeviceAdapter(device_profile)
        normalized_policy = validate_safety_policy(safety_policy)
        if normalized_policy["device_id"] != adapter.device_id:
            raise AutonomousRoundValidationError(
                "safety policy device_id 与 device profile 不一致"
            )

        protocols = self._build_protocols(
            campaign_id=campaign_id,
            round_id=round_id,
            template=protocol_template,
        )

        # Independent T31 preflight for every T27 READY protocol.
        safety_by_candidate: dict[str, SafetyInterlock] = {}
        for candidate_id, protocol in protocols.items():
            interlock = SafetyInterlock(
                interlock_id=(
                    f"{campaign_id}_{round_id}_{candidate_id}_T33"
                ),
                policy=normalized_policy,
                runtime_root=self.runtime_root,
            )
            preflight = interlock.check_protocol(protocol)
            if not preflight.get("allowed"):
                raise AutonomousRoundConflictError(
                    f"T31 protocol preflight 阻断: {candidate_id}",
                    details=preflight,
                )
            safety_by_candidate[candidate_id] = interlock

        self.campaigns.transition_round(
            campaign_id,
            round_id=round_id,
            new_status="RUNNING",
            reason="V0.3-T33 autonomous execution start",
        )

        scheduler = JobScheduler(
            scheduler_id=f"{campaign_id}_{round_id}_T33",
            devices={adapter.device_id: adapter},
            runtime_root=self.runtime_root,
        )
        scheduler_jobs: dict[str, dict[str, Any]] = {}
        for index, experiment in enumerate(experiments, start=1):
            candidate_id = experiment["candidate_id"]
            submitted = scheduler.submit(
                protocols[candidate_id],
                priority=planned_count - index,
                timeout_ticks=scheduler_timeout_ticks,
                max_retries=0,
            )
            scheduler_jobs[candidate_id] = submitted["job"]

        telemetry_by_candidate: dict[str, TelemetryRecorder] = {}
        captured_candidates: set[str] = set()
        safety_stop_count = 0
        scheduler_ticks = 0

        while scheduler_ticks < max_scheduler_ticks:
            snap = scheduler.snapshot()
            active = (
                snap["counts"]["QUEUED"]
                + snap["counts"]["DISPATCHED"]
                + snap["counts"]["RUNNING"]
            )
            if active == 0:
                break

            scheduler.dispatch_once(max_jobs=1)
            started = scheduler.start_dispatched()

            # Capture PREPARING immediately when a job starts.
            for job in started:
                candidate_id = job["candidate_id"]
                recorder = telemetry_by_candidate.get(candidate_id)
                if recorder is None:
                    recorder = TelemetryRecorder(
                        session_id=(
                            f"{campaign_id}_{round_id}_{candidate_id}"
                        ),
                        scheduler_job_id=job["scheduler_job_id"],
                        experiment_id=candidate_id,
                        device_id=adapter.device_id,
                        protocol=protocols[candidate_id],
                        runtime_root=self.runtime_root,
                    )
                    telemetry_by_candidate[candidate_id] = recorder
                row = recorder.capture(adapter)["record"]
                safe = safety_by_candidate[candidate_id].monitor_telemetry(
                    adapter, row
                )
                if not safe.get("safe"):
                    safety_stop_count += 1
                    raise AutonomousRoundConflictError(
                        f"T31 runtime SAFETY_STOP: {candidate_id}",
                        details=safe,
                    )

            changed = scheduler.advance_running(ticks=1)
            scheduler_ticks += 1

            # Capture the new state after each deterministic simulator tick.
            current = scheduler.snapshot()
            current_jobs = {
                j["scheduler_job_id"]: j for j in current["jobs"]
            }
            relevant_ids = {
                j["scheduler_job_id"]
                for j in current["jobs"]
                if j["status"] == "RUNNING"
            } | {
                j["scheduler_job_id"]
                for j in changed
                if j["status"] in {"COMPLETED", "FAILED", "TIMEOUT"}
            }

            for scheduler_job_id in sorted(relevant_ids):
                job = current_jobs.get(scheduler_job_id)
                if job is None:
                    continue
                candidate_id = job["candidate_id"]
                recorder = telemetry_by_candidate.get(candidate_id)
                if recorder is None:
                    recorder = TelemetryRecorder(
                        session_id=(
                            f"{campaign_id}_{round_id}_{candidate_id}"
                        ),
                        scheduler_job_id=scheduler_job_id,
                        experiment_id=candidate_id,
                        device_id=adapter.device_id,
                        protocol=protocols[candidate_id],
                        runtime_root=self.runtime_root,
                    )
                    telemetry_by_candidate[candidate_id] = recorder
                row = recorder.capture(adapter)["record"]
                safe = safety_by_candidate[candidate_id].monitor_telemetry(
                    adapter, row
                )
                if not safe.get("safe"):
                    safety_stop_count += 1
                    raise AutonomousRoundConflictError(
                        f"T31 runtime SAFETY_STOP: {candidate_id}",
                        details=safe,
                    )

                if (
                    job["status"] == "COMPLETED"
                    and candidate_id not in captured_candidates
                ):
                    self.capture.capture(
                        campaign_id,
                        round_id=round_id,
                        adapter=adapter,
                        protocol=protocols[candidate_id],
                        safety_interlock=safety_by_candidate[candidate_id],
                        evaluate=True,
                    )
                    captured_candidates.add(candidate_id)

            terminal_bad = [
                j for j in scheduler.snapshot()["jobs"]
                if j["status"] in {"FAILED", "TIMEOUT", "CANCELLED"}
            ]
            if terminal_bad:
                raise AutonomousRoundConflictError(
                    "T33 scheduler 出现非 COMPLETED 终态",
                    details={"jobs": terminal_bad},
                )
        else:
            raise AutonomousRoundConflictError(
                "T33 超过 max_scheduler_ticks"
            )

        scheduler_final = scheduler.snapshot()
        if scheduler_final["counts"]["COMPLETED"] != planned_count:
            raise AutonomousRoundConflictError(
                "Scheduler 未完成全部 planned experiments",
                details={"counts": scheduler_final["counts"]},
            )
        if len(captured_candidates) != planned_count:
            raise AutonomousRoundConflictError(
                "T32 未自动回流全部设备结果",
                details={
                    "captured": sorted(captured_candidates),
                    "planned": planned_count,
                },
            )

        result_summary = self.results.summary(
            campaign_id,
            round_id=round_id,
        )
        if not result_summary.get("can_close_round"):
            raise AutonomousRoundConflictError(
                "T20 结果尚不足以关闭 Round",
                details=result_summary,
            )

        self.campaigns.transition_round(
            campaign_id,
            round_id=round_id,
            new_status="COMPLETED",
            reason="V0.3-T33 all autonomous simulator results captured",
        )

        evaluation = self.evaluations.evaluate(
            campaign_id,
            round_id=round_id,
            metric=target_metric,
            persist=True,
        )

        dataset_update = self.datasets.update_from_round(
            campaign_store=self.campaigns,
            campaign_id=campaign_id,
            round_id=round_id,
            new_dataset_version=child_dataset_version,
        )
        child_manifest = dataset_update["manifest"]

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
            random_state=33,
            holdout_fraction=0.25,
        )

        # No automatic approval, even when T23 returns PROMOTE.
        registry = self.models.registry.load_registry(
            int(campaign["project_id"]), target_metric
        )
        active_model_version = registry.get("active_model_version")

        bo_report = None
        next_round = None
        if create_next_round:
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
                random_state=33,
            )

            final_campaign = self.campaigns.load(campaign_id)
            next_round = find_round(
                final_campaign, bo_report["next_round_id"]
            )

        telemetry_summary = {}
        for candidate_id, recorder in telemetry_by_candidate.items():
            snap = recorder.snapshot()
            telemetry_summary[candidate_id] = {
                "phase": snap["phase"],
                "phase_history": snap["phase_history"],
                "telemetry_count": snap["telemetry_count"],
                "event_count": snap["event_count"],
                "hash_chain_valid": snap["hash_chain_valid"],
                "is_real_telemetry": snap["is_real_telemetry"],
            }

        receipt_dir = (
            self.runtime_root
            / "v030"
            / "result_capture"
            / campaign_id
            / round_id
        )
        receipt_count = (
            len(list(receipt_dir.glob("*.json")))
            if receipt_dir.exists() else 0
        )

        report = {
            "stage": AUTONOMOUS_ROUND_STAGE,
            "schema_version": AUTONOMOUS_ROUND_SCHEMA_VERSION,
            "idempotent_replay": False,
            "campaign_id": campaign_id,
            "project_id": int(campaign["project_id"]),
            "source_round_id": round_id,
            "source_round_status": "COMPLETED",
            "planned_experiments": planned_count,
            "scheduler": {
                "scheduler_id": scheduler.scheduler_id,
                "ticks": scheduler_ticks,
                "counts": scheduler_final["counts"],
                "state_path": scheduler_final["state_path"],
            },
            "protocol": {
                "ready_count": len(protocols),
                "blocked_count": 0,
            },
            "safety": {
                "preflight_pass_count": len(protocols),
                "runtime_safety_stop_count": safety_stop_count,
                "automatic_resume_used": False,
                "real_device_connected": False,
            },
            "telemetry": {
                "sessions": telemetry_summary,
                "all_completed": all(
                    x["phase"] == "COMPLETED"
                    for x in telemetry_summary.values()
                ),
                "all_hash_chains_valid": all(
                    x["hash_chain_valid"]
                    for x in telemetry_summary.values()
                ),
                "all_simulator": all(
                    x["is_real_telemetry"] is False
                    for x in telemetry_summary.values()
                ),
            },
            "result_capture": {
                "automatic_capture_count": len(captured_candidates),
                "manual_result_submission_count": 0,
                "receipt_count": receipt_count,
                "measurement_origin": "SIMULATOR_FIXTURE",
                "is_real_measurement": False,
            },
            "evaluation": {
                "evaluated": evaluation["counts"]["evaluated"],
                "mae": evaluation["aggregate"]["mae"],
                "rmse": evaluation["aggregate"]["rmse"],
                "r2": evaluation["aggregate"]["r2"],
                "report_json": evaluation.get("report_json"),
            },
            "dataset": {
                "parent_dataset_version": parent_dataset_version,
                "child_dataset_version": child_dataset_version,
                "row_count_before": child_manifest["row_count_before"],
                "added_row_count": child_manifest["added_row_count"],
                "row_count_after": child_manifest["row_count_after"],
                "sha256": child_manifest["sha256"],
            },
            "model_governance": {
                "decision": promotion["decision"],
                "incumbent_model_version": incumbent_model_version,
                "challenger_model_version": challenger_model_version,
                "active_model_version": active_model_version,
                "automatic_activation": False,
                "human_approval_required": (
                    promotion["decision"] == "PROMOTE"
                ),
                "report_json": promotion.get("report_json"),
            },
            "next_round": (
                {
                    "round_id": next_round["round_id"],
                    "status": next_round["status"],
                    "dataset_version": next_round["plan"]["dataset_version"],
                    "planned_experiments": len(
                        next_round.get("experiments") or []
                    ),
                    "bo_report_json": bo_report.get("report_json"),
                    "selected_out_of_domain_count": sum(
                        1
                        for item in bo_report.get("next_experiments", [])
                        if (item.get("applicability_domain") or {}).get("status")
                        == "OUT_OF_DOMAIN"
                    ),
                }
                if next_round is not None and bo_report is not None
                else None
            ),
            "boundary": {
                "one_round_only": True,
                "create_next_round": bool(create_next_round),
                "next_round_not_auto_started": True,
                "real_device_connected": False,
                "simulator_measurements_are_not_real_material_data": True,
                "model_promotion_never_auto_approved": True,
            },
        }

        path = self._report_path(campaign_id, round_id)
        _atomic_json(path, report)
        report["report_json"] = str(path)
        return deepcopy(report)
