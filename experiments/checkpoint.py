from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .campaign import CampaignStore, find_round
from .closed_loop_bo import ClosedLoopBOService
from .dataset_versioning import DatasetVersionStore, sha256_file
from .model_promotion import ModelPromotionService

CHECKPOINT_STAGE = "V0.2-T25_checkpoint_resume"
CHECKPOINT_SCHEMA_VERSION = 1

WORKFLOW_STEPS = [
    "ROUND_COMPLETED",
    "DATASET_UPDATED",
    "MODEL_DECISION_RECORDED",
    "NEXT_ROUND_CREATED",
    "WORKFLOW_COMPLETED",
]


class CheckpointError(RuntimeError):
    pass


class CheckpointValidationError(CheckpointError):
    pass


class CheckpointConflictError(CheckpointError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CheckpointValidationError(f"{name} 不能为空")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", text):
        raise CheckpointValidationError(
            f"{name} 只能包含字母、数字、点、下划线和短横线"
        )
    return text


def _jsonable(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"{name} 必须可 JSON 序列化") from exc


def _fingerprint(context: dict[str, Any]) -> str:
    payload = json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CheckpointStore:
    """Atomic, monotonic workflow checkpoint store.

    Layout:
      .runtime/v020/checkpoints/<campaign_id>/<workflow_id>.json
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)

    def checkpoint_path(self, campaign_id: str, workflow_id: str) -> Path:
        return (
            self.runtime_root
            / "v020"
            / "checkpoints"
            / _safe(campaign_id, "campaign_id")
            / f"{_safe(workflow_id, 'workflow_id')}.json"
        )

    def start_or_resume(
        self,
        *,
        campaign_id: str,
        workflow_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        path = self.checkpoint_path(campaign_id, workflow_id)
        context = _jsonable(context, "context")
        fp = _fingerprint(context)
        now = utc_now_iso()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("stage") != CHECKPOINT_STAGE:
                raise CheckpointConflictError("checkpoint stage 非法")
            if data.get("context_fingerprint") != fp:
                raise CheckpointConflictError(
                    "恢复请求与原 checkpoint 输入不一致，拒绝复用"
                )
            data["resume_count"] = int(data.get("resume_count", 0)) + 1
            data["last_resumed_at"] = now
            self._save(path, data)
            return deepcopy(data)

        data = {
            "stage": CHECKPOINT_STAGE,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "workflow_id": workflow_id,
            "status": "IN_PROGRESS",
            "created_at": now,
            "updated_at": now,
            "last_resumed_at": now,
            "resume_count": 0,
            "context": context,
            "context_fingerprint": fp,
            "completed_steps": [],
            "step_payloads": {},
            "progress": {},
            "last_error": None,
            "events": [
                {
                    "event_id": 1,
                    "event_type": "CHECKPOINT_CREATED",
                    "timestamp": now,
                    "payload": {},
                }
            ],
        }
        self._save(path, data)
        return deepcopy(data)

    def load(self, campaign_id: str, workflow_id: str) -> dict[str, Any]:
        path = self.checkpoint_path(campaign_id, workflow_id)
        if not path.exists():
            raise CheckpointValidationError(f"checkpoint 不存在: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def record_progress(
        self,
        *,
        campaign_id: str,
        workflow_id: str,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        path = self.checkpoint_path(campaign_id, workflow_id)
        data = self.load(campaign_id, workflow_id)
        data["progress"].update(_jsonable(progress, "progress"))
        self._event(data, "PROGRESS_SNAPSHOT", progress)
        self._save(path, data)
        return deepcopy(data)

    def mark_step(
        self,
        *,
        campaign_id: str,
        workflow_id: str,
        step: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if step not in WORKFLOW_STEPS:
            raise CheckpointValidationError(f"未知 workflow step: {step}")
        path = self.checkpoint_path(campaign_id, workflow_id)
        data = self.load(campaign_id, workflow_id)
        completed = list(data.get("completed_steps") or [])
        if step in completed:
            return deepcopy(data)
        expected = WORKFLOW_STEPS[len(completed)]
        if step != expected:
            raise CheckpointConflictError(
                f"checkpoint step 非法跳跃: expected={expected}, got={step}"
            )
        completed.append(step)
        data["completed_steps"] = completed
        data["step_payloads"][step] = _jsonable(payload or {}, "step payload")
        data["last_error"] = None
        if step == "WORKFLOW_COMPLETED":
            data["status"] = "COMPLETED"
        self._event(data, "STEP_COMPLETED", {"step": step})
        self._save(path, data)
        return deepcopy(data)

    def record_error(
        self,
        *,
        campaign_id: str,
        workflow_id: str,
        error: Exception,
    ) -> dict[str, Any]:
        path = self.checkpoint_path(campaign_id, workflow_id)
        data = self.load(campaign_id, workflow_id)
        data["last_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": utc_now_iso(),
        }
        self._event(data, "WORKFLOW_ERROR", data["last_error"])
        self._save(path, data)
        return deepcopy(data)

    def _event(self, data: dict[str, Any], event_type: str, payload: dict[str, Any]) -> None:
        events = data.setdefault("events", [])
        events.append({
            "event_id": len(events) + 1,
            "event_type": event_type,
            "timestamp": utc_now_iso(),
            "payload": _jsonable(payload, "event payload"),
        })

    def _save(self, path: Path, data: dict[str, Any]) -> None:
        data["updated_at"] = utc_now_iso()
        _write_json_atomic(path, data)


class ResumableClosedLoopWorkflow:
    """Checkpointed T22 -> T23 -> T24 coordinator.

    It intentionally does not fabricate or auto-submit experiment results.
    T20 remains the only result-ingestion path. This workflow begins closing
    the loop only after the source round has real terminal results.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.checkpoints = CheckpointStore(runtime_root)
        self.campaigns = CampaignStore(runtime_root)
        self.datasets = DatasetVersionStore(runtime_root)
        self.promotion = ModelPromotionService(runtime_root)
        self.closed_loop = ClosedLoopBOService(runtime_root)

    def resume(
        self,
        *,
        campaign_id: str,
        source_round_id: str,
        parent_dataset_version: str,
        child_dataset_version: str,
        candidate_pool_csv: str | Path,
        target_metric: str,
        target_unit: str,
        gate: dict[str, Any],
        incumbent_model_version: str = "model_v001",
        challenger_model_version: str = "model_v002",
        model_family: str = "ExtraTreesRegressor",
        batch_size: int = 5,
        acquisition: str = "EI",
        direction: str = "maximize",
        random_state: int = 42,
        pause_after_step: str | None = None,
    ) -> dict[str, Any]:
        pool_path = Path(candidate_pool_csv)
        if not pool_path.exists():
            raise CheckpointValidationError(f"candidate_pool_csv 不存在: {pool_path}")
        workflow_id = f"{source_round_id}__to_next_round"
        context = {
            "campaign_id": campaign_id,
            "source_round_id": source_round_id,
            "parent_dataset_version": parent_dataset_version,
            "child_dataset_version": child_dataset_version,
            "candidate_pool_sha256": sha256_file(pool_path),
            "target_metric": target_metric,
            "target_unit": target_unit,
            "gate": gate,
            "incumbent_model_version": incumbent_model_version,
            "challenger_model_version": challenger_model_version,
            "model_family": model_family,
            "batch_size": batch_size,
            "acquisition": acquisition,
            "direction": direction,
            "random_state": random_state,
        }
        checkpoint = self.checkpoints.start_or_resume(
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            context=context,
        )
        resumed = bool(checkpoint.get("completed_steps")) or checkpoint.get("resume_count", 0) > 0

        try:
            checkpoint = self._step_round_completed(
                campaign_id, source_round_id, workflow_id, checkpoint
            )
            if pause_after_step == "ROUND_COMPLETED":
                return self._paused(checkpoint, resumed)

            checkpoint = self._step_dataset_updated(
                campaign_id=campaign_id,
                source_round_id=source_round_id,
                parent_dataset_version=parent_dataset_version,
                child_dataset_version=child_dataset_version,
                workflow_id=workflow_id,
                checkpoint=checkpoint,
            )
            if pause_after_step == "DATASET_UPDATED":
                return self._paused(checkpoint, resumed)

            checkpoint = self._step_model_decision(
                campaign_id=campaign_id,
                source_round_id=source_round_id,
                parent_dataset_version=parent_dataset_version,
                child_dataset_version=child_dataset_version,
                target_metric=target_metric,
                gate=gate,
                incumbent_model_version=incumbent_model_version,
                challenger_model_version=challenger_model_version,
                model_family=model_family,
                random_state=random_state,
                workflow_id=workflow_id,
                checkpoint=checkpoint,
            )
            if pause_after_step == "MODEL_DECISION_RECORDED":
                return self._paused(checkpoint, resumed)

            checkpoint, bo_report = self._step_next_round(
                campaign_id=campaign_id,
                source_round_id=source_round_id,
                child_dataset_version=child_dataset_version,
                candidate_pool_csv=pool_path,
                target_metric=target_metric,
                target_unit=target_unit,
                gate=gate,
                batch_size=batch_size,
                acquisition=acquisition,
                direction=direction,
                random_state=random_state,
                workflow_id=workflow_id,
                checkpoint=checkpoint,
            )
            if pause_after_step == "NEXT_ROUND_CREATED":
                return self._paused(checkpoint, resumed, bo_report=bo_report)

            if "WORKFLOW_COMPLETED" not in checkpoint["completed_steps"]:
                checkpoint = self.checkpoints.mark_step(
                    campaign_id=campaign_id,
                    workflow_id=workflow_id,
                    step="WORKFLOW_COMPLETED",
                    payload={
                        "next_round_id": (
                            checkpoint["step_payloads"]["NEXT_ROUND_CREATED"]
                            ["next_round_id"]
                        )
                    },
                )
            return {
                "status": "COMPLETED",
                "idempotent_replay": resumed and checkpoint.get("status") == "COMPLETED",
                "checkpoint": checkpoint,
                "checkpoint_json": str(
                    self.checkpoints.checkpoint_path(campaign_id, workflow_id)
                ),
                "next_round_id": checkpoint["step_payloads"]["NEXT_ROUND_CREATED"]["next_round_id"],
                "dataset_version": child_dataset_version,
                "model_decision": checkpoint["step_payloads"]["MODEL_DECISION_RECORDED"]["decision"],
                "bo_report": bo_report,
            }
        except Exception as exc:
            self.checkpoints.record_error(
                campaign_id=campaign_id,
                workflow_id=workflow_id,
                error=exc,
            )
            raise

    def _step_round_completed(self, campaign_id, source_round_id, workflow_id, checkpoint):
        if "ROUND_COMPLETED" in checkpoint["completed_steps"]:
            return checkpoint
        campaign = self.campaigns.load(campaign_id)
        round_record = find_round(campaign, source_round_id)
        if round_record.get("status") != "COMPLETED":
            raise CheckpointConflictError(
                "source Round 尚未 COMPLETED，不能进入闭环恢复流程"
            )
        return self.checkpoints.mark_step(
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            step="ROUND_COMPLETED",
            payload={
                "round_status": "COMPLETED",
                "round_progress": deepcopy(round_record.get("progress") or {}),
            },
        )

    def _step_dataset_updated(
        self, *, campaign_id, source_round_id, parent_dataset_version,
        child_dataset_version, workflow_id, checkpoint
    ):
        if "DATASET_UPDATED" in checkpoint["completed_steps"]:
            self.datasets.verify(
                int(self.campaigns.load(campaign_id)["project_id"]),
                child_dataset_version,
            )
            return checkpoint
        campaign = self.campaigns.load(campaign_id)
        project_id = int(campaign["project_id"])
        recovered_existing = False
        if self.datasets.exists(project_id, child_dataset_version):
            manifest = self.datasets.load_manifest(project_id, child_dataset_version)
            source = manifest.get("source") or {}
            if (
                manifest.get("parent_dataset_version") != parent_dataset_version
                or source.get("campaign_id") != campaign_id
                or source.get("round_id") != source_round_id
            ):
                raise CheckpointConflictError(
                    "已存在 child dataset，但 lineage 与 checkpoint 请求不一致"
                )
            self.datasets.verify(project_id, child_dataset_version)
            recovered_existing = True
        else:
            result = self.datasets.update_from_round(
                campaign_store=self.campaigns,
                campaign_id=campaign_id,
                round_id=source_round_id,
                new_dataset_version=child_dataset_version,
            )
            manifest = result["manifest"]
        return self.checkpoints.mark_step(
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            step="DATASET_UPDATED",
            payload={
                "dataset_version": child_dataset_version,
                "row_count": manifest["row_count"],
                "sha256": manifest["sha256"],
                "recovered_existing_dataset": recovered_existing,
            },
        )

    def _step_model_decision(
        self, *, campaign_id, source_round_id, parent_dataset_version,
        child_dataset_version, target_metric, gate, incumbent_model_version,
        challenger_model_version, model_family, random_state, workflow_id,
        checkpoint
    ):
        if "MODEL_DECISION_RECORDED" in checkpoint["completed_steps"]:
            return checkpoint
        project_id = int(self.campaigns.load(campaign_id)["project_id"])
        report = self.promotion.compare_and_register(
            project_id=project_id,
            target_metric=target_metric,
            parent_dataset_version=parent_dataset_version,
            child_dataset_version=child_dataset_version,
            incumbent_model_version=incumbent_model_version,
            challenger_model_version=challenger_model_version,
            model_family=model_family,
            gate=gate,
            folds=5,
            random_state=random_state,
        )
        if report.get("decision") == "BLOCKED":
            raise CheckpointConflictError(
                "T23 model promotion step returned BLOCKED"
            )
        return self.checkpoints.mark_step(
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            step="MODEL_DECISION_RECORDED",
            payload={
                "decision": report.get("decision"),
                "active_model_version": (
                    (report.get("registry") or {}).get("active_model_version_after_decision")
                ),
                "report_json": report.get("report_json"),
                "automatic_activation": False,
            },
        )

    def _step_next_round(
        self, *, campaign_id, source_round_id, child_dataset_version,
        candidate_pool_csv, target_metric, target_unit, gate, batch_size,
        acquisition, direction, random_state, workflow_id, checkpoint
    ):
        if "NEXT_ROUND_CREATED" in checkpoint["completed_steps"]:
            payload = checkpoint["step_payloads"]["NEXT_ROUND_CREATED"]
            report_path = Path(payload["report_json"])
            if not report_path.exists():
                # The checkpoint says this step was committed, but its report
                # disappeared. Re-run T24; T25-enhanced T24 will reuse R2.
                report = self.closed_loop.generate_next_round(
                    campaign_id=campaign_id,
                    source_round_id=source_round_id,
                    latest_dataset_version=child_dataset_version,
                    candidate_pool_csv=candidate_pool_csv,
                    target_metric=target_metric,
                    target_unit=target_unit,
                    gate=gate,
                    batch_size=batch_size,
                    acquisition=acquisition,
                    direction=direction,
                    random_state=random_state,
                )
            else:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["report_json"] = str(report_path)
                report["idempotent_replay"] = True
            return checkpoint, report

        report = self.closed_loop.generate_next_round(
            campaign_id=campaign_id,
            source_round_id=source_round_id,
            latest_dataset_version=child_dataset_version,
            candidate_pool_csv=candidate_pool_csv,
            target_metric=target_metric,
            target_unit=target_unit,
            gate=gate,
            batch_size=batch_size,
            acquisition=acquisition,
            direction=direction,
            random_state=random_state,
        )
        checkpoint = self.checkpoints.mark_step(
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            step="NEXT_ROUND_CREATED",
            payload={
                "next_round_id": report["next_round_id"],
                "report_json": report["report_json"],
                "idempotent_replay": report.get("idempotent_replay", False),
            },
        )
        return checkpoint, report

    def _paused(self, checkpoint, resumed, bo_report=None):
        return {
            "status": "PAUSED",
            "idempotent_replay": resumed,
            "last_completed_step": (
                checkpoint["completed_steps"][-1]
                if checkpoint["completed_steps"] else None
            ),
            "checkpoint": checkpoint,
            "checkpoint_json": str(
                self.checkpoints.checkpoint_path(
                    checkpoint["campaign_id"], checkpoint["workflow_id"]
                )
            ),
            "bo_report": bo_report,
        }
