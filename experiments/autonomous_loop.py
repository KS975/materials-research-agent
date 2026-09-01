from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from .autonomous_round import AutonomousRoundController
from .campaign import CampaignStore, find_round
from .protocol import sha256_json


AUTONOMOUS_LOOP_STAGE = "V0.3-T34_multi_round_autonomous_loop"
AUTONOMOUS_LOOP_SCHEMA_VERSION = 1


class AutonomousLoopError(RuntimeError):
    code = "AUTONOMOUS_LOOP_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class AutonomousLoopValidationError(AutonomousLoopError):
    code = "AUTONOMOUS_LOOP_VALIDATION_ERROR"


class AutonomousLoopConflictError(AutonomousLoopError):
    code = "AUTONOMOUS_LOOP_CONFLICT"


def _text(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise AutonomousLoopValidationError(f"{name} 不能为空")
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


def _feature_key(features: dict[str, Any]) -> str:
    canonical = {
        str(key): value
        for key, value in sorted((features or {}).items())
    }
    return sha256_json(canonical)


class AutonomousMultiRoundLoop:
    """T34: execute a bounded number of autonomous simulator rounds.

    T34 composes T33 rather than duplicating its execution logic. By default it
    runs three rounds. R1/R2 create a PLANNED next round through T24; the final
    round explicitly calls T33 with create_next_round=False so no accidental R4
    is created.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.campaigns = CampaignStore(self.runtime_root)
        self.round_controller = AutonomousRoundController(self.runtime_root)

    def report_path(self, campaign_id: str) -> Path:
        return (
            self.runtime_root
            / "v030"
            / "autonomous_loop"
            / _safe_component(campaign_id)
            / "autonomous_loop_report.json"
        )

    def _load_existing(self, campaign_id: str) -> dict[str, Any] | None:
        path = self.report_path(campaign_id)
        if not path.exists():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        report["idempotent_replay"] = True
        report["report_json"] = str(path)
        return report

    def run(
        self,
        *,
        campaign_id: str,
        first_round_id: str,
        protocol_template: dict[str, Any],
        device_profile: dict[str, Any],
        safety_policy: dict[str, Any],
        candidate_pool_csv: str | Path,
        target_metric: str,
        target_unit: str,
        gate: dict[str, Any],
        dataset_versions: list[str],
        challenger_model_versions: list[str],
        rounds_to_run: int = 3,
        active_incumbent_model_version: str = "model_v001",
        batch_size: int = 5,
        scheduler_timeout_ticks: int = 30,
    ) -> dict[str, Any]:
        existing = self._load_existing(campaign_id)
        if existing is not None:
            return existing

        campaign_id = _text(campaign_id, "campaign_id")
        first_round_id = _text(first_round_id, "first_round_id")
        target_metric = _text(target_metric, "target_metric")
        target_unit = _text(target_unit, "target_unit")
        active_incumbent_model_version = _text(
            active_incumbent_model_version,
            "active_incumbent_model_version",
        )
        if rounds_to_run <= 0:
            raise AutonomousLoopValidationError("rounds_to_run 必须 > 0")
        if len(dataset_versions) != rounds_to_run + 1:
            raise AutonomousLoopValidationError(
                "dataset_versions 数量必须等于 rounds_to_run + 1"
            )
        if len(challenger_model_versions) != rounds_to_run:
            raise AutonomousLoopValidationError(
                "challenger_model_versions 数量必须等于 rounds_to_run"
            )
        if len(set(dataset_versions)) != len(dataset_versions):
            raise AutonomousLoopValidationError("dataset_versions 不能重复")
        if len(set(challenger_model_versions)) != len(challenger_model_versions):
            raise AutonomousLoopValidationError(
                "challenger_model_versions 不能重复"
            )
        if gate.get("training_allowed") is not True:
            raise AutonomousLoopConflictError(
                "Modeling Gate training_allowed=false"
            )
        if gate.get("official_model_allowed") is not True:
            raise AutonomousLoopConflictError(
                "Modeling Gate official_model_allowed=false"
            )

        campaign = self.campaigns.load(campaign_id)
        first_round = find_round(campaign, first_round_id)
        if first_round.get("status") != "PLANNED":
            raise AutonomousLoopConflictError(
                "T34 fresh run 要求 first Round=PLANNED"
            )
        if (first_round.get("plan") or {}).get("dataset_version") != dataset_versions[0]:
            raise AutonomousLoopConflictError(
                "first Round dataset_version 与 dataset_versions[0] 不一致"
            )

        current_round_id = first_round_id
        round_reports: list[dict[str, Any]] = []
        bo_selected_ood_count = 0

        for index in range(rounds_to_run):
            is_final = index == rounds_to_run - 1
            child_dataset_version = dataset_versions[index + 1]
            challenger_model_version = challenger_model_versions[index]

            report = self.round_controller.run_one_round(
                campaign_id=campaign_id,
                round_id=current_round_id,
                protocol_template=protocol_template,
                device_profile=device_profile,
                safety_policy=safety_policy,
                candidate_pool_csv=candidate_pool_csv,
                target_metric=target_metric,
                target_unit=target_unit,
                gate=gate,
                child_dataset_version=child_dataset_version,
                incumbent_model_version=active_incumbent_model_version,
                challenger_model_version=challenger_model_version,
                next_batch_size=batch_size,
                scheduler_timeout_ticks=scheduler_timeout_ticks,
                create_next_round=not is_final,
            )
            round_reports.append(report)

            if report["source_round_status"] != "COMPLETED":
                raise AutonomousLoopConflictError(
                    f"Round 未完成: {current_round_id}"
                )
            if report["result_capture"]["manual_result_submission_count"] != 0:
                raise AutonomousLoopConflictError(
                    "T34 检测到人工 result submission"
                )
            if report["safety"]["runtime_safety_stop_count"] != 0:
                raise AutonomousLoopConflictError(
                    f"Round 触发 SAFETY_STOP: {current_round_id}"
                )
            if report["model_governance"]["automatic_activation"]:
                raise AutonomousLoopConflictError(
                    "T34 禁止自动激活 challenger model"
                )
            if report["dataset"]["child_dataset_version"] != child_dataset_version:
                raise AutonomousLoopConflictError(
                    "T22 child dataset version 与 T34 计划不一致"
                )

            if not is_final:
                next_round = report.get("next_round")
                if not next_round:
                    raise AutonomousLoopConflictError(
                        "非最终轮没有生成 next Round"
                    )
                if next_round["status"] != "PLANNED":
                    raise AutonomousLoopConflictError(
                        "next Round 必须以 PLANNED 创建"
                    )
                bo_selected_ood_count += int(
                    next_round.get("selected_out_of_domain_count", 0)
                )
                current_round_id = next_round["round_id"]
            else:
                if report.get("next_round") is not None:
                    raise AutonomousLoopConflictError(
                        "最终轮不应创建 next Round"
                    )

        final_campaign = self.campaigns.load(campaign_id)
        if len(final_campaign.get("rounds") or []) != rounds_to_run:
            raise AutonomousLoopConflictError(
                "完成 T34 后 Campaign Round 数量不等于 rounds_to_run",
                details={
                    "round_count": len(final_campaign.get("rounds") or []),
                    "expected": rounds_to_run,
                },
            )

        round_ids = [r["round_id"] for r in final_campaign["rounds"]]
        round_statuses = [r["status"] for r in final_campaign["rounds"]]
        candidate_ids: list[str] = []
        feature_keys: list[str] = []
        for round_record in final_campaign["rounds"]:
            for experiment in round_record.get("experiments") or []:
                candidate_ids.append(str(experiment.get("candidate_id") or ""))
                feature_keys.append(_feature_key(experiment.get("features") or {}))

        duplicate_candidate_ids = len(candidate_ids) - len(set(candidate_ids))
        duplicate_feature_points = len(feature_keys) - len(set(feature_keys))

        total_automatic_captures = sum(
            r["result_capture"]["automatic_capture_count"]
            for r in round_reports
        )
        total_manual_submissions = sum(
            r["result_capture"]["manual_result_submission_count"]
            for r in round_reports
        )
        total_receipts = sum(
            r["result_capture"]["receipt_count"]
            for r in round_reports
        )
        total_safety_stops = sum(
            r["safety"]["runtime_safety_stop_count"]
            for r in round_reports
        )
        all_telemetry_completed = all(
            r["telemetry"]["all_completed"] for r in round_reports
        )
        all_telemetry_hash_valid = all(
            r["telemetry"]["all_hash_chains_valid"] for r in round_reports
        )
        all_no_auto_activation = all(
            not r["model_governance"]["automatic_activation"]
            for r in round_reports
        )
        model_decisions = [
            r["model_governance"]["decision"] for r in round_reports
        ]
        active_models = [
            r["model_governance"]["active_model_version"]
            for r in round_reports
        ]

        dataset_lineage = [dataset_versions[0]] + [
            r["dataset"]["child_dataset_version"] for r in round_reports
        ]
        row_counts = [round_reports[0]["dataset"]["row_count_before"]] + [
            r["dataset"]["row_count_after"] for r in round_reports
        ]
        rows_added_per_round = [
            r["dataset"]["added_row_count"] for r in round_reports
        ]

        report = {
            "stage": AUTONOMOUS_LOOP_STAGE,
            "schema_version": AUTONOMOUS_LOOP_SCHEMA_VERSION,
            "idempotent_replay": False,
            "campaign_id": campaign_id,
            "project_id": int(final_campaign["project_id"]),
            "rounds_requested": rounds_to_run,
            "round_count": len(final_campaign["rounds"]),
            "round_ids": round_ids,
            "round_statuses": round_statuses,
            "all_rounds_completed": all(
                status == "COMPLETED" for status in round_statuses
            ),
            "experiments": {
                "total_planned": len(candidate_ids),
                "automatic_capture_count": total_automatic_captures,
                "manual_result_submission_count": total_manual_submissions,
                "receipt_count": total_receipts,
                "duplicate_candidate_ids": duplicate_candidate_ids,
                "duplicate_feature_points": duplicate_feature_points,
                "measurement_origin": "SIMULATOR_FIXTURE",
                "is_real_measurement": False,
            },
            "dataset": {
                "lineage": dataset_lineage,
                "row_counts": row_counts,
                "rows_added_per_round": rows_added_per_round,
                "initial_rows": row_counts[0],
                "final_rows": row_counts[-1],
            },
            "safety": {
                "runtime_safety_stop_count": total_safety_stops,
                "all_telemetry_completed": all_telemetry_completed,
                "all_telemetry_hash_chains_valid": all_telemetry_hash_valid,
                "automatic_resume_used": False,
                "real_device_connected": False,
            },
            "model_governance": {
                "decisions": model_decisions,
                "active_model_versions": active_models,
                "automatic_activation_count": sum(
                    1
                    for r in round_reports
                    if r["model_governance"]["automatic_activation"]
                ),
                "all_rounds_no_auto_activation": all_no_auto_activation,
            },
            "bayesian_optimization": {
                "bo_transition_count": max(rounds_to_run - 1, 0),
                "selected_out_of_domain_count": bo_selected_ood_count,
                "final_round_created_next_round": False,
            },
            "round_reports": [
                {
                    "round_id": r["source_round_id"],
                    "report_json": r["report_json"],
                    "dataset": deepcopy(r["dataset"]),
                    "model_decision": r["model_governance"]["decision"],
                    "next_round_id": (
                        r["next_round"]["round_id"]
                        if r.get("next_round") else None
                    ),
                }
                for r in round_reports
            ],
            "boundary": {
                "bounded_rounds": True,
                "rounds_to_run": rounds_to_run,
                "no_round_after_final": True,
                "real_device_connected": False,
                "simulator_measurements_are_not_real_material_data": True,
                "model_promotion_never_auto_approved": True,
                "crash_resume_owned_by_t35": True,
            },
        }

        if duplicate_candidate_ids != 0:
            raise AutonomousLoopConflictError(
                "T34 检测到跨 Round candidate_id 重复"
            )
        if duplicate_feature_points != 0:
            raise AutonomousLoopConflictError(
                "T34 检测到跨 Round feature point 重复"
            )
        if bo_selected_ood_count != 0:
            raise AutonomousLoopConflictError(
                "T34 BO 选择了 OUT_OF_DOMAIN candidate"
            )
        if total_manual_submissions != 0:
            raise AutonomousLoopConflictError(
                "T34 manual result submission count 必须为 0"
            )
        if total_safety_stops != 0:
            raise AutonomousLoopConflictError(
                "T34 正常 fixture 不应触发 SAFETY_STOP"
            )
        if not all_no_auto_activation:
            raise AutonomousLoopConflictError(
                "T34 检测到自动模型激活"
            )

        path = self.report_path(campaign_id)
        _atomic_json(path, report)
        report["report_json"] = str(path)
        return deepcopy(report)
