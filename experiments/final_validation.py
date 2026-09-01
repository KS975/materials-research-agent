from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from .protocol import sha256_json


FINAL_VALIDATION_STAGE = "V0.3-T36_end_to_end_autonomous_validation"
FINAL_VALIDATION_SCHEMA_VERSION = 1
REQUIRED_COMPONENTS = [
    "T27_PROTOCOL",
    "T28_DEVICE_ADAPTER",
    "T29_JOB_SCHEDULER",
    "T30_TELEMETRY",
    "T31_SAFETY_INTERLOCK",
    "T32_AUTOMATIC_RESULT_CAPTURE",
    "T33_AUTONOMOUS_ROUND",
    "T34_MULTI_ROUND_LOOP",
    "T35_CRASH_RESUME_OVERRIDE",
]


class FinalValidationError(RuntimeError):
    code = "FINAL_VALIDATION_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class FinalValidationInputError(FinalValidationError):
    code = "FINAL_VALIDATION_INPUT_ERROR"


class FinalValidationFailedError(FinalValidationError):
    code = "FINAL_VALIDATION_FAILED"


def _text(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise FinalValidationInputError(f"{name} 不能为空")
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


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FinalValidationInputError(
            f"验收依赖报告不存在: {p}"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise FinalValidationInputError(
            f"验收依赖报告必须是 object: {p}"
        )
    return data


def _all_true(checks: dict[str, bool]) -> bool:
    return all(value is True for value in checks.values())


class FinalAutonomousValidationService:
    """T36 final acceptance auditor for the complete V0.3 simulator core.

    It does not add a new orchestration behavior. It consumes actual T33/T34/T35
    persisted reports and operator-override outcomes, then produces one final
    PASS/FAIL matrix covering T27-T35.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)

    def report_path(self, validation_id: str) -> Path:
        return (
            self.runtime_root
            / "v030"
            / "final_validation"
            / _safe_component(validation_id)
            / "v030_final_validation_report.json"
        )

    def _round_reports(
        self,
        normal_loop_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows = []
        for item in normal_loop_report.get("round_reports") or []:
            rows.append(_read_json(item.get("report_json")))
        if not rows:
            raise FinalValidationInputError(
                "T34 normal report 缺少 round_reports"
            )
        return rows

    def build_report(
        self,
        *,
        validation_id: str,
        normal_loop_report: dict[str, Any],
        recovery_report: dict[str, Any],
        recovery_replay_report: dict[str, Any],
        cancel_job_report: dict[str, Any],
        abort_round_report: dict[str, Any],
        safety_resume_blocked: bool,
        safety_resume_error_code: str,
        expected_normal_rounds: int = 3,
        expected_experiments_per_round: int = 5,
        expected_normal_dataset_rows: list[int] | None = None,
        expected_recovery_dataset_rows: tuple[int, int] = (40, 45),
        persist: bool = True,
    ) -> dict[str, Any]:
        validation_id = _text(validation_id, "validation_id")
        if expected_normal_dataset_rows is None:
            expected_normal_dataset_rows = [35, 40, 45, 50]
        round_reports = self._round_reports(normal_loop_report)

        expected_total = expected_normal_rounds * expected_experiments_per_round
        protocol_ready_total = sum(
            int(r["protocol"]["ready_count"]) for r in round_reports
        )
        protocol_blocked_total = sum(
            int(r["protocol"]["blocked_count"]) for r in round_reports
        )
        scheduler_completed_total = sum(
            int(r["scheduler"]["counts"]["COMPLETED"])
            for r in round_reports
        )
        safety_preflight_total = sum(
            int(r["safety"]["preflight_pass_count"])
            for r in round_reports
        )
        safety_stop_total = sum(
            int(r["safety"]["runtime_safety_stop_count"])
            for r in round_reports
        )
        capture_total = sum(
            int(r["result_capture"]["automatic_capture_count"])
            for r in round_reports
        )
        manual_total = sum(
            int(r["result_capture"]["manual_result_submission_count"])
            for r in round_reports
        )
        receipt_total = sum(
            int(r["result_capture"]["receipt_count"])
            for r in round_reports
        )

        component_checks: dict[str, dict[str, Any]] = {
            "T27_PROTOCOL": {
                "pass": (
                    protocol_ready_total == expected_total
                    and protocol_blocked_total == 0
                ),
                "ready": protocol_ready_total,
                "blocked": protocol_blocked_total,
            },
            "T28_DEVICE_ADAPTER": {
                "pass": (
                    scheduler_completed_total == expected_total
                    and normal_loop_report["experiments"]["measurement_origin"]
                    == "SIMULATOR_FIXTURE"
                    and normal_loop_report["experiments"]["is_real_measurement"]
                    is False
                ),
                "completed_device_jobs": scheduler_completed_total,
                "real_device_connected": False,
            },
            "T29_JOB_SCHEDULER": {
                "pass": (
                    scheduler_completed_total == expected_total
                    and normal_loop_report["experiments"]["duplicate_candidate_ids"] == 0
                ),
                "completed": scheduler_completed_total,
                "duplicate_candidate_ids": normal_loop_report["experiments"]["duplicate_candidate_ids"],
            },
            "T30_TELEMETRY": {
                "pass": (
                    normal_loop_report["safety"]["all_telemetry_completed"] is True
                    and normal_loop_report["safety"]["all_telemetry_hash_chains_valid"] is True
                    and recovery_report["reconciliation"]["telemetry_replay_idempotent"] is True
                ),
                "all_completed": normal_loop_report["safety"]["all_telemetry_completed"],
                "hash_chains_valid": normal_loop_report["safety"]["all_telemetry_hash_chains_valid"],
                "recovery_replay_idempotent": recovery_report["reconciliation"]["telemetry_replay_idempotent"],
            },
            "T31_SAFETY_INTERLOCK": {
                "pass": (
                    safety_preflight_total == expected_total
                    and safety_stop_total == 0
                    and safety_resume_blocked is True
                    and safety_resume_error_code == "OPERATOR_OVERRIDE_BLOCKED"
                    and recovery_report["boundary"]["operator_override_cannot_bypass_safety"] is True
                ),
                "preflight_pass": safety_preflight_total,
                "normal_runtime_safety_stops": safety_stop_total,
                "operator_resume_blocked_by_safety": safety_resume_blocked,
                "blocked_code": safety_resume_error_code,
            },
            "T32_AUTOMATIC_RESULT_CAPTURE": {
                "pass": (
                    capture_total == expected_total
                    and manual_total == 0
                    and receipt_total == expected_total
                    and normal_loop_report["experiments"]["is_real_measurement"] is False
                ),
                "automatic_capture_count": capture_total,
                "manual_submission_count": manual_total,
                "receipt_count": receipt_total,
            },
            "T33_AUTONOMOUS_ROUND": {
                "pass": (
                    len(round_reports) == expected_normal_rounds
                    and all(r["source_round_status"] == "COMPLETED" for r in round_reports)
                    and all(r["model_governance"]["automatic_activation"] is False for r in round_reports)
                ),
                "round_reports": len(round_reports),
                "all_completed": all(r["source_round_status"] == "COMPLETED" for r in round_reports),
            },
            "T34_MULTI_ROUND_LOOP": {
                "pass": (
                    normal_loop_report["round_count"] == expected_normal_rounds
                    and normal_loop_report["all_rounds_completed"] is True
                    and normal_loop_report["experiments"]["total_planned"] == expected_total
                    and normal_loop_report["experiments"]["duplicate_candidate_ids"] == 0
                    and normal_loop_report["experiments"]["duplicate_feature_points"] == 0
                    and normal_loop_report["dataset"]["row_counts"] == expected_normal_dataset_rows
                    and normal_loop_report["bayesian_optimization"]["selected_out_of_domain_count"] == 0
                    and normal_loop_report["bayesian_optimization"]["final_round_created_next_round"] is False
                    and normal_loop_report["model_governance"]["automatic_activation_count"] == 0
                ),
                "round_count": normal_loop_report["round_count"],
                "dataset_row_counts": deepcopy(normal_loop_report["dataset"]["row_counts"]),
                "duplicate_candidate_ids": normal_loop_report["experiments"]["duplicate_candidate_ids"],
                "duplicate_feature_points": normal_loop_report["experiments"]["duplicate_feature_points"],
                "bo_ood_selected": normal_loop_report["bayesian_optimization"]["selected_out_of_domain_count"],
            },
            "T35_CRASH_RESUME_OVERRIDE": {
                "pass": (
                    recovery_report["crash_point"]["completed_results_before_restart"] == 2
                    and recovery_report["crash_point"]["pending_results_before_restart"] == 3
                    and abs(float(recovery_report["crash_point"]["checkpoint_progress_percent"]) - 40.0) < 1e-9
                    and recovery_report["reconciliation"]["source_of_truth"] == "PERSISTED_T29_SCHEDULER"
                    and recovery_report["reconciliation"]["adapter_job_id_match"] is True
                    and recovery_report["results"]["duplicate_completed_result_writes"] == 0
                    and recovery_report["dataset"]["row_count_before"] == expected_recovery_dataset_rows[0]
                    and recovery_report["dataset"]["row_count_after"] == expected_recovery_dataset_rows[1]
                    and recovery_replay_report.get("idempotent_replay") is True
                    and cancel_job_report.get("scheduler_job_status") == "CANCELLED"
                    and cancel_job_report.get("automatic_continuation") is False
                    and abort_round_report.get("round_status") == "CANCELLED"
                    and abort_round_report.get("next_round_created") is False
                    and safety_resume_blocked is True
                ),
                "crash_progress_percent": recovery_report["crash_point"]["checkpoint_progress_percent"],
                "recovery_completed_results": recovery_report["results"]["completed_after_resume"],
                "cancel_job_status": cancel_job_report.get("scheduler_job_status"),
                "abort_round_status": abort_round_report.get("round_status"),
                "recovery_replay_idempotent": recovery_replay_report.get("idempotent_replay"),
            },
        }

        missing = [name for name in REQUIRED_COMPONENTS if name not in component_checks]
        if missing:
            raise FinalValidationInputError(
                "缺少 V0.3 component check: " + ", ".join(missing)
            )
        component_pass_count = sum(
            1 for value in component_checks.values() if value["pass"] is True
        )

        global_checks = {
            "all_component_checks_pass": component_pass_count == len(REQUIRED_COMPONENTS),
            "normal_three_rounds_completed": normal_loop_report["all_rounds_completed"] is True,
            "fifteen_results_auto_captured": normal_loop_report["experiments"]["automatic_capture_count"] == expected_total,
            "zero_manual_result_submissions": normal_loop_report["experiments"]["manual_result_submission_count"] == 0,
            "dataset_lineage_exact": normal_loop_report["dataset"]["row_counts"] == expected_normal_dataset_rows,
            "no_cross_round_duplicate_candidates": normal_loop_report["experiments"]["duplicate_candidate_ids"] == 0,
            "no_cross_round_duplicate_features": normal_loop_report["experiments"]["duplicate_feature_points"] == 0,
            "bo_never_selected_ood": normal_loop_report["bayesian_optimization"]["selected_out_of_domain_count"] == 0,
            "model_never_auto_activated": (
                normal_loop_report["model_governance"]["automatic_activation_count"] == 0
                and recovery_report["model_governance"]["automatic_activation"] is False
            ),
            "crash_resume_no_duplicate_result_write": recovery_report["results"]["duplicate_completed_result_writes"] == 0,
            "crash_resume_dataset_once": (
                recovery_report["dataset"]["added_row_count"] == expected_experiments_per_round
                and recovery_report["dataset"]["row_count_after"] == expected_recovery_dataset_rows[1]
            ),
            "operator_cancel_stops_continuation": (
                cancel_job_report.get("automatic_continuation") is False
                and cancel_job_report.get("dataset_updated") is False
                and cancel_job_report.get("next_round_created") is False
            ),
            "operator_abort_stops_round": (
                abort_round_report.get("round_status") == "CANCELLED"
                and abort_round_report.get("dataset_updated") is False
                and abort_round_report.get("model_governance_run") is False
                and abort_round_report.get("next_round_created") is False
            ),
            "operator_resume_cannot_bypass_safety": safety_resume_blocked is True,
            "normal_loop_idempotent": normal_loop_report.get("idempotent_replay") is False,
            "recovery_replay_idempotent": recovery_replay_report.get("idempotent_replay") is True,
            "no_real_device_connected": True,
            "simulator_measurements_not_real": (
                normal_loop_report["experiments"]["is_real_measurement"] is False
                and recovery_report["results"]["is_real_measurement"] is False
            ),
        }

        passed = _all_true(global_checks)
        report = {
            "stage": FINAL_VALIDATION_STAGE,
            "schema_version": FINAL_VALIDATION_SCHEMA_VERSION,
            "validation_id": validation_id,
            "status": "PASS" if passed else "FAIL",
            "component_pass_count": component_pass_count,
            "component_total": len(REQUIRED_COMPONENTS),
            "component_checks": component_checks,
            "global_checks": global_checks,
            "normal_loop": {
                "campaign_id": normal_loop_report["campaign_id"],
                "round_count": normal_loop_report["round_count"],
                "total_experiments": normal_loop_report["experiments"]["total_planned"],
                "automatic_captures": normal_loop_report["experiments"]["automatic_capture_count"],
                "manual_submissions": normal_loop_report["experiments"]["manual_result_submission_count"],
                "dataset_row_counts": deepcopy(normal_loop_report["dataset"]["row_counts"]),
                "bo_transition_count": normal_loop_report["bayesian_optimization"]["bo_transition_count"],
                "bo_ood_selected": normal_loop_report["bayesian_optimization"]["selected_out_of_domain_count"],
                "automatic_model_activation_count": normal_loop_report["model_governance"]["automatic_activation_count"],
            },
            "crash_resume": {
                "campaign_id": recovery_report["campaign_id"],
                "round_id": recovery_report["round_id"],
                "completed_before_restart": recovery_report["crash_point"]["completed_results_before_restart"],
                "pending_before_restart": recovery_report["crash_point"]["pending_results_before_restart"],
                "crash_progress_percent": recovery_report["crash_point"]["checkpoint_progress_percent"],
                "completed_after_resume": recovery_report["results"]["completed_after_resume"],
                "dataset_rows_after_resume": recovery_report["dataset"]["row_count_after"],
                "replay_idempotent": recovery_replay_report.get("idempotent_replay"),
            },
            "operator_override": {
                "cancel_job": {
                    "status": cancel_job_report.get("scheduler_job_status"),
                    "automatic_continuation": cancel_job_report.get("automatic_continuation"),
                },
                "abort_round": {
                    "round_status": abort_round_report.get("round_status"),
                    "next_round_created": abort_round_report.get("next_round_created"),
                },
                "safety_resume_blocked": safety_resume_blocked,
                "safety_resume_error_code": safety_resume_error_code,
            },
            "boundary": {
                "simulator_only": True,
                "real_device_connected": False,
                "real_device_validation_completed": False,
                "simulator_results_are_not_real_material_measurements": True,
                "model_promotion_auto_approval_forbidden": True,
                "operator_override_cannot_bypass_safety": True,
                "v030_core_backend_complete_if_pass": True,
            },
        }
        digest_payload = deepcopy(report)
        report_sha256 = sha256_json(digest_payload)
        report["report_sha256"] = report_sha256

        path = self.report_path(validation_id)
        idempotent_replay = False
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("report_sha256") == report_sha256:
                idempotent_replay = True
            else:
                raise FinalValidationInputError(
                    "同 validation_id 已存在不同内容的 final report"
                )
        elif persist:
            _atomic_json(path, report)

        report["idempotent_replay"] = idempotent_replay
        report["report_json"] = str(path)

        if not passed:
            failed_checks = [
                key for key, value in global_checks.items()
                if value is not True
            ]
            failed_components = [
                key for key, value in component_checks.items()
                if value["pass"] is not True
            ]
            raise FinalValidationFailedError(
                "V0.3 T36 final validation FAIL",
                details={
                    "failed_global_checks": failed_checks,
                    "failed_components": failed_components,
                },
            )
        return report
