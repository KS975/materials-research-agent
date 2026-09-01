from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .device import (
    DEVICE_STAGE,
    SimulatorDeviceAdapter,
)
from .evaluation import (
    PredictionEvaluationService,
    PredictionEvaluationValidationError,
)
from .protocol import sha256_json
from .results import (
    ExperimentalResultService,
    find_experiment,
)
from .campaign import CampaignStore


RESULT_CAPTURE_STAGE = "V0.3-T32_automatic_result_capture"
RESULT_CAPTURE_SCHEMA_VERSION = 1
SIMULATOR_RESULT_ORIGIN = "SIMULATOR_FIXTURE"


class AutomaticResultCaptureError(RuntimeError):
    code = "AUTOMATIC_RESULT_CAPTURE_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = deepcopy(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


class ResultCaptureValidationError(AutomaticResultCaptureError):
    code = "RESULT_CAPTURE_VALIDATION_ERROR"


class ResultCaptureConflictError(AutomaticResultCaptureError):
    code = "RESULT_CAPTURE_CONFLICT"


class ResultCaptureIntegrityError(AutomaticResultCaptureError):
    code = "RESULT_CAPTURE_INTEGRITY_ERROR"


class ResultNotReadyError(AutomaticResultCaptureError):
    code = "RESULT_NOT_READY"


class SafetyStopActiveError(AutomaticResultCaptureError):
    code = "SAFETY_STOP_ACTIVE"


def _text(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise ResultCaptureValidationError(f"{name} 不能为空")
    return out


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultCaptureValidationError(f"{name} 必须是有限数值")
    out = float(value)
    if not math.isfinite(out):
        raise ResultCaptureValidationError(f"{name} 必须是有限数值")
    return out


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_.\-]+", "_", str(value or "").strip())
    text = text.strip("._")
    if not text:
        raise ResultCaptureValidationError("路径标识不能为空")
    return text[:120]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _result_digest_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in result.items()
        if key not in {"result_id", "content_sha256"}
    }


def verify_device_result_integrity(result: Any) -> bool:
    if not isinstance(result, dict):
        raise ResultCaptureValidationError("device result 必须是 JSON object")
    supplied = _text(result.get("content_sha256"), "content_sha256")
    expected = sha256_json(_result_digest_payload(result))
    if supplied != expected:
        raise ResultCaptureIntegrityError(
            "device result content_sha256 校验失败",
            details={"expected": expected, "actual": supplied},
        )
    result_id = _text(result.get("result_id"), "result_id")
    expected_id = "simres_" + expected[:20]
    if result_id != expected_id:
        raise ResultCaptureIntegrityError(
            "device result_id 与内容 hash 不一致",
            details={"expected": expected_id, "actual": result_id},
        )
    return True


def normalize_device_result_for_t20(
    result: dict[str, Any],
    *,
    experiment: dict[str, Any],
    protocol_id: str,
    device_id: str,
) -> dict[str, Any]:
    verify_device_result_integrity(result)

    if result.get("stage") != DEVICE_STAGE:
        raise ResultCaptureValidationError(
            f"device result.stage 必须是 {DEVICE_STAGE!r}"
        )
    if str(result.get("status") or "").upper() != "COMPLETED":
        raise ResultNotReadyError("T32 只自动回流 COMPLETED device result")

    # T32 acceptance still has no verified real instrument driver. A source
    # claiming to be real is rejected rather than silently trusted.
    if result.get("measurement_origin") != SIMULATOR_RESULT_ORIGIN:
        raise ResultCaptureValidationError(
            "T32 acceptance 只允许 SIMULATOR_FIXTURE result；真实设备来源尚未启用"
        )
    if result.get("synthetic") is not True:
        raise ResultCaptureValidationError(
            "SIMULATOR_FIXTURE 必须显式 synthetic=true"
        )
    if result.get("is_real_measurement") is not False:
        raise ResultCaptureValidationError(
            "SIMULATOR_FIXTURE 必须显式 is_real_measurement=false"
        )

    candidate_id = _text(result.get("candidate_id"), "candidate_id")
    expected_candidate = _text(experiment.get("candidate_id"), "experiment.candidate_id")
    if candidate_id != expected_candidate:
        raise ResultCaptureConflictError(
            "device result candidate_id 与 Round experiment 不一致",
            details={"expected": expected_candidate, "actual": candidate_id},
        )

    result_protocol_id = _text(result.get("protocol_id"), "protocol_id")
    if result_protocol_id != _text(protocol_id, "protocol_id"):
        raise ResultCaptureConflictError(
            "device result protocol_id 与 T27 protocol 不一致",
            details={"expected": protocol_id, "actual": result_protocol_id},
        )

    result_device_id = _text(result.get("device_id"), "device_id")
    if result_device_id != _text(device_id, "device_id"):
        raise ResultCaptureConflictError(
            "device result device_id 与 adapter 不一致",
            details={"expected": device_id, "actual": result_device_id},
        )

    required_metrics = list(experiment.get("required_metrics") or [])
    if not required_metrics:
        raise ResultCaptureValidationError("experiment.required_metrics 不能为空")
    expected_units = dict(experiment.get("units") or {})
    expected_condition = _text(
        experiment.get("expected_test_condition_signature"),
        "expected_test_condition_signature",
    )

    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ResultCaptureValidationError("device result.outputs 必须是非空 list")

    by_metric: dict[str, dict[str, Any]] = {}
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ResultCaptureValidationError(
                f"outputs[{index}] 必须是 object"
            )
        metric = _text(output.get("metric"), f"outputs[{index}].metric")
        if metric in by_metric:
            raise ResultCaptureValidationError(
                f"device result metric 重复: {metric}"
            )
        by_metric[metric] = output

    missing = [metric for metric in required_metrics if metric not in by_metric]
    if missing:
        raise ResultCaptureValidationError(
            "device result 缺少必需指标: " + ", ".join(missing)
        )
    extras = [metric for metric in by_metric if metric not in required_metrics]
    if extras:
        raise ResultCaptureValidationError(
            "device result 包含 Round 未声明指标: " + ", ".join(extras)
        )

    measurements: dict[str, float] = {}
    units: dict[str, str] = {}
    for metric in required_metrics:
        output = by_metric[metric]
        measurements[metric] = _finite(
            output.get("value"), f"outputs[{metric}].value"
        )
        supplied_unit = _text(output.get("unit"), f"outputs[{metric}].unit")
        expected_unit = str(expected_units.get(metric) or "").strip()
        if expected_unit and supplied_unit != expected_unit:
            raise ResultCaptureConflictError(
                f"{metric} 单位不一致: expected={expected_unit}, got={supplied_unit}"
            )
        units[metric] = supplied_unit

        supplied_condition = _text(
            output.get("condition_signature"),
            f"outputs[{metric}].condition_signature",
        )
        if supplied_condition != expected_condition:
            raise ResultCaptureConflictError(
                f"{metric} 测试条件不一致: expected={expected_condition}, got={supplied_condition}"
            )

    return {
        "candidate_id": expected_candidate,
        "status": "COMPLETED",
        "test_condition_signature": expected_condition,
        "measurements": measurements,
        "units": units,
        "notes": (
            "AUTO_CAPTURE T32; source=SIMULATOR_FIXTURE; synthetic=true; "
            "is_real_measurement=false; result_id=" + str(result["result_id"])
        ),
    }


class AutomaticResultCaptureService:
    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.store = CampaignStore(self.runtime_root)
        self.results = ExperimentalResultService(str(self.runtime_root))
        self.evaluations = PredictionEvaluationService(self.runtime_root)

    def receipt_path(
        self,
        campaign_id: str,
        *,
        round_id: str,
        candidate_id: str,
    ) -> Path:
        return (
            self.runtime_root
            / "v030"
            / "result_capture"
            / _safe_component(campaign_id)
            / _safe_component(round_id)
            / (_safe_component(candidate_id) + ".json")
        )

    def _safety_guard(self, safety_interlock: Any | None) -> None:
        if safety_interlock is None:
            return
        try:
            snapshot = safety_interlock.snapshot()
        except Exception as exc:
            raise ResultCaptureValidationError(
                "无法读取 T31 safety interlock 状态",
                details={"reason": str(exc)},
            ) from exc
        if snapshot.get("state") != "SAFE":
            raise SafetyStopActiveError(
                "T31 SAFETY_STOP 未清除，禁止自动回流设备结果",
                details={
                    "state": snapshot.get("state"),
                    "current_trip": snapshot.get("current_trip"),
                },
            )

    def capture(
        self,
        campaign_id: str,
        *,
        round_id: str,
        adapter: SimulatorDeviceAdapter,
        protocol: dict[str, Any],
        safety_interlock: Any | None = None,
        evaluate: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(adapter, SimulatorDeviceAdapter):
            raise ResultCaptureValidationError(
                "T32 acceptance 只允许 SimulatorDeviceAdapter"
            )
        self._safety_guard(safety_interlock)

        status = adapter.status()
        if status.get("state") != "COMPLETED" or status.get("result_ready") is not True:
            raise ResultNotReadyError(
                "device result 尚未 ready；需要 adapter state=COMPLETED"
            )

        campaign = self.store.load(campaign_id)
        experiment = find_experiment(
            campaign,
            round_id=round_id,
            candidate_id=_text(protocol.get("candidate_id"), "protocol.candidate_id"),
        )
        if str(protocol.get("protocol_id") or "") != str(status.get("prepared_protocol_id") or ""):
            raise ResultCaptureConflictError(
                "adapter prepared protocol 与传入 protocol 不一致"
            )

        result = adapter.read_result()
        payload = normalize_device_result_for_t20(
            result,
            experiment=experiment,
            protocol_id=str(protocol["protocol_id"]),
            device_id=str(adapter.device_id),
        )

        receipt_path = self.receipt_path(
            campaign_id,
            round_id=round_id,
            candidate_id=experiment["candidate_id"],
        )
        existing = None
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing.get("source_result_id") != result["result_id"]:
                raise ResultCaptureConflictError(
                    "该 candidate 已有不同 source_result_id 的 capture receipt"
                )
            if existing.get("source_result_sha256") != result["content_sha256"]:
                raise ResultCaptureConflictError(
                    "该 candidate 已有不同 source_result_sha256 的 capture receipt"
                )

        ingested = self.results.ingest(
            campaign_id,
            round_id=round_id,
            payload=payload,
        )

        evaluation_reports: dict[str, Any] = {}
        evaluation_skipped: dict[str, str] = {}
        if evaluate:
            for metric in experiment.get("required_metrics") or []:
                try:
                    report = self.evaluations.evaluate(
                        campaign_id,
                        round_id=round_id,
                        metric=metric,
                        persist=True,
                    )
                    evaluation_reports[metric] = {
                        "evaluated": report["counts"]["evaluated"],
                        "mae": report["aggregate"]["mae"],
                        "rmse": report["aggregate"]["rmse"],
                        "r2": report["aggregate"]["r2"],
                        "report_json": report.get("report_json"),
                    }
                except PredictionEvaluationValidationError as exc:
                    evaluation_skipped[metric] = str(exc)

        receipt = {
            "stage": RESULT_CAPTURE_STAGE,
            "schema_version": RESULT_CAPTURE_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "round_id": round_id,
            "candidate_id": experiment["candidate_id"],
            "protocol_id": protocol["protocol_id"],
            "device_id": adapter.device_id,
            "source_result_id": result["result_id"],
            "source_result_sha256": result["content_sha256"],
            "measurement_origin": result["measurement_origin"],
            "synthetic": True,
            "is_real_measurement": False,
            "normalized_t20_payload": payload,
            "t20_training_eligible": bool(
                ingested["experiment"].get("result", {}).get("training_eligible")
            ),
            "t20_idempotent_replay": bool(ingested["idempotent_replay"]),
            "round_progress": deepcopy(ingested["round_progress"]),
            "evaluation_reports": evaluation_reports,
            "evaluation_skipped": evaluation_skipped,
            "fixture_warning": (
                "该 measurement 来自 deterministic Simulator fixture，"
                "只用于 V0.3 工程验收，不是真实材料实验结果。"
            ),
        }

        if existing is None:
            _atomic_json(receipt_path, receipt)
            receipt_idempotent = False
        else:
            # Reconcile actual T20 state rather than trusting the receipt alone.
            receipt_idempotent = True

        return {
            "idempotent_replay": bool(receipt_idempotent and ingested["idempotent_replay"]),
            "receipt": deepcopy(existing if existing is not None else receipt),
            "t20": ingested,
            "evaluation_reports": evaluation_reports,
            "evaluation_skipped": evaluation_skipped,
            "receipt_json": str(receipt_path),
        }
