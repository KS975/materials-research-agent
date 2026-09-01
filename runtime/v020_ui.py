from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from experiments import (
    CampaignConflictError,
    CampaignStore,
    ExperimentalResultService,
    ModelPromotionConflictError,
    ModelRegistry,
    PredictionEvaluationError,
    PredictionEvaluationService,
    ResumableClosedLoopWorkflow,
    find_round,
)


class V020UIError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V020UIError(f"无法读取 V0.2 运行报告: {path}: {exc}") from exc
    return data if isinstance(data, dict) else None


def _round_no(round_id: str) -> int:
    m = re.search(r"-R(\d+)$", str(round_id or ""))
    return int(m.group(1)) if m else -1


def _version_no(version: str) -> int:
    m = re.search(r"(\d+)$", str(version or ""))
    return int(m.group(1)) if m else -1


def _next_version(version: str, prefix: str) -> str:
    number = _version_no(version)
    if number < 0:
        raise V020UIError(f"无法从版本号推导下一版本: {version}")
    return f"{prefix}{number + 1:03d}"


def _campaign_files(runtime_root: Path) -> list[Path]:
    root = runtime_root / "v020" / "campaigns"
    if not root.exists():
        return []
    return sorted(root.glob("*/campaign.json"))


def latest_campaign_id_for_project(runtime_root: str | Path, project_id: int) -> str:
    root = Path(runtime_root)
    matches = []
    for path in _campaign_files(root):
        data = _read_json(path)
        if data and int(data.get("project_id", -1)) == int(project_id):
            matches.append((str(data.get("updated_at") or ""), path.stat().st_mtime, data))
    if not matches:
        raise V020UIError(f"Project {project_id} 尚无 V0.2 Campaign 运行记录")
    matches.sort(key=lambda x: (x[0], x[1]))
    return str(matches[-1][2]["campaign_id"])


def _dataset_manifests(runtime_root: Path, project_id: int) -> list[dict[str, Any]]:
    root = runtime_root / "v020" / "datasets" / f"project_{project_id}"
    items = []
    if root.exists():
        for path in root.glob("*/manifest.json"):
            data = _read_json(path)
            if data:
                items.append(data)
    items.sort(key=lambda x: _version_no(str(x.get("dataset_version") or "")))
    return items


def _latest_evaluation(runtime_root: Path, campaign_id: str) -> dict[str, Any] | None:
    root = runtime_root / "v020" / "evaluations" / campaign_id
    candidates = []
    if root.exists():
        for path in root.glob("*/prediction_vs_measurement_*.json"):
            data = _read_json(path)
            if data:
                candidates.append((_round_no(str(data.get("round_id") or "")), path.stat().st_mtime, data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _latest_checkpoint(runtime_root: Path, campaign_id: str) -> dict[str, Any] | None:
    root = runtime_root / "v020" / "checkpoints" / campaign_id
    candidates = []
    if root.exists():
        for path in root.glob("*.json"):
            data = _read_json(path)
            if data:
                candidates.append((_round_no(str(data.get("context", {}).get("source_round_id") or "")), path.stat().st_mtime, data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _latest_closed_loop_report(runtime_root: Path, campaign_id: str) -> dict[str, Any] | None:
    root = runtime_root / "v020" / "closed_loop_bo" / campaign_id
    candidates = []
    if root.exists():
        for path in root.glob("*/closed_loop_bo_report.json"):
            data = _read_json(path)
            if data:
                candidates.append((_round_no(str(data.get("source_round_id") or "")), path.stat().st_mtime, data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _model_view(runtime_root: Path, project_id: int, target_metric: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    target_dir = runtime_root / "v020" / "models" / f"project_{project_id}" / target_metric
    registry = _read_json(target_dir / "registry.json")
    report = _read_json(
        runtime_root / "v020" / "model_promotion" / f"project_{project_id}" / target_metric / "promotion_report.json"
    )
    return registry, report


def _end_to_end(runtime_root: Path, campaign_id: str) -> dict[str, Any] | None:
    return _read_json(runtime_root / "v020" / "end_to_end" / campaign_id / "end_to_end_report.json")


def _fixture_input_dir(runtime_root: Path, campaign_id: str) -> Path | None:
    fixtures = runtime_root / "v020" / "fixtures"
    if not fixtures.exists():
        return None
    for create_path in fixtures.glob("*/campaign_create.json"):
        data = _read_json(create_path)
        if data and data.get("campaign_id") == campaign_id:
            return create_path.parent
    return None


def resolve_advance_inputs(runtime_root: str | Path, campaign: dict[str, Any]) -> dict[str, Any]:
    root = Path(runtime_root)
    project_id = int(campaign["project_id"])
    metric = str((campaign.get("target_metrics") or [""])[0])
    ui_dir = root / "v020" / "ui_inputs" / f"project_{project_id}"
    pool = ui_dir / "candidate_pool.csv"
    gate_options = [ui_dir / f"gate_{metric}.json", ui_dir / "gate.json"]
    gate = next((p for p in gate_options if p.exists()), None)
    source = "ui_inputs"

    if not pool.exists() or gate is None:
        fixture_dir = _fixture_input_dir(root, str(campaign["campaign_id"]))
        if fixture_dir is not None:
            fixture_pool = fixture_dir / "candidate_pool.csv"
            fixture_gate = fixture_dir / "gate_pass.json"
            if fixture_pool.exists() and fixture_gate.exists():
                pool, gate, source = fixture_pool, fixture_gate, "fixture_fallback"

    return {
        "ready": pool.exists() and gate is not None and gate.exists(),
        "candidate_pool_csv": str(pool) if pool.exists() else None,
        "gate_json": str(gate) if gate is not None and gate.exists() else None,
        "source": source if pool.exists() and gate is not None and gate.exists() else None,
    }


def _experiment_card(item: dict[str, Any]) -> dict[str, Any]:
    prediction_snapshot = item.get("prediction_snapshot") or {}
    prediction = {}
    for metric, value in prediction_snapshot.items():
        if isinstance(value, dict):
            prediction[metric] = {
                "value": value.get("value"),
                "posterior_std": value.get("posterior_std"),
                "acquisition_value": value.get("acquisition_value"),
                "adjusted_acquisition": value.get("adjusted_acquisition"),
            }
    return {
        "candidate_id": item.get("candidate_id"),
        "status": item.get("status"),
        "required_metrics": deepcopy(item.get("required_metrics") or []),
        "expected_test_condition_signature": item.get("expected_test_condition_signature"),
        "units": deepcopy(item.get("units") or {}),
        "features": deepcopy(item.get("features") or {}),
        "prediction": prediction,
        "result": deepcopy(item.get("result")),
    }


def build_campaign_overview(
    runtime_root: str | Path,
    *,
    campaign_id: str | None = None,
    project_id: int | None = None,
) -> dict[str, Any]:
    root = Path(runtime_root)
    if not campaign_id:
        if project_id is None:
            raise V020UIError("campaign_id / project_id 至少提供一个")
        campaign_id = latest_campaign_id_for_project(root, int(project_id))

    store = CampaignStore(root)
    try:
        campaign = store.load(campaign_id)
    except Exception as exc:
        raise V020UIError(str(exc)) from exc

    actual_project_id = int(campaign["project_id"])
    if project_id is not None and int(project_id) != actual_project_id:
        raise V020UIError("campaign 与 project_id 不匹配")

    target_metrics = list(campaign.get("target_metrics") or [])
    primary_metric = str(target_metrics[0]) if target_metrics else ""
    rounds = list(campaign.get("rounds") or [])
    latest_round = rounds[-1] if rounds else None
    datasets = _dataset_manifests(root, actual_project_id)
    registry, promotion = _model_view(root, actual_project_id, primary_metric) if primary_metric else (None, None)
    evaluation = _latest_evaluation(root, campaign_id)
    checkpoint = _latest_checkpoint(root, campaign_id)
    bo_report = _latest_closed_loop_report(root, campaign_id)
    e2e = _end_to_end(root, campaign_id)
    advance_inputs = resolve_advance_inputs(root, campaign)

    round_cards = []
    for r in rounds:
        progress = deepcopy(r.get("progress") or {})
        experiments = list(r.get("experiments") or [])
        pending = sum(1 for x in experiments if x.get("status") == "PLANNED")
        round_cards.append({
            "round_id": r.get("round_id"),
            "round_no": r.get("round_no"),
            "status": r.get("status"),
            "dataset_version": (r.get("plan") or {}).get("dataset_version"),
            "model_versions": deepcopy((r.get("plan") or {}).get("model_versions") or {}),
            "planned_experiments": (r.get("plan") or {}).get("planned_experiment_count"),
            "progress": progress,
            "pending": pending,
            "source": (r.get("plan") or {}).get("source"),
        })

    latest_round_view = None
    if latest_round is not None:
        experiments = list(latest_round.get("experiments") or [])
        pending_experiments = [_experiment_card(x) for x in experiments if x.get("status") == "PLANNED"]
        latest_round_view = {
            "round_id": latest_round.get("round_id"),
            "round_no": latest_round.get("round_no"),
            "status": latest_round.get("status"),
            "dataset_version": (latest_round.get("plan") or {}).get("dataset_version"),
            "progress": deepcopy(latest_round.get("progress") or {}),
            "experiment_count": len(experiments),
            "experiments": [_experiment_card(x) for x in experiments],
            "pending_experiments": pending_experiments,
            "can_start": latest_round.get("status") == "PLANNED",
            "can_close_round": (
                latest_round.get("status") in {"RUNNING", "PARTIALLY_COMPLETED"}
                and bool(experiments)
                and not pending_experiments
            ),
            "can_advance": (
                latest_round.get("status") == "COMPLETED"
                and campaign.get("status") == "ACTIVE"
            ),
        }

    latest_dataset = datasets[-1] if datasets else None
    active_model = registry.get("active_model_version") if registry else None
    promotion_decision = promotion.get("decision") if promotion else None
    latest_round_status = latest_round.get("status") if latest_round else "NO_ROUND"
    pending_count = len(latest_round_view.get("pending_experiments") or []) if latest_round_view else 0

    if campaign.get("status") == "COMPLETED":
        answer = (
            f"Campaign {campaign_id} 已完成，共 {len(rounds)} 轮；"
            f"最新数据集 {latest_dataset.get('dataset_version') if latest_dataset else '-'}，"
            f"{latest_dataset.get('row_count') if latest_dataset else '-'} 条数据。"
        )
    else:
        answer = (
            f"Campaign {campaign_id} 当前 Round {latest_round.get('round_no') if latest_round else '-'} "
            f"状态为 {latest_round_status}，待处理实验 {pending_count} 组。"
        )

    return {
        "kind": "v020_feedback_loop",
        "status": campaign.get("status"),
        "answer": answer,
        "campaign": {
            "campaign_id": campaign_id,
            "project_id": actual_project_id,
            "name": campaign.get("name"),
            "status": campaign.get("status"),
            "target_metrics": target_metrics,
            "current_round_no": campaign.get("current_round_no"),
            "round_count": len(rounds),
            "created_at": campaign.get("created_at"),
            "updated_at": campaign.get("updated_at"),
        },
        "rounds": round_cards,
        "latest_round": latest_round_view,
        "datasets": [
            {
                "dataset_version": x.get("dataset_version"),
                "parent_dataset_version": x.get("parent_dataset_version"),
                "row_count": x.get("row_count"),
                "added_row_count": x.get("added_row_count"),
                "sha256": x.get("sha256"),
                "source": deepcopy(x.get("source")),
            }
            for x in datasets
        ],
        "latest_dataset": (
            {
                "dataset_version": latest_dataset.get("dataset_version"),
                "row_count": latest_dataset.get("row_count"),
                "sha256": latest_dataset.get("sha256"),
            }
            if latest_dataset else None
        ),
        "evaluation": deepcopy(evaluation),
        "model_registry": deepcopy(registry),
        "model_promotion": deepcopy(promotion),
        "checkpoint": deepcopy(checkpoint),
        "closed_loop_bo": deepcopy(bo_report),
        "end_to_end": deepcopy(e2e),
        "summary": {
            "active_model_version": active_model,
            "promotion_decision": promotion_decision,
            "checkpoint_status": checkpoint.get("status") if checkpoint else None,
            "end_to_end_decision": e2e.get("decision") if e2e else None,
        },
        "advance_inputs": advance_inputs,
        "safety": {
            "result_ingestion_uses_t20": True,
            "failed_invalid_not_tested_excluded_from_training": True,
            "dataset_versions_immutable": True,
            "automatic_model_replacement": False,
            "advance_requires_explicit_user_action": True,
            "synthetic_fixture_warning": bool(e2e and e2e.get("fixture")),
        },
    }


def start_round_for_ui(runtime_root: str | Path, *, campaign_id: str, round_id: str) -> dict[str, Any]:
    store = CampaignStore(runtime_root)
    try:
        store.transition_round(campaign_id, round_id=round_id, new_status="RUNNING", reason="V0.2 UI explicit start")
    except Exception as exc:
        raise V020UIError(str(exc)) from exc
    return build_campaign_overview(runtime_root, campaign_id=campaign_id)


def submit_result_for_ui(
    runtime_root: str | Path,
    *,
    campaign_id: str,
    round_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    service = ExperimentalResultService(runtime_root)
    try:
        result = service.ingest(campaign_id, round_id=round_id, payload=payload)
    except Exception as exc:
        raise V020UIError(str(exc)) from exc
    overview = build_campaign_overview(runtime_root, campaign_id=campaign_id)
    overview["last_result_ingestion"] = result
    return overview


def close_round_for_ui(runtime_root: str | Path, *, campaign_id: str, round_id: str) -> dict[str, Any]:
    store = CampaignStore(runtime_root)
    service = ExperimentalResultService(runtime_root)
    try:
        summary = service.summary(campaign_id, round_id=round_id)
        if not summary.get("can_close_round"):
            raise V020UIError("当前 Round 仍有未处理实验，不能关闭")
        store.transition_round(campaign_id, round_id=round_id, new_status="COMPLETED", reason="V0.2 UI explicit close")
        campaign = store.load(campaign_id)
        evaluator = PredictionEvaluationService(runtime_root)
        for metric in campaign.get("target_metrics") or []:
            try:
                evaluator.evaluate(campaign_id, round_id=round_id, metric=metric, persist=True)
            except PredictionEvaluationError:
                # A round can validly lack prediction_snapshot (e.g. imported plan).
                pass
    except V020UIError:
        raise
    except Exception as exc:
        raise V020UIError(str(exc)) from exc
    return build_campaign_overview(runtime_root, campaign_id=campaign_id)


def advance_campaign_for_ui(runtime_root: str | Path, *, campaign_id: str) -> dict[str, Any]:
    root = Path(runtime_root)
    store = CampaignStore(root)
    try:
        campaign = store.load(campaign_id)
        if campaign.get("status") != "ACTIVE":
            raise V020UIError("只有 ACTIVE Campaign 可以生成下一轮")
        rounds = list(campaign.get("rounds") or [])
        if not rounds or rounds[-1].get("status") != "COMPLETED":
            raise V020UIError("最新 Round 必须 COMPLETED 才能生成下一轮")
        source_round = rounds[-1]
        project_id = int(campaign["project_id"])
        target_metrics = list(campaign.get("target_metrics") or [])
        if len(target_metrics) != 1:
            raise V020UIError("V0.2-T24 UI advance 当前仅支持单目标 Campaign")
        metric = str(target_metrics[0])
        parent_dataset_version = str((source_round.get("plan") or {}).get("dataset_version") or "")
        child_dataset_version = _next_version(parent_dataset_version, "dataset_v")
        model_version = str((source_round.get("plan") or {}).get("model_versions", {}).get(metric) or "model_v001")
        # Challenger version follows the child dataset generation, not the
        # currently active model. The active model may intentionally remain
        # model_v001 across several REVIEW_REQUIRED rounds.
        challenger_model_version = child_dataset_version.replace("dataset_v", "model_v", 1)
        inputs = resolve_advance_inputs(root, campaign)
        if not inputs["ready"]:
            raise V020UIError(
                "缺少闭环 UI 输入。请准备 .runtime/v020/ui_inputs/"
                f"project_{project_id}/candidate_pool.csv 与 gate.json。"
            )
        gate = _read_json(Path(inputs["gate_json"]))
        if not gate:
            raise V020UIError("gate.json 无法读取")
        first_experiment = next(iter(source_round.get("experiments") or []), None)
        target_unit = ""
        if first_experiment:
            target_unit = str((first_experiment.get("units") or {}).get(metric) or "")
        if not target_unit:
            raise V020UIError(f"无法确定 {metric} 的单位")

        workflow = ResumableClosedLoopWorkflow(root)
        result = workflow.resume(
            campaign_id=campaign_id,
            source_round_id=str(source_round["round_id"]),
            parent_dataset_version=parent_dataset_version,
            child_dataset_version=child_dataset_version,
            candidate_pool_csv=inputs["candidate_pool_csv"],
            target_metric=metric,
            target_unit=target_unit,
            gate=gate,
            incumbent_model_version=model_version,
            challenger_model_version=challenger_model_version,
            batch_size=5,
            acquisition="EI",
            direction="maximize",
            random_state=42 + int(source_round.get("round_no") or 1),
        )
    except V020UIError:
        raise
    except Exception as exc:
        raise V020UIError(str(exc)) from exc

    overview = build_campaign_overview(root, campaign_id=campaign_id)
    overview["last_advance"] = result
    return overview


def approve_model_for_ui(
    runtime_root: str | Path,
    *,
    campaign_id: str,
    approved_by: str,
) -> dict[str, Any]:
    root = Path(runtime_root)
    store = CampaignStore(root)
    try:
        campaign = store.load(campaign_id)
        project_id = int(campaign["project_id"])
        target_metrics = list(campaign.get("target_metrics") or [])
        if len(target_metrics) != 1:
            raise V020UIError("当前只支持单目标模型晋级批准")
        metric = str(target_metrics[0])
        report_path = root / "v020" / "model_promotion" / f"project_{project_id}" / metric / "promotion_report.json"
        report = _read_json(report_path)
        if not report:
            raise V020UIError("尚无模型晋级报告")
        challenger = str(report.get("challenger_model_version") or "")
        registry = ModelRegistry(root)
        registry.approve_promotion(
            project_id=project_id,
            target_metric=metric,
            challenger_model_version=challenger,
            promotion_report=report,
            approved_by=approved_by,
        )
    except ModelPromotionConflictError as exc:
        raise V020UIError(str(exc)) from exc
    except V020UIError:
        raise
    except Exception as exc:
        raise V020UIError(str(exc)) from exc
    return build_campaign_overview(root, campaign_id=campaign_id)
