from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

CAMPAIGN_STAGE = "V0.2-T19_experiment_campaign"
SCHEMA_VERSION = 1
CAMPAIGN_STATUSES = {"ACTIVE", "COMPLETED", "CANCELLED"}
ROUND_STATUSES = {"PLANNED", "RUNNING", "PARTIALLY_COMPLETED", "COMPLETED", "CANCELLED"}
ROUND_TRANSITIONS = {
    "PLANNED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"PARTIALLY_COMPLETED", "COMPLETED", "CANCELLED"},
    "PARTIALLY_COMPLETED": {"RUNNING", "COMPLETED", "CANCELLED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
}

class CampaignError(RuntimeError):
    pass
class CampaignNotFoundError(CampaignError):
    pass
class CampaignConflictError(CampaignError):
    pass
class CampaignValidationError(CampaignError):
    pass

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _nonempty_string(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise CampaignValidationError(f"{name} 不能为空")
    return result

def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignValidationError(f"{name} 必须是正整数")
    return int(value)

def _safe_id(value: str, name: str) -> str:
    result = _nonempty_string(value, name)
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", result):
        raise CampaignValidationError(f"{name} 只能包含字母、数字、点、下划线和短横线")
    return result

def _deep_jsonable(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CampaignValidationError(f"{name} 必须是可序列化 JSON 数据") from exc

def validate_round_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise CampaignValidationError("round plan 必须是 JSON object")
    count = _positive_int(plan.get("planned_experiment_count"), "planned_experiment_count")
    dataset_version = _nonempty_string(plan.get("dataset_version"), "dataset_version")
    model_versions = plan.get("model_versions")
    if not isinstance(model_versions, dict) or not model_versions:
        raise CampaignValidationError("model_versions 必须是非空 JSON object")
    model_versions = {
        _nonempty_string(metric, "model_versions key"): _nonempty_string(version, f"model_versions[{metric}]")
        for metric, version in model_versions.items()
    }
    search_space_snapshot = plan.get("search_space_snapshot")
    if not isinstance(search_space_snapshot, dict) or not search_space_snapshot:
        raise CampaignValidationError("search_space_snapshot 必须是非空 JSON object")
    constraints_snapshot = plan.get("constraints_snapshot")
    if not isinstance(constraints_snapshot, dict):
        raise CampaignValidationError("constraints_snapshot 必须是 JSON object")
    optimizer_config = plan.get("optimizer_config")
    if not isinstance(optimizer_config, dict):
        raise CampaignValidationError("optimizer_config 必须是 JSON object")
    return {
        "planned_experiment_count": count,
        "dataset_version": dataset_version,
        "model_versions": _deep_jsonable(model_versions, "model_versions"),
        "search_space_snapshot": _deep_jsonable(search_space_snapshot, "search_space_snapshot"),
        "constraints_snapshot": _deep_jsonable(constraints_snapshot, "constraints_snapshot"),
        "optimizer_config": _deep_jsonable(optimizer_config, "optimizer_config"),
        "source": str(plan.get("source") or "manual").strip() or "manual",
        "notes": str(plan.get("notes") or "").strip(),
    }

def make_campaign(*, campaign_id: str, project_id: int, name: str, target_metrics: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    campaign_id = _safe_id(campaign_id, "campaign_id")
    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise CampaignValidationError("project_id 必须是整数")
    name = _nonempty_string(name, "name")
    if not isinstance(target_metrics, list) or not target_metrics:
        raise CampaignValidationError("target_metrics 必须是非空 list")
    clean_metrics, seen = [], set()
    for metric in target_metrics:
        metric = _nonempty_string(metric, "target_metric")
        if metric not in seen:
            seen.add(metric); clean_metrics.append(metric)
    now = utc_now_iso()
    return {
        "stage": CAMPAIGN_STAGE,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "project_id": project_id,
        "name": name,
        "target_metrics": clean_metrics,
        "status": "ACTIVE",
        "created_at": now,
        "updated_at": now,
        "current_round_no": 0,
        "rounds": [],
        "metadata": _deep_jsonable(metadata or {}, "metadata"),
        "events": [{"event_id": 1, "event_type": "CAMPAIGN_CREATED", "timestamp": now, "payload": {"project_id": project_id, "target_metrics": clean_metrics}}],
    }

def _append_event(campaign: dict[str, Any], *, event_type: str, payload: dict[str, Any]) -> None:
    events = campaign.setdefault("events", [])
    events.append({"event_id": len(events)+1, "event_type": event_type, "timestamp": utc_now_iso(), "payload": _deep_jsonable(payload, "event payload")})

def _touch(campaign: dict[str, Any]) -> None:
    campaign["updated_at"] = utc_now_iso()

def add_round(campaign: dict[str, Any], *, plan: dict[str, Any]) -> dict[str, Any]:
    if campaign.get("status") != "ACTIVE":
        raise CampaignConflictError("只有 ACTIVE campaign 可以创建新 round")
    rounds = campaign.setdefault("rounds", [])
    if rounds and rounds[-1].get("status") != "COMPLETED":
        raise CampaignConflictError("上一轮实验未 COMPLETED，不能创建下一轮")
    validated = validate_round_plan(plan)
    round_no = len(rounds)+1
    campaign_id = _safe_id(campaign.get("campaign_id"), "campaign_id")
    round_id = f"{campaign_id}-R{round_no:03d}"
    now = utc_now_iso()
    record = {
        "round_id": round_id,
        "round_no": round_no,
        "status": "PLANNED",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "cancelled_at": None,
        "plan": validated,
        "progress": {"planned": validated["planned_experiment_count"], "completed": 0, "failed": 0, "invalid": 0, "not_tested": validated["planned_experiment_count"]},
    }
    rounds.append(record)
    campaign["current_round_no"] = round_no
    _touch(campaign)
    _append_event(campaign, event_type="ROUND_CREATED", payload={"round_id": round_id, "round_no": round_no, "dataset_version": validated["dataset_version"], "model_versions": validated["model_versions"], "planned_experiment_count": validated["planned_experiment_count"]})
    return deepcopy(record)

def find_round(campaign: dict[str, Any], round_id: str) -> dict[str, Any]:
    round_id = _nonempty_string(round_id, "round_id")
    for record in campaign.get("rounds", []):
        if record.get("round_id") == round_id:
            return record
    raise CampaignNotFoundError(f"round 不存在: {round_id}")

def transition_round(campaign: dict[str, Any], *, round_id: str, new_status: str, reason: str | None = None) -> dict[str, Any]:
    if campaign.get("status") != "ACTIVE":
        raise CampaignConflictError("只有 ACTIVE campaign 可以变更 round 状态")
    record = find_round(campaign, round_id)
    current = str(record.get("status") or "").strip()
    new_status = str(new_status or "").strip().upper()
    if new_status not in ROUND_STATUSES:
        raise CampaignValidationError(f"未知 round status: {new_status}")
    if new_status == current:
        return deepcopy(record)
    if new_status not in ROUND_TRANSITIONS.get(current, set()):
        raise CampaignConflictError(f"非法 round 状态迁移: {current} -> {new_status}")
    # V0.2-T20: if explicit experiments are registered, every one must
    # have a terminal outcome before the round can be closed.
    if new_status == "COMPLETED" and record.get("experiments") is not None:
        pending = [
            item.get("candidate_id")
            for item in (record.get("experiments") or [])
            if item.get("status") == "PLANNED"
        ]
        if pending:
            raise CampaignConflictError(
                "仍有未提交实验结果，不能将 Round 标记为 COMPLETED: "
                + ", ".join(str(x) for x in pending[:10])
            )

    now = utc_now_iso(); record["status"] = new_status
    if new_status == "RUNNING" and record.get("started_at") is None: record["started_at"] = now
    if new_status == "COMPLETED":
        record["completed_at"] = now
        # Legacy T19 rounds had no experiment registry.
        if record.get("experiments") is None:
            record["progress"]["not_tested"] = 0
    if new_status == "CANCELLED": record["cancelled_at"] = now
    _touch(campaign)
    _append_event(campaign, event_type="ROUND_STATUS_CHANGED", payload={"round_id": round_id, "from_status": current, "to_status": new_status, "reason": str(reason or "").strip()})
    return deepcopy(record)

def complete_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    if campaign.get("status") != "ACTIVE":
        raise CampaignConflictError("campaign 当前不是 ACTIVE")
    rounds = campaign.get("rounds", [])
    if not rounds:
        raise CampaignConflictError("没有 round 的 campaign 不能完成")
    if rounds[-1].get("status") != "COMPLETED":
        raise CampaignConflictError("最新 round 未 COMPLETED，不能完成 campaign")
    campaign["status"] = "COMPLETED"; _touch(campaign)
    _append_event(campaign, event_type="CAMPAIGN_COMPLETED", payload={"round_count": len(rounds)})
    return deepcopy(campaign)

class CampaignStore:
    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
    def campaign_dir(self, campaign_id: str) -> Path:
        campaign_id = _safe_id(campaign_id, "campaign_id")
        return self.runtime_root / "v020" / "campaigns" / campaign_id
    def campaign_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / "campaign.json"
    def create(self, *, campaign_id: str, project_id: int, name: str, target_metrics: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.campaign_path(campaign_id)
        if path.exists():
            raise CampaignConflictError(f"campaign 已存在: {campaign_id}")
        campaign = make_campaign(campaign_id=campaign_id, project_id=project_id, name=name, target_metrics=target_metrics, metadata=metadata)
        self.save(campaign); return deepcopy(campaign)
    def load(self, campaign_id: str) -> dict[str, Any]:
        path = self.campaign_path(campaign_id)
        if not path.exists():
            raise CampaignNotFoundError(f"campaign 不存在: {campaign_id}")
        with path.open("r", encoding="utf-8") as f: campaign = json.load(f)
        if campaign.get("stage") != CAMPAIGN_STAGE:
            raise CampaignValidationError(f"campaign stage 非法: {campaign.get('stage')}")
        return campaign
    def save(self, campaign: dict[str, Any]) -> Path:
        campaign_id = _safe_id(campaign.get("campaign_id"), "campaign_id")
        directory = self.campaign_dir(campaign_id); directory.mkdir(parents=True, exist_ok=True)
        path = directory / "campaign.json"; temp_path = directory / "campaign.json.tmp"
        payload = json.dumps(campaign, ensure_ascii=False, indent=2)
        with temp_path.open("w", encoding="utf-8") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(temp_path, path); return path
    def add_round(self, campaign_id: str, *, plan: dict[str, Any]) -> dict[str, Any]:
        campaign = self.load(campaign_id); record = add_round(campaign, plan=plan); self.save(campaign); return record
    def transition_round(self, campaign_id: str, *, round_id: str, new_status: str, reason: str | None = None) -> dict[str, Any]:
        campaign = self.load(campaign_id); record = transition_round(campaign, round_id=round_id, new_status=new_status, reason=reason); self.save(campaign); return record
    def complete_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.load(campaign_id); result = complete_campaign(campaign); self.save(campaign); return result
