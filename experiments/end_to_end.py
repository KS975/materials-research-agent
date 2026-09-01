from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from .campaign import CampaignStore
from .dataset_versioning import DatasetVersionStore

END_TO_END_STAGE = "V0.2-T26_end_to_end_closed_loop"
END_TO_END_SCHEMA_VERSION = 1


class EndToEndValidationError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def feature_key(features: dict[str, Any], columns: list[str], digits: int = 10) -> tuple[float, ...]:
    values = []
    for column in columns:
        try:
            value = float(features[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise EndToEndValidationError(f"缺少/非法特征: {column}") from exc
        if not math.isfinite(value):
            raise EndToEndValidationError(f"特征不是有限数值: {column}")
        values.append(round(value, digits))
    return tuple(values)


def verify_dataset_lineage_manifests(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    if not manifests:
        raise EndToEndValidationError("dataset manifests 不能为空")
    errors = []
    for index, manifest in enumerate(manifests):
        version = manifest.get("dataset_version")
        if index == 0:
            if manifest.get("parent_dataset_version") is not None:
                errors.append(f"{version} 应为 base dataset")
            continue
        parent = manifests[index - 1]
        if manifest.get("parent_dataset_version") != parent.get("dataset_version"):
            errors.append(
                f"{version} parent 不连续: "
                f"{manifest.get('parent_dataset_version')} != {parent.get('dataset_version')}"
            )
        if manifest.get("parent_sha256") != parent.get("sha256"):
            errors.append(f"{version} parent_sha256 与上一版本不一致")
    return {"valid": not errors, "errors": errors}


def ensure_monotonic_best(values: list[float], direction: str) -> bool:
    if direction not in {"maximize", "minimize"}:
        raise EndToEndValidationError("direction 必须是 maximize/minimize")
    if len(values) < 2:
        return True
    if direction == "maximize":
        return all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    return all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def summarize_experiment_integrity(
    campaign: dict[str, Any],
    *,
    feature_columns: list[str],
) -> dict[str, Any]:
    candidate_ids: list[str] = []
    feature_keys: list[tuple[float, ...]] = []
    terminal = training_eligible = completed = 0
    result_missing = 0
    ood_marked = 0

    for round_record in campaign.get("rounds") or []:
        for experiment in round_record.get("experiments") or []:
            cid = str(experiment.get("candidate_id") or "").strip()
            if cid:
                candidate_ids.append(cid)
            features = experiment.get("features") or {}
            feature_keys.append(feature_key(features, feature_columns))
            status = experiment.get("status")
            if status in {"COMPLETED", "FAILED", "INVALID", "NOT_TESTED"}:
                terminal += 1
            if status == "COMPLETED":
                completed += 1
            result = experiment.get("result")
            if not result:
                result_missing += 1
            elif result.get("training_eligible") is True:
                training_eligible += 1
            ad = (experiment.get("applicability_domain") or {}).get("status")
            if ad == "OUT_OF_DOMAIN":
                ood_marked += 1

    return {
        "experiment_count": len(candidate_ids),
        "terminal_count": terminal,
        "completed_count": completed,
        "training_eligible_count": training_eligible,
        "result_missing_count": result_missing,
        "duplicate_candidate_id_count": len(candidate_ids) - len(set(candidate_ids)),
        "duplicate_feature_point_count": len(feature_keys) - len(set(feature_keys)),
        "out_of_domain_marked_count": ood_marked,
    }


def summarize_model_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(x.get("decision") or "") for x in decisions]
    return {
        "count": len(labels),
        "decisions": labels,
        "blocked_count": sum(x == "BLOCKED" for x in labels),
        "promote_count": sum(x == "PROMOTE" for x in labels),
        "keep_incumbent_count": sum(x == "KEEP_INCUMBENT" for x in labels),
        "review_required_count": sum(x == "REVIEW_REQUIRED" for x in labels),
    }


def _dataset_best(path: Path, target_metric: str, direction: str) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    candidates = [target_metric, f"target::{target_metric}"]
    target_col = next((c for c in candidates if c in (reader.fieldnames or [])), None)
    if target_col is None:
        raise EndToEndValidationError(f"dataset 缺少目标列: {target_metric}")
    values=[]
    for row in rows:
        try: value=float(row[target_col])
        except (TypeError,ValueError,KeyError): continue
        if math.isfinite(value): values.append(value)
    if not values:
        raise EndToEndValidationError("dataset 没有有效目标值")
    return max(values) if direction=="maximize" else min(values)


class EndToEndAuditService:
    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.campaigns = CampaignStore(runtime_root)
        self.datasets = DatasetVersionStore(runtime_root)

    def report_path(self, campaign_id: str) -> Path:
        return self.runtime_root / "v020" / "end_to_end" / campaign_id / "end_to_end_report.json"

    def build_report(
        self,
        *,
        campaign_id: str,
        dataset_versions: list[str],
        target_metric: str,
        direction: str,
        evaluation_reports: list[dict[str, Any]],
        model_decisions: list[dict[str, Any]],
        checkpoint_reports: list[dict[str, Any]],
        bo_reports: list[dict[str, Any]],
        expected_rounds: int = 3,
        expected_experiments_per_round: int = 5,
        persist: bool = True,
        fixture: bool = False,
    ) -> dict[str, Any]:
        campaign = self.campaigns.load(campaign_id)
        project_id = int(campaign["project_id"])
        rounds = campaign.get("rounds") or []
        if not dataset_versions:
            raise EndToEndValidationError("dataset_versions 不能为空")

        manifests=[]; verifications=[]; best_curve=[]
        for version in dataset_versions:
            manifests.append(self.datasets.load_manifest(project_id,version))
            verifications.append(self.datasets.verify(project_id,version))
            best_curve.append(_dataset_best(
                self.datasets.dataset_path(project_id,version),
                target_metric,direction,
            ))

        lineage=verify_dataset_lineage_manifests(manifests)
        feature_columns=[
            c for c in manifests[-1].get("columns",[])
            if c.startswith("formula::") or c.startswith("process::")
        ]
        integrity=summarize_experiment_integrity(
            campaign,feature_columns=feature_columns
        )
        model_summary=summarize_model_decisions(model_decisions)

        checkpoint_summary={
            "count":len(checkpoint_reports),
            "completed_count":sum(
                (x.get("checkpoint") or x).get("status")=="COMPLETED"
                for x in checkpoint_reports
            ),
            "resume_count_total":sum(
                int((x.get("checkpoint") or x).get("resume_count",0))
                for x in checkpoint_reports
            ),
        }
        bo_summary={
            "count":len(bo_reports),
            "out_of_domain_selected_count":0,
            "next_experiment_count":0,
        }
        for report in bo_reports:
            selected=(report.get("next_experiments") or report.get("selected") or [])
            bo_summary["next_experiment_count"] += len(selected)
            for item in selected:
                ad=item.get("applicability_domain") or {}
                if ad.get("status")=="OUT_OF_DOMAIN":
                    bo_summary["out_of_domain_selected_count"] += 1

        checks={
            "campaign_completed":campaign.get("status")=="COMPLETED",
            "round_count_expected":len(rounds)==expected_rounds,
            "all_rounds_completed":all(r.get("status")=="COMPLETED" for r in rounds),
            "experiments_per_round_expected":all(
                len(r.get("experiments") or [])==expected_experiments_per_round
                for r in rounds
            ),
            "all_experiments_terminal":(
                integrity["terminal_count"]==expected_rounds*expected_experiments_per_round
            ),
            "all_experiments_have_results":integrity["result_missing_count"]==0,
            "no_duplicate_candidate_ids":integrity["duplicate_candidate_id_count"]==0,
            "no_duplicate_feature_points":integrity["duplicate_feature_point_count"]==0,
            "dataset_lineage_valid":lineage["valid"],
            "dataset_versions_verified":all(x.get("verified") for x in verifications),
            "dataset_row_count_monotonic":all(
                b["row_count"]>=a["row_count"]
                for a,b in zip(verifications,verifications[1:])
            ),
            "best_so_far_monotonic":ensure_monotonic_best(best_curve,direction),
            "evaluation_every_round":len(evaluation_reports)==expected_rounds,
            "no_blocked_model_decision":model_summary["blocked_count"]==0,
            "checkpoint_workflows_completed":(
                checkpoint_summary["count"]>=2
                and checkpoint_summary["completed_count"]==checkpoint_summary["count"]
            ),
            "bo_never_selected_ood":bo_summary["out_of_domain_selected_count"]==0,
        }
        decision="PASS" if all(checks.values()) else "FAIL"

        report={
            "stage":END_TO_END_STAGE,
            "schema_version":END_TO_END_SCHEMA_VERSION,
            "generated_at":utc_now_iso(),
            "decision":decision,
            "fixture":bool(fixture),
            "campaign_id":campaign_id,
            "project_id":project_id,
            "target_metric":target_metric,
            "direction":direction,
            "campaign_status":campaign.get("status"),
            "round_count":len(rounds),
            "rounds":[{
                "round_id":r.get("round_id"),
                "round_no":r.get("round_no"),
                "status":r.get("status"),
                "dataset_version":(r.get("plan") or {}).get("dataset_version"),
                "experiment_count":len(r.get("experiments") or []),
                "progress":deepcopy(r.get("progress") or {}),
            } for r in rounds],
            "datasets":{
                "versions":dataset_versions,
                "row_counts":[x["row_count"] for x in verifications],
                "sha256":[x["sha256"] for x in verifications],
                "lineage":lineage,
            },
            "best_so_far":{
                "values":best_curve,
                "initial":best_curve[0],
                "final":best_curve[-1],
                "net_improvement":(
                    best_curve[-1]-best_curve[0]
                    if direction=="maximize"
                    else best_curve[0]-best_curve[-1]
                ),
                "monotonic":ensure_monotonic_best(best_curve,direction),
            },
            "experiment_integrity":integrity,
            "evaluations": [{
                "round_id":x.get("round_id"),
                "mae":(x.get("aggregate") or {}).get("mae"),
                "rmse":(x.get("aggregate") or {}).get("rmse"),
                "r2":(x.get("aggregate") or {}).get("r2"),
                "overconfident_2sigma_miss_count":(
                    (x.get("uncertainty") or {}).get("overconfident_2sigma_miss_count")
                ),
            } for x in evaluation_reports],
            "model_governance":model_summary,
            "checkpoints":checkpoint_summary,
            "bayesian_optimization":bo_summary,
            "checks":checks,
            "safety":{
                "synthetic_fixture_measurements":bool(fixture),
                "measurement_ingestion_path":"V0.2-T20 ExperimentalResultService",
                "automatic_model_replacement":False,
                "note":(
                    "T26 fixture 的实测值由确定性 synthetic oracle 生成，仅用于工程验收；"
                    "生产环境必须由真实实验结果通过 T20 回流。"
                ),
            },
        }
        if persist:
            path=self.report_path(campaign_id)
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(tmp,path)
            report["report_json"]=str(path)
        return deepcopy(report)
