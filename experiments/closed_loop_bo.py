from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import CampaignConflictError, CampaignStore, find_round
from .dataset_versioning import DatasetVersionStore
from .model_promotion import ModelRegistry
from .results import ExperimentalResultService

try:
    from optimization.applicability import (
        ApplicabilityDomainCalibrator,
        ApplicabilityDomainError,
    )
    from optimization.bayesian_optimization import (
        BOConfig,
        BayesianOptimizationError,
        GaussianProcessBayesianOptimizer,
        feature_key_from_vector,
        filter_already_observed_candidate_indices,
    )
except ImportError as exc:  # pragma: no cover - environment error path
    raise RuntimeError(
        "T24 需要 V0.1.4 optimization 模块（T13/T18）。"
    ) from exc


CLOSED_LOOP_BO_STAGE = "V0.2-T24_closed_loop_bayesian_optimization"
CLOSED_LOOP_BO_SCHEMA_VERSION = 1


class ClosedLoopBOError(RuntimeError):
    pass


class ClosedLoopBOValidationError(ClosedLoopBOError):
    pass


class ClosedLoopBOConflictError(ClosedLoopBOError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ClosedLoopBOValidationError(f"{name} 必须是数值") from exc
    if not math.isfinite(number):
        raise ClosedLoopBOValidationError(f"{name} 必须是有限数值")
    return number


def _load_dataset(
    path: Path,
    *,
    target_metric: str,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        raise ClosedLoopBOValidationError("dataset CSV 缺少表头")

    target_col = next(
        (c for c in (target_metric, f"target::{target_metric}") if c in fields),
        None,
    )
    if target_col is None:
        raise ClosedLoopBOValidationError(
            f"dataset 缺少目标列: {target_metric}"
        )

    feature_columns = [
        c for c in fields
        if c.startswith("formula::") or c.startswith("process::")
    ]
    if not feature_columns:
        raise ClosedLoopBOValidationError(
            "dataset 没有 formula:: / process:: 数值特征"
        )

    X, y, candidate_ids, conditions = [], [], [], []
    for row in rows:
        try:
            vector = [_finite(row.get(c), c) for c in feature_columns]
            target = _finite(row.get(target_col), target_col)
        except ClosedLoopBOValidationError:
            continue
        cid = str(row.get("candidate_id") or "").strip()
        if not cid:
            continue
        X.append(vector)
        y.append(target)
        candidate_ids.append(cid)
        conditions.append(
            str(row.get("test_condition_signature") or "").strip()
        )

    if len(X) < 10:
        raise ClosedLoopBOValidationError(
            f"闭环 BO 至少需要 10 条完整历史实验，当前 {len(X)}"
        )

    condition_set = {x for x in conditions if x}
    if len(condition_set) != 1:
        raise ClosedLoopBOConflictError(
            f"测试条件签名必须唯一，实际数量={len(condition_set)}"
        )

    return {
        "X": np.asarray(X, dtype=float),
        "y": np.asarray(y, dtype=float),
        "candidate_ids": candidate_ids,
        "feature_columns": feature_columns,
        "target_column": target_col,
        "condition_signature": next(iter(condition_set)),
        "rows": len(X),
    }


def _load_candidate_pool(
    path: Path,
    *,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        raise ClosedLoopBOValidationError(
            f"candidate pool 不存在: {path}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    required = ["candidate_id", *feature_columns]
    missing = [c for c in required if c not in fields]
    if missing:
        raise ClosedLoopBOValidationError(
            f"candidate pool 缺少字段: {missing}"
        )

    result = []
    seen_ids = set()
    for row in rows:
        cid = str(row.get("candidate_id") or "").strip()
        if not cid:
            raise ClosedLoopBOValidationError(
                "candidate pool 存在空 candidate_id"
            )
        if cid in seen_ids:
            raise ClosedLoopBOValidationError(
                f"candidate pool candidate_id 重复: {cid}"
            )
        seen_ids.add(cid)

        hard_valid_text = str(row.get("hard_valid", "true")).strip().lower()
        hard_valid = hard_valid_text not in {"false", "0", "no"}
        features = {
            c: _finite(row.get(c), f"{cid}.{c}")
            for c in feature_columns
        }
        result.append({
            "candidate_id": cid,
            "features": features,
            "hard_valid": hard_valid,
            "soft_penalty": max(
                0.0,
                _finite(row.get("soft_penalty", 0.0), f"{cid}.soft_penalty"),
            ),
        })
    return result


def _all_campaign_candidate_ids(
    campaign: dict[str, Any],
    *,
    excluded_round_ids: set[str] | None = None,
) -> set[str]:
    excluded_round_ids = excluded_round_ids or set()
    ids: set[str] = set()
    for round_record in campaign.get("rounds", []):
        if round_record.get("round_id") in excluded_round_ids:
            continue
        for experiment in round_record.get("experiments") or []:
            cid = str(experiment.get("candidate_id") or "").strip()
            if cid:
                ids.add(cid)
    return ids


def _best(y: np.ndarray, direction: str) -> float:
    return (
        float(np.max(y))
        if direction == "maximize"
        else float(np.min(y))
    )


class ClosedLoopBOService:
    """Generate the next experiment round from completed feedback.

    T24 intentionally reuses the frozen T18 GP + acquisition + Kriging
    Believer implementation. It adds campaign/dataset lineage, completed
    experiment exclusion, AD filtering, and next-round registration.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.campaigns = CampaignStore(runtime_root)
        self.datasets = DatasetVersionStore(runtime_root)
        self.results = ExperimentalResultService(runtime_root)
        self.models = ModelRegistry(runtime_root)

    def _report_path(self, campaign_id: str, source_round_id: str) -> Path:
        safe_round = source_round_id.replace("/", "_").replace("\\", "_")
        return (
            self.runtime_root
            / "v020"
            / "closed_loop_bo"
            / campaign_id
            / safe_round
            / "closed_loop_bo_report.json"
        )

    def _load_existing_report(
        self,
        campaign_id: str,
        source_round_id: str,
    ) -> dict[str, Any] | None:
        path = self._report_path(campaign_id, source_round_id)
        if not path.exists():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        report["idempotent_replay"] = True
        report["report_json"] = str(path)
        return report

    def generate_next_round(
        self,
        *,
        campaign_id: str,
        source_round_id: str,
        latest_dataset_version: str,
        candidate_pool_csv: str | Path,
        target_metric: str,
        target_unit: str,
        gate: dict[str, Any],
        batch_size: int = 5,
        acquisition: str = "EI",
        direction: str = "maximize",
        xi: float = 0.01,
        kappa: float = 2.0,
        min_batch_distance: float = 0.20,
        soft_penalty_weight: float = 0.10,
        allow_borderline_for_exploration: bool = True,
        random_state: int = 42,
    ) -> dict[str, Any]:
        existing = self._load_existing_report(
            campaign_id, source_round_id
        )
        if existing is not None:
            return existing

        if gate.get("training_allowed") is not True:
            raise ClosedLoopBOConflictError(
                "Modeling Gate training_allowed=false，禁止闭环 BO"
            )
        if gate.get("official_model_allowed") is not True:
            raise ClosedLoopBOConflictError(
                "Modeling Gate official_model_allowed=false，禁止闭环 BO"
            )
        if batch_size <= 0:
            raise ClosedLoopBOValidationError("batch_size 必须 > 0")

        campaign = self.campaigns.load(campaign_id)
        source_round = find_round(campaign, source_round_id)
        if source_round.get("status") != "COMPLETED":
            raise ClosedLoopBOConflictError(
                "只有 COMPLETED Round 才能生成下一轮 BO"
            )

        # T25 recovery: normally the source round must be the latest round.
        # If a previous process already created exactly one PLANNED T24 next
        # round but died before the final report was committed, reuse it
        # instead of creating a third round.
        rounds = campaign.get("rounds") or []
        source_index = next(
            (i for i, r in enumerate(rounds) if r.get("round_id") == source_round_id),
            None,
        )
        if source_index is None:
            raise ClosedLoopBOConflictError("source Round 不存在")
        following_rounds = rounds[source_index + 1 :]
        if len(following_rounds) > 1:
            raise ClosedLoopBOConflictError(
                "source Round 后已经存在多个 Round，拒绝自动恢复"
            )

        recovery_next_round = None
        if following_rounds:
            candidate = following_rounds[0]
            plan = candidate.get("plan") or {}
            if (
                candidate.get("status") != "PLANNED"
                or plan.get("source") != "V0.2-T24_closed_loop_BO"
                or plan.get("dataset_version") != latest_dataset_version
            ):
                raise ClosedLoopBOConflictError(
                    "source Round 后存在非本次 T24 生成的 Round，拒绝自动恢复"
                )
            recovery_next_round = candidate
        elif not rounds or rounds[-1].get("round_id") != source_round_id:
            raise ClosedLoopBOConflictError(
                "只能从 Campaign 最新的 COMPLETED Round 生成下一轮"
            )

        project_id = int(campaign["project_id"])
        manifest = self.datasets.load_manifest(
            project_id, latest_dataset_version
        )
        self.datasets.verify(project_id, latest_dataset_version)
        source = manifest.get("source") or {}
        if (
            source.get("type") != "CAMPAIGN_ROUND_UPDATE"
            or source.get("campaign_id") != campaign_id
            or source.get("round_id") != source_round_id
        ):
            raise ClosedLoopBOConflictError(
                "latest dataset lineage 必须来自当前 source Round"
            )

        current = _load_dataset(
            self.datasets.dataset_path(project_id, latest_dataset_version),
            target_metric=target_metric,
        )
        feature_columns = current["feature_columns"]
        X_obs = current["X"]
        y_obs = current["y"]

        parent_version = manifest.get("parent_dataset_version")
        previous_best = None
        if parent_version:
            parent = _load_dataset(
                self.datasets.dataset_path(project_id, parent_version),
                target_metric=target_metric,
            )
            previous_best = _best(parent["y"], direction)
        current_best = _best(y_obs, direction)
        best_improvement = (
            None
            if previous_best is None
            else (
                current_best - previous_best
                if direction == "maximize"
                else previous_best - current_best
            )
        )

        raw_pool = _load_candidate_pool(
            Path(candidate_pool_csv),
            feature_columns=feature_columns,
        )

        hard_valid = [row for row in raw_pool if row["hard_valid"]]
        hard_invalid_excluded = len(raw_pool) - len(hard_valid)

        recovery_round_id = (
            recovery_next_round.get("round_id")
            if recovery_next_round is not None
            else None
        )
        used_ids = _all_campaign_candidate_ids(
            campaign,
            excluded_round_ids=(
                {recovery_round_id} if recovery_round_id else set()
            ),
        )
        id_filtered = [
            row for row in hard_valid
            if row["candidate_id"] not in used_ids
        ]
        used_candidate_id_filtered = len(hard_valid) - len(id_filtered)

        if not id_filtered:
            raise ClosedLoopBOConflictError(
                "candidate pool 在已做实验排除后为空"
            )

        X_pool = np.asarray(
            [
                [row["features"][c] for c in feature_columns]
                for row in id_filtered
            ],
            dtype=float,
        )
        keep_indices, duplicate_indices = (
            filter_already_observed_candidate_indices(X_pool, X_obs)
        )
        feature_duplicate_filtered = len(duplicate_indices)
        deduped = [id_filtered[i] for i in keep_indices]

        calibrator = ApplicabilityDomainCalibrator(
            feature_columns=feature_columns,
            X=X_obs,
            dropped_rows=0,
        )
        eligible = []
        ad_out_excluded = 0
        borderline_kept = 0
        borderline_excluded = 0
        for row in deduped:
            ad = calibrator.evaluate(row["features"])
            row = {**row, "applicability_domain": ad}
            if ad["status"] == "OUT_OF_DOMAIN":
                ad_out_excluded += 1
                continue
            if ad["status"] == "BORDERLINE":
                if allow_borderline_for_exploration:
                    borderline_kept += 1
                else:
                    borderline_excluded += 1
                    continue
            eligible.append(row)

        if len(eligible) < batch_size:
            raise ClosedLoopBOConflictError(
                f"AD 过滤后候选不足: {len(eligible)} < {batch_size}"
            )

        X_candidates = np.asarray(
            [
                [row["features"][c] for c in feature_columns]
                for row in eligible
            ],
            dtype=float,
        )
        candidate_ids = [row["candidate_id"] for row in eligible]
        penalties = np.asarray(
            [row["soft_penalty"] for row in eligible],
            dtype=float,
        )

        try:
            optimizer = GaussianProcessBayesianOptimizer(
                X_obs,
                y_obs,
                config=BOConfig(
                    acquisition=acquisition,
                    direction=direction,
                    xi=xi,
                    kappa=kappa,
                    batch_size=batch_size,
                    min_batch_distance=min_batch_distance,
                    random_state=random_state,
                ),
            )
            fit_summary = optimizer.fit_summary()
            proposal = optimizer.propose_batch(
                X_candidates,
                candidate_ids,
                candidate_penalties=penalties,
                penalty_weight=soft_penalty_weight,
            )
        except BayesianOptimizationError as exc:
            raise ClosedLoopBOError(str(exc)) from exc

        selected = []
        for selection in proposal["rounds"]:
            idx = int(selection["candidate_index"])
            source_candidate = eligible[idx]
            selected.append({
                "candidate_id": source_candidate["candidate_id"],
                "features": deepcopy(source_candidate["features"]),
                "soft_penalty": source_candidate["soft_penalty"],
                "applicability_domain": deepcopy(
                    source_candidate["applicability_domain"]
                ),
                "posterior_mean": selection["posterior_mean"],
                "posterior_std": selection["posterior_std"],
                "acquisition_value": selection["acquisition_value"],
                "adjusted_acquisition": selection["adjusted_acquisition"],
                "selection_round": selection["round"],
            })

        # Determine the currently active official model, if T23 registry exists.
        registry = self.models.load_registry(project_id, target_metric)
        active_model_version = registry.get("active_model_version")
        if not active_model_version:
            active_model_version = (
                source_round.get("plan", {})
                .get("model_versions", {})
                .get(target_metric)
                or "UNSPECIFIED"
            )

        next_plan = {
            "planned_experiment_count": batch_size,
            "dataset_version": latest_dataset_version,
            "model_versions": {
                target_metric: active_model_version,
            },
            "search_space_snapshot": deepcopy(
                source_round["plan"]["search_space_snapshot"]
            ),
            "constraints_snapshot": deepcopy(
                source_round["plan"]["constraints_snapshot"]
            ),
            "optimizer_config": {
                "engine": "GaussianProcess",
                "acquisition": acquisition,
                "batch_strategy": "kriging_believer",
                "batch_size": batch_size,
                "xi": xi,
                "kappa": kappa,
                "min_batch_distance": min_batch_distance,
                "soft_penalty_weight": soft_penalty_weight,
                "allow_borderline_for_exploration": (
                    allow_borderline_for_exploration
                ),
                "source_stage": CLOSED_LOOP_BO_STAGE,
            },
            "source": "V0.2-T24_closed_loop_BO",
            "notes": (
                f"Generated from {source_round_id} using "
                f"{latest_dataset_version}; GP uses latest measured dataset."
            ),
        }

        reused_existing_next_round = recovery_next_round is not None
        if recovery_next_round is None:
            try:
                next_round = self.campaigns.add_round(
                    campaign_id,
                    plan=next_plan,
                )
            except CampaignConflictError as exc:
                raise ClosedLoopBOConflictError(str(exc)) from exc
        else:
            next_round = deepcopy(recovery_next_round)
            existing_plan = next_round.get("plan") or {}
            if (
                int(existing_plan.get("planned_experiment_count", -1)) != batch_size
                or existing_plan.get("dataset_version") != latest_dataset_version
            ):
                raise ClosedLoopBOConflictError(
                    "恢复中的 next Round plan 与本次 BO 请求不一致"
                )

        planned_experiments = []
        for item in selected:
            planned_experiments.append({
                "candidate_id": item["candidate_id"],
                "required_metrics": [target_metric],
                "expected_test_condition_signature": current[
                    "condition_signature"
                ],
                "units": {target_metric: target_unit},
                "features": item["features"],
                "prediction_snapshot": {
                    target_metric: {
                        "value": item["posterior_mean"],
                        "posterior_std": item["posterior_std"],
                        "acquisition": acquisition,
                        "acquisition_value": item["acquisition_value"],
                        "adjusted_acquisition": item[
                            "adjusted_acquisition"
                        ],
                        "source": CLOSED_LOOP_BO_STAGE,
                    }
                },
            })

        existing_experiments = next_round.get("experiments")
        recovered_missing_registration = False
        recovered_existing_registration = False
        if existing_experiments is None:
            self.results.register_planned_experiments(
                campaign_id,
                round_id=next_round["round_id"],
                experiments=planned_experiments,
            )
            if reused_existing_next_round:
                recovered_missing_registration = True
        else:
            expected_ids = [x["candidate_id"] for x in planned_experiments]
            actual_ids = [x.get("candidate_id") for x in existing_experiments]
            if actual_ids != expected_ids:
                raise ClosedLoopBOConflictError(
                    "恢复中的 next Round experiments 与重新计算的 BO 结果不一致"
                )
            recovered_existing_registration = reused_existing_next_round

        report = {
            "stage": CLOSED_LOOP_BO_STAGE,
            "schema_version": CLOSED_LOOP_BO_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "idempotent_replay": False,
            "recovery": {
                "reused_existing_next_round": reused_existing_next_round,
                "recovered_missing_experiment_registration": (
                    recovered_missing_registration
                ),
                "recovered_existing_experiment_registration": (
                    recovered_existing_registration
                ),
            },
            "project_id": project_id,
            "campaign_id": campaign_id,
            "target_metric": target_metric,
            "source_round_id": source_round_id,
            "next_round_id": next_round["round_id"],
            "latest_dataset_version": latest_dataset_version,
            "parent_dataset_version": parent_version,
            "gate": {
                "decision": gate.get("decision"),
                "training_allowed": gate.get("training_allowed"),
                "official_model_allowed": gate.get(
                    "official_model_allowed"
                ),
            },
            "best_so_far": {
                "direction": direction,
                "previous_dataset_best": previous_best,
                "current_dataset_best": current_best,
                "improvement_from_feedback_round": best_improvement,
            },
            "observations": {
                "rows": current["rows"],
                "feature_columns": feature_columns,
                "test_condition_signature": current[
                    "condition_signature"
                ],
            },
            "candidate_flow": {
                "candidate_pool_rows": len(raw_pool),
                "hard_invalid_excluded": hard_invalid_excluded,
                "used_candidate_id_filtered": used_candidate_id_filtered,
                "already_observed_feature_filtered": (
                    feature_duplicate_filtered
                ),
                "out_of_domain_excluded": ad_out_excluded,
                "borderline_kept_for_exploration": borderline_kept,
                "borderline_excluded": borderline_excluded,
                "eligible_for_bo": len(eligible),
            },
            "gaussian_process": fit_summary,
            "bayesian_optimization": {
                "acquisition": acquisition,
                "batch_strategy": proposal["batch_strategy"],
                "batch_size": batch_size,
                "selection_score": proposal["selection_score"],
                "soft_penalty_weight": soft_penalty_weight,
            },
            "model_governance": {
                "active_official_model_version": active_model_version,
                "note": (
                    "T24 GP surrogate is refit from latest measured dataset; "
                    "it does not auto-promote a T23 challenger model."
                ),
            },
            "next_experiments": selected,
            "safety": {
                "source_round_must_be_completed": True,
                "latest_dataset_lineage_verified": True,
                "already_observed_feature_points_removed": True,
                "campaign_candidate_ids_removed": True,
                "out_of_domain_removed": True,
                "future_measurements_fabricated": False,
                "next_round_starts_as_planned": True,
            },
        }

        path = self._report_path(campaign_id, source_round_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        report["report_json"] = str(path)
        return deepcopy(report)
