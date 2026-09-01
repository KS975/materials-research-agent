from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, pstdev
from typing import Any

from .dataset_versioning import DatasetVersionStore

MODEL_STAGE = "V0.2-T23_model_retraining_promotion"
MODEL_SCHEMA_VERSION = 1

PROMOTION_DECISIONS = {
    "PROMOTE",
    "KEEP_INCUMBENT",
    "REVIEW_REQUIRED",
    "BLOCKED",
}


class ModelPromotionError(RuntimeError):
    pass


class ModelPromotionValidationError(ModelPromotionError):
    pass


class ModelPromotionConflictError(ModelPromotionError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_name(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ModelPromotionValidationError(f"{name} 不能为空")
    return re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", text)


def _finite_float(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelPromotionValidationError(f"{name} 必须是数值") from exc
    if not math.isfinite(value):
        raise ModelPromotionValidationError(f"{name} 必须是有限数值")
    return value


def _load_dataset(path: Path, target_metric: str):
    try:
        import numpy as np
    except ImportError as exc:
        raise ModelPromotionError("T23 需要 numpy/scikit-learn") from exc

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    target_candidates = [target_metric, f"target::{target_metric}"]
    target_col = next((c for c in target_candidates if c in fields), None)
    if target_col is None:
        raise ModelPromotionValidationError(
            f"dataset 缺少目标列: {target_metric}"
        )

    feature_cols = [
        c for c in fields
        if c.startswith("formula::") or c.startswith("process::")
    ]
    if not feature_cols:
        raise ModelPromotionValidationError(
            "dataset 没有 formula:: / process:: 特征列"
        )
    if "candidate_id" not in fields:
        raise ModelPromotionValidationError("dataset 缺少 candidate_id")

    X, y, ids, conditions = [], [], [], []
    for row in rows:
        try:
            features = [_finite_float(row.get(c), c) for c in feature_cols]
            target = _finite_float(row.get(target_col), target_col)
        except ModelPromotionValidationError:
            continue
        cid = str(row.get("candidate_id") or "").strip()
        if not cid:
            continue
        X.append(features)
        y.append(target)
        ids.append(cid)
        conditions.append(str(row.get("test_condition_signature") or "").strip())

    if len(X) < 20:
        raise ModelPromotionValidationError(
            f"可用训练数据过少: {len(X)} < 20"
        )

    return {
        "X": np.asarray(X, dtype=float),
        "y": np.asarray(y, dtype=float),
        "candidate_ids": ids,
        "conditions": conditions,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "usable_rows": len(X),
    }


def _make_model(model_family: str, random_state: int):
    try:
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
    except ImportError as exc:
        raise ModelPromotionError(
            "T23 需要 scikit-learn"
        ) from exc

    if model_family == "ExtraTreesRegressor":
        return ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_family == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_family == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
    raise ModelPromotionValidationError(
        f"不支持 model_family: {model_family}"
    )


def _metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def _cross_validate(
    X,
    y,
    *,
    model_family: str,
    folds: int,
    random_state: int,
) -> dict[str, Any]:
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    if len(y) < folds * 2:
        raise ModelPromotionValidationError(
            f"数据不足以执行 {folds}-fold CV"
        )

    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    prototype = _make_model(model_family, random_state)
    fold_metrics = []
    for fold_no, (train_idx, test_idx) in enumerate(
        splitter.split(X), start=1
    ):
        model = clone(prototype)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        item = _metrics(y[test_idx], pred)
        item["fold"] = fold_no
        fold_metrics.append(item)

    summary = {}
    for key in ("r2", "mae", "rmse"):
        values = [x[key] for x in fold_metrics]
        summary[key] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return {"folds": fold_metrics, "summary": summary}


def decide_promotion(
    *,
    gate: dict[str, Any],
    incumbent_holdout: dict[str, float],
    challenger_holdout: dict[str, float],
    challenger_cv: dict[str, Any],
    policy: dict[str, float] | None = None,
) -> dict[str, Any]:
    policy = {
        "promote_min_rmse_improvement_fraction": 0.05,
        "promote_min_mae_improvement_fraction": 0.05,
        "max_allowed_r2_drop": 0.02,
        "min_challenger_cv_r2": 0.50,
        "max_challenger_cv_r2_std": 0.35,
        "keep_rmse_degradation_fraction": 0.05,
        "keep_r2_drop": 0.05,
        **(policy or {}),
    }

    reasons, warnings = [], []

    if gate.get("training_allowed") is not True:
        return {
            "decision": "BLOCKED",
            "reasons": ["Modeling Gate: training_allowed=false"],
            "warnings": [],
            "policy": policy,
        }
    if gate.get("official_model_allowed") is not True:
        return {
            "decision": "BLOCKED",
            "reasons": ["Modeling Gate: official_model_allowed=false"],
            "warnings": [],
            "policy": policy,
        }

    inc_rmse = incumbent_holdout["rmse"]
    ch_rmse = challenger_holdout["rmse"]
    inc_mae = incumbent_holdout["mae"]
    ch_mae = challenger_holdout["mae"]
    inc_r2 = incumbent_holdout["r2"]
    ch_r2 = challenger_holdout["r2"]

    rmse_improvement = (inc_rmse - ch_rmse) / max(abs(inc_rmse), 1e-12)
    mae_improvement = (inc_mae - ch_mae) / max(abs(inc_mae), 1e-12)
    r2_delta = ch_r2 - inc_r2

    cv_r2_mean = challenger_cv["summary"]["r2"]["mean"]
    cv_r2_std = challenger_cv["summary"]["r2"]["std"]

    if cv_r2_mean < policy["min_challenger_cv_r2"]:
        reasons.append(
            f"challenger CV R² 过低: {cv_r2_mean:.6f}"
        )
        return {
            "decision": "KEEP_INCUMBENT",
            "reasons": reasons,
            "warnings": warnings,
            "policy": policy,
        }

    if cv_r2_std > policy["max_challenger_cv_r2_std"]:
        warnings.append(
            f"challenger CV R² 波动较大: std={cv_r2_std:.6f}"
        )

    promote_quality = (
        rmse_improvement
        >= policy["promote_min_rmse_improvement_fraction"]
        or mae_improvement
        >= policy["promote_min_mae_improvement_fraction"]
    )
    no_material_r2_harm = r2_delta >= -policy["max_allowed_r2_drop"]
    stable_enough = cv_r2_std <= policy["max_challenger_cv_r2_std"]

    if promote_quality and no_material_r2_harm and stable_enough:
        reasons.append(
            "challenger 在共同 holdout 上显著改善，且 CV 达到晋级门槛"
        )
        decision = "PROMOTE"
    elif (
        ch_rmse
        > inc_rmse * (1.0 + policy["keep_rmse_degradation_fraction"])
        or ch_r2 < inc_r2 - policy["keep_r2_drop"]
    ):
        reasons.append(
            "challenger 在共同 holdout 上出现实质退化"
        )
        decision = "KEEP_INCUMBENT"
    else:
        reasons.append(
            "challenger 改善不足以自动建议晋级，需要人工复核"
        )
        decision = "REVIEW_REQUIRED"

    return {
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "policy": policy,
        "deltas": {
            "rmse_improvement_fraction": rmse_improvement,
            "mae_improvement_fraction": mae_improvement,
            "r2_delta": r2_delta,
        },
    }


class ModelRegistry:
    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)

    def target_dir(self, project_id: int, target_metric: str) -> Path:
        return (
            self.runtime_root
            / "v020"
            / "models"
            / f"project_{int(project_id)}"
            / _safe_name(target_metric, "target_metric")
        )

    def registry_path(self, project_id: int, target_metric: str) -> Path:
        return self.target_dir(project_id, target_metric) / "registry.json"

    def model_dir(
        self,
        project_id: int,
        target_metric: str,
        model_version: str,
    ) -> Path:
        return self.target_dir(project_id, target_metric) / _safe_name(
            model_version, "model_version"
        )

    def load_registry(self, project_id: int, target_metric: str) -> dict[str, Any]:
        path = self.registry_path(project_id, target_metric)
        if not path.exists():
            return {
                "project_id": int(project_id),
                "target_metric": target_metric,
                "active_model_version": None,
                "models": {},
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_registry(self, project_id: int, target_metric: str, data: dict[str, Any]) -> None:
        path = self.registry_path(project_id, target_metric)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def save_model(
        self,
        *,
        project_id: int,
        target_metric: str,
        model_version: str,
        model,
        metadata: dict[str, Any],
        make_active: bool = False,
    ) -> dict[str, Any]:
        import joblib

        directory = self.model_dir(
            project_id, target_metric, model_version
        )
        if directory.exists():
            raise ModelPromotionConflictError(
                f"model_version 已存在: {model_version}"
            )
        directory.mkdir(parents=True)
        model_path = directory / "model.joblib"
        meta_path = directory / "metadata.json"
        joblib.dump(model, model_path)
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        registry = self.load_registry(project_id, target_metric)
        registry["models"][model_version] = {
            "status": "ACTIVE" if make_active else "CANDIDATE",
            "model_path": str(model_path),
            "metadata_path": str(meta_path),
            "dataset_version": metadata.get("dataset_version"),
            "created_at": metadata.get("created_at"),
        }
        if make_active:
            old = registry.get("active_model_version")
            if old and old in registry["models"]:
                registry["models"][old]["status"] = "RETIRED"
            registry["active_model_version"] = model_version
        self._save_registry(project_id, target_metric, registry)
        return deepcopy(registry)

    def approve_promotion(
        self,
        *,
        project_id: int,
        target_metric: str,
        challenger_model_version: str,
        promotion_report: dict[str, Any],
        approved_by: str,
    ) -> dict[str, Any]:
        if promotion_report.get("decision") != "PROMOTE":
            raise ModelPromotionConflictError(
                "只有 PROMOTE 决策允许人工批准晋级"
            )
        approved_by = str(approved_by or "").strip()
        if not approved_by:
            raise ModelPromotionValidationError("approved_by 不能为空")

        registry = self.load_registry(project_id, target_metric)
        if challenger_model_version not in registry["models"]:
            raise ModelPromotionConflictError(
                "challenger model 不在 registry 中"
            )
        old = registry.get("active_model_version")
        if old and old in registry["models"]:
            registry["models"][old]["status"] = "RETIRED"
        registry["active_model_version"] = challenger_model_version
        registry["models"][challenger_model_version]["status"] = "ACTIVE"
        registry["last_human_approval"] = {
            "approved_by": approved_by,
            "approved_at": utc_now_iso(),
            "promotion_report_decision": "PROMOTE",
        }
        self._save_registry(project_id, target_metric, registry)
        return deepcopy(registry)


class ModelPromotionService:
    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)
        self.datasets = DatasetVersionStore(runtime_root)
        self.registry = ModelRegistry(runtime_root)

    def compare_and_register(
        self,
        *,
        project_id: int,
        target_metric: str,
        parent_dataset_version: str,
        child_dataset_version: str,
        incumbent_model_version: str = "model_v001",
        challenger_model_version: str = "model_v002",
        model_family: str = "ExtraTreesRegressor",
        gate: dict[str, Any],
        folds: int = 5,
        random_state: int = 42,
        holdout_fraction: float = 0.25,
    ) -> dict[str, Any]:
        from sklearn.model_selection import train_test_split

        parent_manifest = self.datasets.load_manifest(
            project_id, parent_dataset_version
        )
        child_manifest = self.datasets.load_manifest(
            project_id, child_dataset_version
        )
        self.datasets.verify(project_id, parent_dataset_version)
        self.datasets.verify(project_id, child_dataset_version)

        if child_manifest.get("parent_dataset_version") != parent_dataset_version:
            raise ModelPromotionConflictError(
                "child dataset lineage 与 parent_dataset_version 不一致"
            )

        parent = _load_dataset(
            self.datasets.dataset_path(project_id, parent_dataset_version),
            target_metric,
        )
        child = _load_dataset(
            self.datasets.dataset_path(project_id, child_dataset_version),
            target_metric,
        )

        if parent["feature_columns"] != child["feature_columns"]:
            raise ModelPromotionConflictError(
                "parent/child feature schema 不一致"
            )

        condition_set = {
            c for c in child["conditions"] if c
        }
        if len(condition_set) != 1:
            blocked = {
                "decision": "BLOCKED",
                "reasons": [
                    f"测试条件签名数量必须为 1，实际为 {len(condition_set)}"
                ],
                "warnings": [],
                "policy": {},
            }
            return self._persist_report(
                project_id=project_id,
                target_metric=target_metric,
                parent_dataset_version=parent_dataset_version,
                child_dataset_version=child_dataset_version,
                incumbent_model_version=incumbent_model_version,
                challenger_model_version=challenger_model_version,
                model_family=model_family,
                gate=gate,
                decision=blocked,
                details={"condition_signatures": sorted(condition_set)},
            )

        if gate.get("training_allowed") is not True or gate.get("official_model_allowed") is not True:
            blocked = decide_promotion(
                gate=gate,
                incumbent_holdout={"r2": 0, "mae": 0, "rmse": 1},
                challenger_holdout={"r2": 0, "mae": 0, "rmse": 1},
                challenger_cv={"summary": {"r2": {"mean": 0, "std": 0}}},
            )
            return self._persist_report(
                project_id=project_id,
                target_metric=target_metric,
                parent_dataset_version=parent_dataset_version,
                child_dataset_version=child_dataset_version,
                incumbent_model_version=incumbent_model_version,
                challenger_model_version=challenger_model_version,
                model_family=model_family,
                gate=gate,
                decision=blocked,
                details={},
            )

        indices = list(range(parent["usable_rows"]))
        train_idx, hold_idx = train_test_split(
            indices,
            test_size=holdout_fraction,
            random_state=random_state,
            shuffle=True,
        )
        holdout_ids = {
            parent["candidate_ids"][i] for i in hold_idx
        }

        X_parent_train = parent["X"][train_idx]
        y_parent_train = parent["y"][train_idx]
        X_hold = parent["X"][hold_idx]
        y_hold = parent["y"][hold_idx]

        child_train_idx = [
            i for i, cid in enumerate(child["candidate_ids"])
            if cid not in holdout_ids
        ]
        X_child_train = child["X"][child_train_idx]
        y_child_train = child["y"][child_train_idx]

        incumbent_cv = _cross_validate(
            X_parent_train,
            y_parent_train,
            model_family=model_family,
            folds=folds,
            random_state=random_state,
        )
        challenger_cv = _cross_validate(
            X_child_train,
            y_child_train,
            model_family=model_family,
            folds=folds,
            random_state=random_state,
        )

        incumbent = _make_model(model_family, random_state)
        incumbent.fit(X_parent_train, y_parent_train)
        incumbent_holdout = _metrics(
            y_hold, incumbent.predict(X_hold)
        )

        challenger = _make_model(model_family, random_state)
        challenger.fit(X_child_train, y_child_train)
        challenger_holdout = _metrics(
            y_hold, challenger.predict(X_hold)
        )

        decision = decide_promotion(
            gate=gate,
            incumbent_holdout=incumbent_holdout,
            challenger_holdout=challenger_holdout,
            challenger_cv=challenger_cv,
        )

        created_at = utc_now_iso()
        common_meta = {
            "stage": MODEL_STAGE,
            "schema_version": MODEL_SCHEMA_VERSION,
            "project_id": project_id,
            "target_metric": target_metric,
            "model_family": model_family,
            "feature_columns": parent["feature_columns"],
            "test_condition_signature": next(iter(condition_set)),
            "created_at": created_at,
        }

        registry_before = self.registry.load_registry(
            project_id, target_metric
        )
        if registry_before.get("active_model_version") is None:
            self.registry.save_model(
                project_id=project_id,
                target_metric=target_metric,
                model_version=incumbent_model_version,
                model=incumbent,
                metadata={
                    **common_meta,
                    "role": "INCUMBENT",
                    "dataset_version": parent_dataset_version,
                    "cv": incumbent_cv,
                    "holdout": incumbent_holdout,
                },
                make_active=True,
            )

        registry_mid = self.registry.load_registry(
            project_id, target_metric
        )
        if challenger_model_version not in registry_mid["models"]:
            self.registry.save_model(
                project_id=project_id,
                target_metric=target_metric,
                model_version=challenger_model_version,
                model=challenger,
                metadata={
                    **common_meta,
                    "role": "CHALLENGER",
                    "dataset_version": child_dataset_version,
                    "cv": challenger_cv,
                    "holdout": challenger_holdout,
                },
                make_active=False,
            )

        registry_after = self.registry.load_registry(
            project_id, target_metric
        )

        details = {
            "dataset": {
                "parent_rows": parent["usable_rows"],
                "child_rows": child["usable_rows"],
                "parent_train_rows": len(train_idx),
                "challenger_train_rows": len(child_train_idx),
                "common_holdout_rows": len(hold_idx),
                "holdout_candidate_ids": sorted(holdout_ids),
                "child_added_rows": int(
                    child_manifest.get("added_row_count", 0)
                ),
            },
            "incumbent": {
                "model_version": incumbent_model_version,
                "dataset_version": parent_dataset_version,
                "cv": incumbent_cv,
                "holdout": incumbent_holdout,
            },
            "challenger": {
                "model_version": challenger_model_version,
                "dataset_version": child_dataset_version,
                "cv": challenger_cv,
                "holdout": challenger_holdout,
            },
            "registry": {
                "active_model_version_after_decision": registry_after.get(
                    "active_model_version"
                ),
                "challenger_status_after_decision": (
                    registry_after["models"]
                    .get(challenger_model_version, {})
                    .get("status")
                ),
                "automatic_activation": False,
                "human_approval_required": (
                    decision.get("decision") == "PROMOTE"
                ),
            },
        }

        return self._persist_report(
            project_id=project_id,
            target_metric=target_metric,
            parent_dataset_version=parent_dataset_version,
            child_dataset_version=child_dataset_version,
            incumbent_model_version=incumbent_model_version,
            challenger_model_version=challenger_model_version,
            model_family=model_family,
            gate=gate,
            decision=decision,
            details=details,
        )

    def _persist_report(
        self,
        *,
        project_id: int,
        target_metric: str,
        parent_dataset_version: str,
        child_dataset_version: str,
        incumbent_model_version: str,
        challenger_model_version: str,
        model_family: str,
        gate: dict[str, Any],
        decision: dict[str, Any],
        details: dict[str, Any],
    ) -> dict[str, Any]:
        report = {
            "stage": MODEL_STAGE,
            "schema_version": MODEL_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "project_id": project_id,
            "target_metric": target_metric,
            "model_family": model_family,
            "parent_dataset_version": parent_dataset_version,
            "child_dataset_version": child_dataset_version,
            "incumbent_model_version": incumbent_model_version,
            "challenger_model_version": challenger_model_version,
            "gate": {
                "decision": gate.get("decision"),
                "training_allowed": gate.get("training_allowed"),
                "official_model_allowed": gate.get("official_model_allowed"),
            },
            **decision,
            **details,
            "safety": {
                "automatic_model_replacement": False,
                "human_approval_required_for_activation": True,
                "note": (
                    "PROMOTE 是系统建议，不等于模型已切换。"
                    "V0.2 必须由人工明确批准后才改变 active_model_version。"
                ),
            },
        }
        path = (
            self.runtime_root
            / "v020"
            / "model_promotion"
            / f"project_{project_id}"
            / _safe_name(target_metric, "target_metric")
            / "promotion_report.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        report["report_json"] = str(path)
        return deepcopy(report)
