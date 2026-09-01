from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from .campaign import CampaignStore, find_round

DATASET_STAGE = "V0.2-T22_dataset_version"
DATASET_SCHEMA_VERSION = 1


class DatasetVersionError(RuntimeError):
    pass


class DatasetVersionValidationError(DatasetVersionError):
    pass


class DatasetVersionConflictError(DatasetVersionError):
    pass


class DatasetVersionNotFoundError(DatasetVersionError):
    pass


class DatasetIntegrityError(DatasetVersionError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise DatasetVersionValidationError(f"{name} 不能为空")
    return result


def _safe_id(value: Any, name: str) -> str:
    result = _text(value, name)
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", result):
        raise DatasetVersionValidationError(
            f"{name} 只能包含字母、数字、点、下划线和短横线"
        )
    return result


def _project_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetVersionValidationError("project_id 必须是整数")
    return int(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise DatasetVersionNotFoundError(f"CSV 不存在: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise DatasetVersionValidationError(f"CSV 缺少表头: {path}")
        columns = [str(c) for c in reader.fieldnames]
        rows = [dict(row) for row in reader]
    return columns, rows


def _write_csv_atomic(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Stable readable serialization while preserving numeric identity.
        return format(value, ".15g")
    return str(value)


def build_training_row(
    *,
    project_id: int,
    campaign_id: str,
    round_id: str,
    experiment: dict[str, Any],
) -> dict[str, str]:
    result = experiment.get("result") or {}
    if result.get("training_eligible") is not True:
        raise DatasetVersionValidationError(
            f"experiment 不是 training_eligible: {experiment.get('candidate_id')}"
        )
    if experiment.get("status") != "COMPLETED":
        raise DatasetVersionValidationError(
            "只有 COMPLETED experiment 可以构建训练行"
        )

    row: dict[str, str] = {
        "candidate_id": _text(experiment.get("candidate_id"), "candidate_id"),
        "project_id": str(project_id),
        "test_condition_signature": _text(
            result.get("test_condition_signature"),
            "test_condition_signature",
        ),
        "source_campaign": campaign_id,
        "source_round": round_id,
    }

    features = experiment.get("features") or {}
    if not isinstance(features, dict):
        raise DatasetVersionValidationError("experiment.features 必须是 JSON object")
    for key, value in features.items():
        row[str(key)] = _normalize_scalar(value)

    measurements = result.get("measurements") or {}
    if not isinstance(measurements, dict) or not measurements:
        raise DatasetVersionValidationError(
            f"COMPLETED experiment 缺少 measurements: {row['candidate_id']}"
        )
    for metric, value in measurements.items():
        row[str(metric)] = _normalize_scalar(value)

    return row


def _row_signature(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_scalar(row.get(c, "")) for c in columns)


class DatasetVersionStore:
    """Immutable, file-backed dataset versions for V0.2.

    Layout:
      <runtime_root>/v020/datasets/project_<project_id>/<dataset_version>/
        dataset.csv
        manifest.json
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.runtime_root = Path(runtime_root)

    def project_dir(self, project_id: int) -> Path:
        return self.runtime_root / "v020" / "datasets" / f"project_{_project_id(project_id)}"

    def version_dir(self, project_id: int, dataset_version: str) -> Path:
        return self.project_dir(project_id) / _safe_id(dataset_version, "dataset_version")

    def dataset_path(self, project_id: int, dataset_version: str) -> Path:
        return self.version_dir(project_id, dataset_version) / "dataset.csv"

    def manifest_path(self, project_id: int, dataset_version: str) -> Path:
        return self.version_dir(project_id, dataset_version) / "manifest.json"

    def exists(self, project_id: int, dataset_version: str) -> bool:
        return self.dataset_path(project_id, dataset_version).exists() and self.manifest_path(project_id, dataset_version).exists()

    def load_manifest(self, project_id: int, dataset_version: str) -> dict[str, Any]:
        path = self.manifest_path(project_id, dataset_version)
        if not path.exists():
            raise DatasetVersionNotFoundError(
                f"dataset manifest 不存在: project={project_id}, version={dataset_version}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("stage") != DATASET_STAGE:
            raise DatasetIntegrityError(
                f"dataset manifest stage 非法: {data.get('stage')}"
            )
        return data

    def load_rows(self, project_id: int, dataset_version: str, *, verify: bool = True) -> tuple[list[str], list[dict[str, str]]]:
        if verify:
            self.verify(project_id, dataset_version)
        return _read_csv_rows(self.dataset_path(project_id, dataset_version))

    def verify(self, project_id: int, dataset_version: str) -> dict[str, Any]:
        manifest = self.load_manifest(project_id, dataset_version)
        path = self.dataset_path(project_id, dataset_version)
        if not path.exists():
            raise DatasetIntegrityError(f"dataset.csv 缺失: {path}")
        actual_hash = sha256_file(path)
        expected_hash = str(manifest.get("sha256") or "")
        if actual_hash != expected_hash:
            raise DatasetIntegrityError(
                f"dataset SHA256 不一致: expected={expected_hash}, actual={actual_hash}"
            )
        columns, rows = _read_csv_rows(path)
        if len(rows) != int(manifest.get("row_count", -1)):
            raise DatasetIntegrityError("dataset row_count 与 manifest 不一致")
        if columns != list(manifest.get("columns") or []):
            raise DatasetIntegrityError("dataset columns 与 manifest 不一致")
        return {
            "verified": True,
            "project_id": project_id,
            "dataset_version": dataset_version,
            "sha256": actual_hash,
            "row_count": len(rows),
            "column_count": len(columns),
        }

    def register_base_csv(
        self,
        *,
        project_id: int,
        dataset_version: str,
        source_csv: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project_id = _project_id(project_id)
        dataset_version = _safe_id(dataset_version, "dataset_version")
        source_csv = Path(source_csv)
        columns, rows = _read_csv_rows(source_csv)
        if not rows:
            raise DatasetVersionValidationError("base dataset 不能为空")
        if "candidate_id" not in columns:
            raise DatasetVersionValidationError("base dataset 必须包含 candidate_id")
        ids = [str(r.get("candidate_id") or "").strip() for r in rows]
        if any(not x for x in ids):
            raise DatasetVersionValidationError("base dataset candidate_id 不能为空")
        if len(ids) != len(set(ids)):
            raise DatasetVersionValidationError("base dataset candidate_id 存在重复")
        if self.version_dir(project_id, dataset_version).exists():
            raise DatasetVersionConflictError(
                f"dataset version 已存在，禁止覆盖: {dataset_version}"
            )

        out = self.dataset_path(project_id, dataset_version)
        _write_csv_atomic(out, columns, rows)
        digest = sha256_file(out)
        manifest = {
            "stage": DATASET_STAGE,
            "schema_version": DATASET_SCHEMA_VERSION,
            "project_id": project_id,
            "dataset_version": dataset_version,
            "parent_dataset_version": None,
            "parent_sha256": None,
            "created_at": utc_now_iso(),
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "sha256": digest,
            "source": {
                "type": "BASE_IMPORT",
                "source_csv_name": source_csv.name,
            },
            "added_candidate_ids": ids,
            "added_row_count": len(rows),
            "excluded_nontraining": {},
            "duplicate_skipped_candidate_ids": [],
            "metadata": deepcopy(metadata or {}),
        }
        _write_json_atomic(self.manifest_path(project_id, dataset_version), manifest)
        return deepcopy(manifest)

    def _find_round_application(self, project_id: int, campaign_id: str, round_id: str) -> dict[str, Any] | None:
        project_dir = self.project_dir(project_id)
        if not project_dir.exists():
            return None
        for manifest_path in project_dir.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            source = manifest.get("source") or {}
            if (
                source.get("type") == "CAMPAIGN_ROUND_UPDATE"
                and source.get("campaign_id") == campaign_id
                and source.get("round_id") == round_id
            ):
                return manifest
        return None

    def update_from_round(
        self,
        *,
        campaign_store: CampaignStore,
        campaign_id: str,
        round_id: str,
        new_dataset_version: str,
    ) -> dict[str, Any]:
        campaign = campaign_store.load(campaign_id)
        project_id = _project_id(campaign.get("project_id"))
        round_record = find_round(campaign, round_id)

        if round_record.get("status") != "COMPLETED":
            raise DatasetVersionConflictError(
                "只有 COMPLETED Round 可以生成新 dataset version"
            )

        parent_version = _safe_id(
            round_record.get("plan", {}).get("dataset_version"),
            "round.plan.dataset_version",
        )
        new_dataset_version = _safe_id(new_dataset_version, "new_dataset_version")
        if new_dataset_version == parent_version:
            raise DatasetVersionConflictError(
                "new_dataset_version 不能与 parent dataset version 相同"
            )

        existing_application = self._find_round_application(project_id, campaign_id, round_id)
        if existing_application is not None:
            if existing_application.get("dataset_version") == new_dataset_version:
                self.verify(project_id, new_dataset_version)
                return {
                    "idempotent_replay": True,
                    "manifest": deepcopy(existing_application),
                }
            raise DatasetVersionConflictError(
                f"该 Round 已生成 dataset version: {existing_application.get('dataset_version')}"
            )

        if self.version_dir(project_id, new_dataset_version).exists():
            raise DatasetVersionConflictError(
                f"dataset version 已存在，禁止覆盖: {new_dataset_version}"
            )

        parent_manifest = self.load_manifest(project_id, parent_version)
        if int(parent_manifest.get("project_id")) != project_id:
            raise DatasetVersionConflictError("parent dataset project_id 不一致")
        self.verify(project_id, parent_version)
        parent_path = self.dataset_path(project_id, parent_version)
        parent_hash_before = sha256_file(parent_path)
        parent_columns, parent_rows = _read_csv_rows(parent_path)

        experiments = round_record.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise DatasetVersionValidationError("Round 没有实验记录，无法更新数据集")

        excluded = {"FAILED": 0, "INVALID": 0, "NOT_TESTED": 0}
        training_rows: list[dict[str, str]] = []
        for experiment in experiments:
            status = str(experiment.get("status") or "")
            result = experiment.get("result") or {}
            if status == "COMPLETED" and result.get("training_eligible") is True:
                training_rows.append(
                    build_training_row(
                        project_id=project_id,
                        campaign_id=campaign_id,
                        round_id=round_id,
                        experiment=experiment,
                    )
                )
            elif status in excluded:
                excluded[status] += 1
            elif status == "COMPLETED":
                # A completed-but-not-training-eligible row must never silently enter.
                excluded.setdefault("COMPLETED_NOT_ELIGIBLE", 0)
                excluded["COMPLETED_NOT_ELIGIBLE"] += 1
            else:
                raise DatasetVersionConflictError(
                    f"Round 含非终态实验，不能更新 dataset: {experiment.get('candidate_id')}={status}"
                )

        if not training_rows:
            raise DatasetVersionConflictError(
                "本 Round 没有 training_eligible 实验，不能创建空更新版本"
            )

        # Schema is immutable in T22. Every training row must fit the parent columns.
        missing_required = []
        for row in training_rows:
            missing = [c for c in parent_columns if c not in row]
            if missing:
                missing_required.append((row.get("candidate_id"), missing))
            extras = [c for c in row if c not in parent_columns]
            if extras:
                raise DatasetVersionValidationError(
                    f"新实验字段不在 parent schema 中: {row.get('candidate_id')} extras={extras}"
                )
        if missing_required:
            cid, missing = missing_required[0]
            raise DatasetVersionValidationError(
                f"新实验缺少 parent schema 字段: {cid} missing={missing}"
            )

        parent_by_id = {str(r.get("candidate_id") or "").strip(): r for r in parent_rows}
        rows_to_add: list[dict[str, str]] = []
        duplicate_skipped: list[str] = []

        for row in training_rows:
            cid = row["candidate_id"]
            existing = parent_by_id.get(cid)
            if existing is None:
                rows_to_add.append(row)
                continue
            if _row_signature(existing, parent_columns) == _row_signature(row, parent_columns):
                duplicate_skipped.append(cid)
                continue
            raise DatasetVersionConflictError(
                f"parent dataset 已存在相同 candidate_id 但内容不同，拒绝覆盖: {cid}"
            )

        if not rows_to_add and duplicate_skipped:
            raise DatasetVersionConflictError(
                "所有 training_eligible candidate 已存在于 parent dataset；拒绝创建无变化新版本"
            )

        child_rows = parent_rows + rows_to_add
        out_path = self.dataset_path(project_id, new_dataset_version)
        _write_csv_atomic(out_path, parent_columns, child_rows)
        child_hash = sha256_file(out_path)

        # Parent must be byte-for-byte untouched.
        parent_hash_after = sha256_file(parent_path)
        if parent_hash_after != parent_hash_before:
            # Defensive cleanup. This should be impossible because parent is read-only here.
            shutil_target = self.version_dir(project_id, new_dataset_version)
            if shutil_target.exists():
                import shutil
                shutil.rmtree(shutil_target, ignore_errors=True)
            raise DatasetIntegrityError("parent dataset 在更新过程中发生变化")

        manifest = {
            "stage": DATASET_STAGE,
            "schema_version": DATASET_SCHEMA_VERSION,
            "project_id": project_id,
            "dataset_version": new_dataset_version,
            "parent_dataset_version": parent_version,
            "parent_sha256": parent_hash_before,
            "created_at": utc_now_iso(),
            "row_count": len(child_rows),
            "row_count_before": len(parent_rows),
            "row_count_after": len(child_rows),
            "column_count": len(parent_columns),
            "columns": parent_columns,
            "sha256": child_hash,
            "source": {
                "type": "CAMPAIGN_ROUND_UPDATE",
                "campaign_id": campaign_id,
                "round_id": round_id,
            },
            "added_candidate_ids": [r["candidate_id"] for r in rows_to_add],
            "added_row_count": len(rows_to_add),
            "excluded_nontraining": excluded,
            "duplicate_skipped_candidate_ids": duplicate_skipped,
            "duplicate_skipped_count": len(duplicate_skipped),
        }
        _write_json_atomic(self.manifest_path(project_id, new_dataset_version), manifest)
        self.verify(project_id, new_dataset_version)
        return {
            "idempotent_replay": False,
            "manifest": deepcopy(manifest),
        }
