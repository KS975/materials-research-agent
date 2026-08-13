from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_target_metric(target_metric: str) -> str:
    value = target_metric.strip()
    if not value:
        raise ValueError("target_metric 不能为空")
    if len(value) > 80 or ".." in value or "/" in value or "\\" in value:
        raise ValueError("target_metric 包含非法路径字符")
    return value


def load_json_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"V0.1.3 报告格式错误: {path.name}")
    return data


def load_v013_status(runtime_root: Path, project_id: int, target_metric: str) -> dict[str, Any]:
    target_metric = validate_target_metric(target_metric)
    stem = f"project_{project_id}_{target_metric}"

    reality = load_json_report(runtime_root / "reality" / f"{stem}_reality.json")
    gate = load_json_report(runtime_root / "gates" / f"{stem}_modeling_gate.json")
    comparison = load_json_report(
        runtime_root / "model_comparison" / stem / "leaderboard.json"
    )
    cross_validation = load_json_report(
        runtime_root / "cross_validation" / stem / "cv_report.json"
    )

    ad_dir = runtime_root / "applicability_domain" / stem
    ad_paths = sorted(
        ad_dir.glob("*_ad_report.json") if ad_dir.is_dir() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    latest_ad = load_json_report(ad_paths[0]) if ad_paths else None

    return {
        "kind": "v013_modeling_status",
        "stage": "V0.1.3-UI_status",
        "project_id": project_id,
        "target_metric": target_metric,
        "availability": {
            "reality": reality is not None,
            "gate": gate is not None,
            "model_comparison": comparison is not None,
            "cross_validation": cross_validation is not None,
            "applicability_domain": latest_ad is not None,
        },
        "reality": reality,
        "gate": gate,
        "model_comparison": comparison,
        "cross_validation": cross_validation,
        "latest_applicability_domain": latest_ad,
    }
