from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments.protocol import PROTOCOL_TEMPLATE_STAGE


def fixture_template() -> dict:
    return {
        "stage": PROTOCOL_TEMPLATE_STAGE,
        "template_id": "V030_T27_POLYMER_IMPACT_V1",
        "name": "聚合物配混 + 冲击强度实验协议",
        "project_id": 9030,
        "parameters": [
            {"name": "ABS", "source_feature": "formula::ABS", "section": "material_recipe", "kind": "continuous", "canonical_unit": "%", "default_input_unit": "%", "required": True, "safety": {"min": 0, "max": 100}},
            {"name": "PC", "source_feature": "formula::PC", "section": "material_recipe", "kind": "continuous", "canonical_unit": "%", "default_input_unit": "%", "required": True, "safety": {"min": 0, "max": 100}},
            {"name": "增韧剂", "source_feature": "formula::增韧剂", "section": "material_recipe", "kind": "continuous", "canonical_unit": "%", "default_input_unit": "%", "required": False, "safety": {"min": 0, "max": 30}},
            {"name": "加工温度", "source_feature": "process::加工温度", "section": "process_parameter", "kind": "continuous", "canonical_unit": "°C", "default_input_unit": "°C", "required": True, "safety": {"min": 180, "max": 280}},
            {"name": "螺杆转速", "source_feature": "process::螺杆转速", "section": "process_parameter", "kind": "integer", "canonical_unit": "rpm", "default_input_unit": "rpm", "required": True, "safety": {"min": 50, "max": 800}},
            {"name": "混炼时间", "source_feature": "process::混炼时间", "section": "process_parameter", "kind": "continuous", "canonical_unit": "min", "default_input_unit": "min", "required": True, "safety": {"min": 1, "max": 180}},
            {"name": "模式", "source_feature": "process::模式", "section": "process_parameter", "kind": "categorical", "canonical_unit": "", "required": True, "safety": {"allowed": ["standard", "gentle"]}},
        ],
        "process_steps": [
            {"step_id": "weigh", "name": "称量", "device_role": "material_dispenser", "parameters": ["ABS", "PC", "增韧剂"], "instructions": "按配方比例称量原料。"},
            {"step_id": "compound", "name": "熔融混炼", "device_role": "compounder", "parameters": ["加工温度", "螺杆转速", "混炼时间", "模式"], "instructions": "执行确定性混炼程序。"},
        ],
        "measurement_steps": [
            {"step_id": "impact", "name": "悬臂梁冲击测试", "device_role": "impact_tester", "metric": "冲击强度", "unit": "kJ/m²", "condition_signature": "ISO180_23C_NOTCHED", "instructions": "按固定条件测试。"}
        ],
        "expected_outputs": [
            {"metric": "冲击强度", "unit": "kJ/m²", "required": True}
        ],
        "metadata": {"fixture_only": True, "note": "T27 工程协议 fixture，不代表单位真实工艺参数。"},
    }


def candidates() -> dict:
    base = {
        "candidate_id": "V030_T27_VALID",
        "source_context": {"campaign_id": "V030_T27_DEMO", "round_id": "R001"},
        "features": {
            "formula::ABS": 42.0,
            "formula::PC": 53.0,
            "formula::增韧剂": 5.0,
            "process::加工温度": {"value": 503.15, "unit": "K"},
            "process::螺杆转速": {"value": 320, "unit": "rpm"},
            "process::混炼时间": {"value": 7200, "unit": "s"},
            "process::模式": "standard",
        },
    }
    missing = json.loads(json.dumps(base, ensure_ascii=False))
    missing["candidate_id"] = "V030_T27_MISSING"
    del missing["features"]["process::加工温度"]

    bad_unit = json.loads(json.dumps(base, ensure_ascii=False))
    bad_unit["candidate_id"] = "V030_T27_BAD_UNIT"
    bad_unit["features"]["process::加工温度"] = {"value": 230, "unit": "fahrenheit"}

    unsafe = json.loads(json.dumps(base, ensure_ascii=False))
    unsafe["candidate_id"] = "V030_T27_UNSAFE"
    unsafe["features"]["process::加工温度"] = {"value": 320, "unit": "°C"}

    category = json.loads(json.dumps(base, ensure_ascii=False))
    category["candidate_id"] = "V030_T27_BAD_CATEGORY"
    category["features"]["process::模式"] = "turbo"

    optional_missing = json.loads(json.dumps(base, ensure_ascii=False))
    optional_missing["candidate_id"] = "V030_T27_OPTIONAL_MISSING"
    del optional_missing["features"]["formula::增韧剂"]

    return {
        "valid": base,
        "missing": missing,
        "bad_unit": bad_unit,
        "unsafe": unsafe,
        "bad_category": category,
        "optional_missing": optional_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(args.runtime_root) / "v030" / "fixtures" / "t27"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol_template.json").write_text(json.dumps(fixture_template(), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "candidates.json").write_text(json.dumps(candidates(), ensure_ascii=False, indent=2), encoding="utf-8")

    print("V0.3-T27 FIXTURE BUILDER")
    print(f"template: {root / 'protocol_template.json'}")
    print(f"candidates: {len(candidates())}")
    print("fixture_only: true")
    print()
    print("V0.3-T27 FIXTURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
