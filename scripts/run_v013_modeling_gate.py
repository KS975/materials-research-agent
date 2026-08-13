from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIN_CORE_FOR_ANY_MODEL = 10
MIN_CORE_FOR_PASS = 30
MIN_CONDITION_COVERAGE = 0.80


def require_dict(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SystemExit(
            f"ERROR: invalid V0.1.3-A reality schema: "
            f"missing or invalid object '{key}'"
        )
    return value


def require_int(data: dict, key: str, path: str) -> int:
    if key not in data:
        raise SystemExit(
            f"ERROR: invalid V0.1.3-A reality schema: "
            f"missing '{path}.{key}'"
        )

    value = data[key]

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    raise SystemExit(
        f"ERROR: invalid V0.1.3-A reality schema: "
        f"'{path}.{key}' is not numeric: {value!r}"
    )


def find_key(obj: Any, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            result = find_key(value, key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_key(item, key)
            if result is not None:
                return result
    return None


def optional_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def count_unresolved_dynamic_fields(unresolved: Any) -> int:
    if unresolved is None:
        return 0

    if not isinstance(unresolved, dict):
        raise SystemExit(
            "ERROR: invalid V0.1.3-A reality schema: "
            "'unresolved_dynamic_fields' must be an object"
        )

    total = 0

    for field_name, count in unresolved.items():
        if not isinstance(count, (int, float)):
            raise SystemExit(
                "ERROR: invalid unresolved dynamic field count: "
                f"{field_name}={count!r}"
            )
        total += int(count)

    return total


def main():
    parser = argparse.ArgumentParser(
        description="V0.1.3-B Modeling Gate"
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--reality-json", default=None)
    args = parser.parse_args()

    project_id = args.project_id
    target_metric = args.target

    if args.reality_json:
        reality_path = Path(args.reality_json)
    else:
        reality_path = (
            Path(".runtime")
            / "v013"
            / "reality"
            / f"project_{project_id}_{target_metric}_reality.json"
        )

    print("V0.1.3-B MODELING GATE")
    print(f"project_id: {project_id}")
    print(f"target_metric: {target_metric}")
    print()

    if not reality_path.exists():
        raise SystemExit(
            f"ERROR: reality report not found: {reality_path}\n"
            "Run V0.1.3-A Dataset Reality Check first."
        )

    with reality_path.open("r", encoding="utf-8") as f:
        reality = json.load(f)

    stage = reality.get("stage")
    if stage != "V0.1.3-A_dataset_reality_check":
        raise SystemExit(
            "ERROR: input JSON is not a valid "
            "V0.1.3-A Dataset Reality report.\n"
            f"stage={stage!r}"
        )

    summary = require_dict(reality, "summary")
    target_info = require_dict(reality, "target")
    test_conditions = require_dict(reality, "test_conditions")

    total_samples = require_int(summary, "total_samples", "summary")
    formula_present = require_int(summary, "formula_present", "summary")
    process_present = require_int(summary, "process_present", "summary")
    target_present = require_int(summary, "target_present", "summary")
    conditions_present = require_int(summary, "conditions_present", "summary")
    core_closed = require_int(
        summary,
        "core_closed_formula_process_target",
        "summary",
    )
    strict_closed = require_int(
        summary,
        "strict_closed_with_conditions",
        "summary",
    )
    target_numeric_count = require_int(
        target_info,
        "numeric_count",
        "target",
    )
    condition_signatures = require_int(
        test_conditions,
        "unique_nonempty_signatures",
        "test_conditions",
    )

    unresolved_dynamic_fields = count_unresolved_dynamic_fields(
        reality.get("unresolved_dynamic_fields")
    )

    duplicate_sample_names = optional_count(
        find_key(reality, "duplicate_sample_name_groups")
    )
    duplicate_formula_process_target = optional_count(
        find_key(reality, "duplicate_formula_process_target_groups")
    )

    schema_warnings: list[str] = []

    if target_numeric_count > target_present:
        raise SystemExit(
            "ERROR: invalid reality report: "
            "target.numeric_count > summary.target_present"
        )

    if strict_closed > core_closed:
        raise SystemExit(
            "ERROR: invalid reality report: "
            "strict_closed_with_conditions "
            "> core_closed_formula_process_target"
        )

    if core_closed > total_samples:
        raise SystemExit(
            "ERROR: invalid reality report: "
            "core_closed > total_samples"
        )

    reality_target_metric = reality.get("target_metric")
    if reality_target_metric and reality_target_metric != target_metric:
        schema_warnings.append(
            "A 阶段 JSON 中 target_metric 与命令行目标名称不一致；"
            "当前已知可能存在中文编码显示问题"
        )

    condition_coverage_on_core = (
        strict_closed / core_closed if core_closed > 0 else 0.0
    )

    failures: list[str] = []
    warnings: list[str] = []

    if core_closed < MIN_CORE_FOR_ANY_MODEL:
        failures.append(
            f"配方+工艺+目标闭环样本仅 {core_closed} 条，"
            f"低于最低建模准入值 {MIN_CORE_FOR_ANY_MODEL}"
        )

    if strict_closed == 0:
        failures.append(
            "没有任何同时包含测试条件的完整闭环样本"
        )

    if (
        core_closed > 0
        and condition_coverage_on_core < MIN_CONDITION_COVERAGE
    ):
        failures.append(
            "闭环样本测试条件覆盖不足："
            f"{strict_closed}/{core_closed} "
            f"({condition_coverage_on_core:.1%})，"
            f"最低要求 {MIN_CONDITION_COVERAGE:.0%}"
        )

    if condition_signatures == 0:
        failures.append(
            "没有可识别的非空测试条件签名，无法证明目标值可比"
        )
    elif condition_signatures > 1:
        failures.append(
            f"检测到 {condition_signatures} 种测试条件签名；"
            "当前版本禁止将不同测试条件直接混合为正式监督学习数据集"
        )

    if unresolved_dynamic_fields > 0:
        warnings.append(
            f"存在 {unresolved_dynamic_fields} 个未解析动态字段实例"
        )

    if duplicate_sample_names > 0:
        warnings.append(
            f"存在 {duplicate_sample_names} 组重复样品名称"
        )

    if duplicate_formula_process_target > 0:
        warnings.append(
            f"存在 {duplicate_formula_process_target} "
            "组重复配方-工艺-目标组合"
        )

    warnings.extend(schema_warnings)

    conditional_reasons: list[str] = []

    if failures:
        decision = "FAIL"
        training_allowed = False
        official_model_allowed = False
    else:
        if core_closed < MIN_CORE_FOR_PASS:
            conditional_reasons.append(
                f"闭环样本 {core_closed} 条，"
                f"尚未达到正式 PASS 推荐值 {MIN_CORE_FOR_PASS}"
            )

        if condition_coverage_on_core < 1.0:
            conditional_reasons.append(
                "部分核心闭环样本缺少测试条件"
            )

        if unresolved_dynamic_fields > 0:
            conditional_reasons.append(
                "仍存在未解析动态字段"
            )

        if conditional_reasons:
            decision = "CONDITIONAL_PASS"
            training_allowed = True
            official_model_allowed = False
            warnings.extend(conditional_reasons)
        else:
            decision = "PASS"
            training_allowed = True
            official_model_allowed = True

    print("DATASET SUMMARY")
    print(f"total_samples: {total_samples}")
    print(f"formula_present: {formula_present}")
    print(f"process_present: {process_present}")
    print(f"target_present: {target_present}")
    print(f"target_numeric_count: {target_numeric_count}")
    print(f"conditions_present: {conditions_present}")
    print(f"core_closed_samples: {core_closed}")
    print(f"strict_closed_samples: {strict_closed}")
    print(
        "condition_coverage_on_core: "
        f"{condition_coverage_on_core:.1%}"
    )
    print(
        "test_condition_unique_nonempty_signatures: "
        f"{condition_signatures}"
    )
    print(
        "unresolved_dynamic_field_instances: "
        f"{unresolved_dynamic_fields}"
    )
    print()

    print("DECISION")
    print(decision)
    print()

    if failures:
        print("FAIL REASONS")
        print()
        for reason in failures:
            print(f"- {reason}")
        print()

    if warnings:
        print("WARNINGS")
        print()
        for warning in warnings:
            print(f"- {warning}")
        print()

    print(
        "TRAINING_ALLOWED: "
        f"{str(training_allowed).lower()}"
    )
    print(
        "OFFICIAL_MODEL_ALLOWED: "
        f"{str(official_model_allowed).lower()}"
    )

    output_dir = Path(".runtime") / "v013" / "gates"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"project_{project_id}_{target_metric}_modeling_gate.json"
    )

    output = {
        "stage": "V0.1.3-B_modeling_gate",
        "project_id": project_id,
        "target_metric": target_metric,
        "source_reality_json": str(reality_path),
        "dataset_summary": {
            "total_samples": total_samples,
            "formula_present": formula_present,
            "process_present": process_present,
            "target_present": target_present,
            "target_numeric_count": target_numeric_count,
            "conditions_present": conditions_present,
            "core_closed_samples": core_closed,
            "strict_closed_samples": strict_closed,
            "condition_coverage_on_core": condition_coverage_on_core,
            "test_condition_unique_nonempty_signatures": condition_signatures,
            "unresolved_dynamic_field_instances": unresolved_dynamic_fields,
            "duplicate_sample_name_groups": duplicate_sample_names,
            "duplicate_formula_process_target_groups":
                duplicate_formula_process_target,
        },
        "gate_policy": {
            "min_core_for_any_model": MIN_CORE_FOR_ANY_MODEL,
            "min_core_for_pass": MIN_CORE_FOR_PASS,
            "min_condition_coverage": MIN_CONDITION_COVERAGE,
        },
        "decision": decision,
        "training_allowed": training_allowed,
        "official_model_allowed": official_model_allowed,
        "fail_reasons": failures,
        "warnings": warnings,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("OUTPUT")
    print(f"gate_json: {output_path}")
    print()

    if decision == "FAIL":
        print(
            "V0.1.3-B MODELING GATE PASS "
            "(correctly blocked modeling)"
        )
    else:
        print(
            "V0.1.3-B MODELING GATE PASS "
            f"({decision})"
        )


if __name__ == "__main__":
    main()
