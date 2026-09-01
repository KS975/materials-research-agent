from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROTOCOL_STAGE = "V0.3-T27_experiment_protocol"
PROTOCOL_TEMPLATE_STAGE = "V0.3-T27_protocol_template"
PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_STATUSES = {"READY", "BLOCKED"}
PARAMETER_SECTIONS = {"material_recipe", "process_parameter"}
PARAMETER_KINDS = {"continuous", "integer", "categorical"}


class ExperimentProtocolError(RuntimeError):
    pass


class ExperimentProtocolValidationError(ExperimentProtocolError):
    pass


class ExperimentProtocolConflictError(ExperimentProtocolError):
    pass


# Canonical-unit normalization intentionally supports only an explicit, small,
# deterministic set. T27 must never silently guess a unit conversion.
_UNIT_ALIASES = {
    "%": "%", "wt%": "%", "wt.%": "%", "mass%": "%",
    "°c": "°C", "c": "°C", "degc": "°C", "℃": "°C",
    "k": "K",
    "pa": "Pa", "kpa": "kPa", "mpa": "MPa", "bar": "bar",
    "rpm": "rpm", "r/min": "rpm", "rev/min": "rpm",
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "min": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "g": "g", "kg": "kg",
    "ml": "mL", "l": "L",
    "kj/m²": "kJ/m²", "kj/m2": "kJ/m²",
    "g/10min": "g/10min", "g/10 min": "g/10min",
    "": "",
}


_DIMENSION = {
    "%": "fraction",
    "°C": "temperature", "K": "temperature",
    "Pa": "pressure", "kPa": "pressure", "MPa": "pressure", "bar": "pressure",
    "rpm": "rotation",
    "s": "time", "min": "time", "h": "time",
    "g": "mass", "kg": "mass",
    "mL": "volume", "L": "volume",
    "kJ/m²": "impact_strength",
    "g/10min": "mfr",
    "": "dimensionless",
}


_SCALE_TO_BASE = {
    "%": 1.0,
    "Pa": 1.0, "kPa": 1_000.0, "MPa": 1_000_000.0, "bar": 100_000.0,
    "rpm": 1.0,
    "s": 1.0, "min": 60.0, "h": 3600.0,
    "g": 1.0, "kg": 1000.0,
    "mL": 1.0, "L": 1000.0,
    "kJ/m²": 1.0,
    "g/10min": 1.0,
    "": 1.0,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ExperimentProtocolValidationError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentProtocolValidationError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ExperimentProtocolValidationError(f"{name} 必须是有限数值")
    return number


def _normalize_unit_text(unit: Any) -> str:
    text = str(unit or "").strip()
    alias = _UNIT_ALIASES.get(text.lower())
    if alias is None:
        raise ExperimentProtocolValidationError(f"不支持的单位: {text!r}")
    return alias


def convert_unit(value: Any, from_unit: Any, to_unit: Any) -> float:
    number = _finite(value, "value")
    src = _normalize_unit_text(from_unit)
    dst = _normalize_unit_text(to_unit)
    if _DIMENSION[src] != _DIMENSION[dst]:
        raise ExperimentProtocolValidationError(
            f"单位维度不兼容: {src!r} -> {dst!r}"
        )
    if src == dst:
        return number

    if _DIMENSION[src] == "temperature":
        # temperature is affine, not scale-only
        kelvin = number if src == "K" else number + 273.15
        out = kelvin if dst == "K" else kelvin - 273.15
        return float(out)

    base = number * _SCALE_TO_BASE[src]
    return float(base / _SCALE_TO_BASE[dst])


def _parameter_template(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ExperimentProtocolValidationError(
            f"parameters[{index}] 必须是 object"
        )
    item = deepcopy(raw)
    name = str(item.get("name") or "").strip()
    source_feature = str(item.get("source_feature") or "").strip()
    section = str(item.get("section") or "").strip()
    kind = str(item.get("kind") or "continuous").strip().lower()
    required = bool(item.get("required", True))
    canonical_unit = str(item.get("canonical_unit") or "").strip()
    default_input_unit = str(
        item.get("default_input_unit")
        if item.get("default_input_unit") is not None
        else canonical_unit
    ).strip()

    if not name:
        raise ExperimentProtocolValidationError(
            f"parameters[{index}].name 不能为空"
        )
    if not source_feature:
        raise ExperimentProtocolValidationError(
            f"{name}.source_feature 不能为空"
        )
    if section not in PARAMETER_SECTIONS:
        raise ExperimentProtocolValidationError(
            f"{name}.section 必须是 {sorted(PARAMETER_SECTIONS)}"
        )
    if kind not in PARAMETER_KINDS:
        raise ExperimentProtocolValidationError(
            f"{name}.kind 必须是 {sorted(PARAMETER_KINDS)}"
        )

    if kind in {"continuous", "integer"}:
        # Validate declared units at template load time.
        _normalize_unit_text(canonical_unit)
        _normalize_unit_text(default_input_unit)
        if _DIMENSION[_normalize_unit_text(canonical_unit)] != _DIMENSION[_normalize_unit_text(default_input_unit)]:
            raise ExperimentProtocolValidationError(
                f"{name}: default_input_unit 与 canonical_unit 维度不兼容"
            )

    safety = item.get("safety") or {}
    if not isinstance(safety, dict):
        raise ExperimentProtocolValidationError(f"{name}.safety 必须是 object")
    minimum = safety.get("min")
    maximum = safety.get("max")
    allowed = safety.get("allowed")
    if minimum is not None:
        minimum = _finite(minimum, f"{name}.safety.min")
    if maximum is not None:
        maximum = _finite(maximum, f"{name}.safety.max")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ExperimentProtocolValidationError(
            f"{name}.safety min 不能大于 max"
        )
    if allowed is not None:
        if not isinstance(allowed, list) or not allowed:
            raise ExperimentProtocolValidationError(
                f"{name}.safety.allowed 必须是非空列表"
            )

    return {
        "name": name,
        "source_feature": source_feature,
        "section": section,
        "kind": kind,
        "required": required,
        "canonical_unit": canonical_unit,
        "default_input_unit": default_input_unit,
        "safety": {
            "min": minimum,
            "max": maximum,
            "allowed": deepcopy(allowed),
        },
    }


def validate_protocol_template(template: Any) -> dict[str, Any]:
    if not isinstance(template, dict):
        raise ExperimentProtocolValidationError("protocol template 必须是 object")
    if template.get("stage") != PROTOCOL_TEMPLATE_STAGE:
        raise ExperimentProtocolValidationError(
            f"template.stage 必须是 {PROTOCOL_TEMPLATE_STAGE!r}"
        )
    template_id = str(template.get("template_id") or "").strip()
    name = str(template.get("name") or "").strip()
    if not template_id or not name:
        raise ExperimentProtocolValidationError(
            "template_id / name 不能为空"
        )

    project_id = template.get("project_id")
    if project_id is not None and (
        isinstance(project_id, bool) or not isinstance(project_id, int)
    ):
        raise ExperimentProtocolValidationError(
            "project_id 必须是 integer 或 null"
        )

    raw_parameters = template.get("parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ExperimentProtocolValidationError("parameters 必须是非空列表")
    parameters = [_parameter_template(x, i) for i, x in enumerate(raw_parameters)]
    names = [x["name"] for x in parameters]
    sources = [x["source_feature"] for x in parameters]
    if len(set(names)) != len(names):
        raise ExperimentProtocolValidationError("parameter name 不能重复")
    if len(set(sources)) != len(sources):
        raise ExperimentProtocolValidationError("source_feature 不能重复")

    process_steps = deepcopy(template.get("process_steps") or [])
    measurement_steps = deepcopy(template.get("measurement_steps") or [])
    expected_outputs = deepcopy(template.get("expected_outputs") or [])
    for label, value in (
        ("process_steps", process_steps),
        ("measurement_steps", measurement_steps),
        ("expected_outputs", expected_outputs),
    ):
        if not isinstance(value, list):
            raise ExperimentProtocolValidationError(f"{label} 必须是 list")

    parameter_names = set(names)
    step_ids: set[str] = set()
    normalized_steps = []
    for i, raw in enumerate(process_steps):
        if not isinstance(raw, dict):
            raise ExperimentProtocolValidationError(
                f"process_steps[{i}] 必须是 object"
            )
        step_id = str(raw.get("step_id") or "").strip()
        if not step_id or step_id in step_ids:
            raise ExperimentProtocolValidationError(
                "process step_id 不能为空且不能重复"
            )
        step_ids.add(step_id)
        refs = raw.get("parameters") or []
        if not isinstance(refs, list):
            raise ExperimentProtocolValidationError(
                f"{step_id}.parameters 必须是 list"
            )
        unknown = [str(x) for x in refs if str(x) not in parameter_names]
        if unknown:
            raise ExperimentProtocolValidationError(
                f"{step_id} 引用了未知 parameter: {unknown}"
            )
        normalized_steps.append({
            "step_id": step_id,
            "name": str(raw.get("name") or step_id),
            "device_role": str(raw.get("device_role") or "").strip(),
            "parameters": [str(x) for x in refs],
            "instructions": str(raw.get("instructions") or "").strip(),
        })

    normalized_measurements = []
    measurement_ids: set[str] = set()
    for i, raw in enumerate(measurement_steps):
        if not isinstance(raw, dict):
            raise ExperimentProtocolValidationError(
                f"measurement_steps[{i}] 必须是 object"
            )
        step_id = str(raw.get("step_id") or "").strip()
        metric = str(raw.get("metric") or "").strip()
        unit = str(raw.get("unit") or "").strip()
        condition_signature = str(raw.get("condition_signature") or "").strip()
        if not step_id or step_id in measurement_ids:
            raise ExperimentProtocolValidationError(
                "measurement step_id 不能为空且不能重复"
            )
        if not metric or not condition_signature:
            raise ExperimentProtocolValidationError(
                f"{step_id}: metric / condition_signature 不能为空"
            )
        measurement_ids.add(step_id)
        normalized_measurements.append({
            "step_id": step_id,
            "name": str(raw.get("name") or step_id),
            "device_role": str(raw.get("device_role") or "").strip(),
            "metric": metric,
            "unit": unit,
            "condition_signature": condition_signature,
            "instructions": str(raw.get("instructions") or "").strip(),
        })

    normalized_outputs = []
    for i, raw in enumerate(expected_outputs):
        if not isinstance(raw, dict):
            raise ExperimentProtocolValidationError(
                f"expected_outputs[{i}] 必须是 object"
            )
        metric = str(raw.get("metric") or "").strip()
        if not metric:
            raise ExperimentProtocolValidationError(
                f"expected_outputs[{i}].metric 不能为空"
            )
        normalized_outputs.append({
            "metric": metric,
            "unit": str(raw.get("unit") or "").strip(),
            "required": bool(raw.get("required", True)),
        })

    normalized = {
        "stage": PROTOCOL_TEMPLATE_STAGE,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "template_id": template_id,
        "name": name,
        "project_id": project_id,
        "parameters": parameters,
        "process_steps": normalized_steps,
        "measurement_steps": normalized_measurements,
        "expected_outputs": normalized_outputs,
        "metadata": deepcopy(template.get("metadata") or {}),
    }
    normalized["template_sha256"] = sha256_json(normalized)
    return normalized


def _candidate_feature(candidate: dict[str, Any], source_feature: str) -> tuple[bool, Any, str | None]:
    features = candidate.get("features")
    if not isinstance(features, dict):
        return False, None, None
    if source_feature not in features:
        return False, None, None
    raw = features[source_feature]
    if isinstance(raw, dict):
        if "value" not in raw:
            return True, None, str(raw.get("unit") or "")
        return True, raw.get("value"), str(raw.get("unit") or "")
    return True, raw, None


def _issue(code: str, parameter: str | None, message: str) -> dict[str, Any]:
    return {"code": code, "parameter": parameter, "message": message}


def _resolve_parameter(
    candidate: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    present, raw_value, explicit_unit = _candidate_feature(
        candidate, spec["source_feature"]
    )
    issues: list[dict[str, Any]] = []
    if not present or raw_value is None or (
        isinstance(raw_value, str) and not raw_value.strip()
    ):
        if spec["required"]:
            issues.append(_issue(
                "MISSING_REQUIRED_PARAMETER",
                spec["name"],
                f"缺少必要参数 {spec['name']} ({spec['source_feature']})",
            ))
        return None, issues

    kind = spec["kind"]
    if kind == "categorical":
        value = raw_value
        allowed = spec["safety"].get("allowed")
        if allowed is not None and value not in allowed:
            issues.append(_issue(
                "INVALID_CHOICE",
                spec["name"],
                f"{spec['name']}={value!r} 不在允许集合 {allowed!r}",
            ))
        return {
            "name": spec["name"],
            "source_feature": spec["source_feature"],
            "kind": kind,
            "value": value,
            "unit": "",
        }, issues

    try:
        input_unit = (
            explicit_unit if explicit_unit is not None and explicit_unit.strip()
            else spec["default_input_unit"]
        )
        value = convert_unit(
            raw_value,
            input_unit,
            spec["canonical_unit"],
        )
    except ExperimentProtocolValidationError as exc:
        issues.append(_issue(
            "UNSUPPORTED_UNIT",
            spec["name"],
            str(exc),
        ))
        return None, issues

    if kind == "integer":
        rounded = round(value)
        if abs(value - rounded) > 1e-9:
            issues.append(_issue(
                "INVALID_INTEGER",
                spec["name"],
                f"{spec['name']} 转换后不是整数: {value}",
            ))
        value = int(rounded)

    minimum = spec["safety"].get("min")
    maximum = spec["safety"].get("max")
    if minimum is not None and float(value) < float(minimum) - 1e-9:
        issues.append(_issue(
            "SAFETY_LIMIT",
            spec["name"],
            f"{spec['name']}={value} {spec['canonical_unit']} < safety.min={minimum}",
        ))
    if maximum is not None and float(value) > float(maximum) + 1e-9:
        issues.append(_issue(
            "SAFETY_LIMIT",
            spec["name"],
            f"{spec['name']}={value} {spec['canonical_unit']} > safety.max={maximum}",
        ))

    return {
        "name": spec["name"],
        "source_feature": spec["source_feature"],
        "kind": kind,
        "value": value,
        "unit": spec["canonical_unit"],
    }, issues


def validate_protocol_document(protocol: Any) -> None:
    if not isinstance(protocol, dict):
        raise ExperimentProtocolValidationError("protocol 必须是 object")
    required = {
        "stage", "schema_version", "protocol_id", "template_id",
        "candidate_id", "status", "material_recipe", "process_parameters",
        "process_steps", "measurement_steps", "expected_outputs",
        "safety_limits", "validation", "content_sha256",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ExperimentProtocolValidationError(
            f"protocol 缺少字段: {missing}"
        )
    if protocol["stage"] != PROTOCOL_STAGE:
        raise ExperimentProtocolValidationError("protocol.stage 错误")
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ExperimentProtocolValidationError("protocol.schema_version 错误")
    if protocol["status"] not in PROTOCOL_STATUSES:
        raise ExperimentProtocolValidationError("protocol.status 错误")
    if not isinstance(protocol["validation"], dict):
        raise ExperimentProtocolValidationError("protocol.validation 必须是 object")
    issues = protocol["validation"].get("issues")
    if not isinstance(issues, list):
        raise ExperimentProtocolValidationError("validation.issues 必须是 list")
    if protocol["status"] == "READY" and issues:
        raise ExperimentProtocolValidationError(
            "READY protocol 不能带 blocking issues"
        )
    if protocol["status"] == "BLOCKED" and not issues:
        raise ExperimentProtocolValidationError(
            "BLOCKED protocol 必须包含 blocking issue"
        )

    content = deepcopy(protocol)
    actual_hash = str(content.pop("content_sha256") or "")
    expected_hash = sha256_json(content)
    if actual_hash != expected_hash:
        raise ExperimentProtocolValidationError(
            "protocol content_sha256 校验失败"
        )
    expected_id = f"proto_{expected_hash[:20]}"
    if protocol["protocol_id"] != expected_id:
        raise ExperimentProtocolValidationError(
            "protocol_id 与内容哈希不一致"
        )


class ExperimentProtocolBuilder:
    def __init__(self, template: dict[str, Any]) -> None:
        self.template = validate_protocol_template(template)

    def build(
        self,
        candidate: dict[str, Any],
        *,
        source_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise ExperimentProtocolValidationError("candidate 必须是 object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ExperimentProtocolValidationError(
                "candidate_id 不能为空"
            )
        if not isinstance(candidate.get("features"), dict):
            raise ExperimentProtocolValidationError(
                "candidate.features 必须是 object"
            )

        recipe: list[dict[str, Any]] = []
        process_parameters: list[dict[str, Any]] = []
        resolved_by_name: dict[str, dict[str, Any]] = {}
        issues: list[dict[str, Any]] = []
        safety_limits: list[dict[str, Any]] = []

        for spec in self.template["parameters"]:
            value, parameter_issues = _resolve_parameter(candidate, spec)
            issues.extend(parameter_issues)
            if value is not None:
                resolved_by_name[spec["name"]] = value
                if spec["section"] == "material_recipe":
                    recipe.append(value)
                else:
                    process_parameters.append(value)
            safety_limits.append({
                "parameter": spec["name"],
                "kind": spec["kind"],
                "unit": spec["canonical_unit"],
                "min": spec["safety"].get("min"),
                "max": spec["safety"].get("max"),
                "allowed": deepcopy(spec["safety"].get("allowed")),
            })

        resolved_steps: list[dict[str, Any]] = []
        for step in self.template["process_steps"]:
            resolved_steps.append({
                **deepcopy(step),
                "resolved_parameters": [
                    deepcopy(resolved_by_name[name])
                    for name in step["parameters"]
                    if name in resolved_by_name
                ],
            })

        status = "BLOCKED" if issues else "READY"
        protocol = {
            "stage": PROTOCOL_STAGE,
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            # Placeholder is part of hashed content, so keep deterministic.
            "protocol_id": "",
            "template_id": self.template["template_id"],
            "template_sha256": self.template["template_sha256"],
            "project_id": self.template.get("project_id"),
            "candidate_id": candidate_id,
            "status": status,
            "source_context": deepcopy(source_context or candidate.get("source_context") or {}),
            "material_recipe": recipe,
            "process_parameters": process_parameters,
            "process_steps": resolved_steps,
            "measurement_steps": deepcopy(self.template["measurement_steps"]),
            "expected_outputs": deepcopy(self.template["expected_outputs"]),
            "safety_limits": safety_limits,
            "validation": {
                "blocking_issue_count": len(issues),
                "issues": issues,
            },
        }
        # ID is derived from content with protocol_id intentionally blank. This
        # avoids recursive hashing and makes identical input deterministic.
        pre_id_hash = sha256_json(protocol)
        protocol["protocol_id"] = f"proto_{pre_id_hash[:20]}"
        protocol["content_sha256"] = sha256_json(protocol)
        # Rebind ID to final content hash to make persisted docs self-checking.
        # One extra deterministic pass is used because the ID itself is content.
        protocol["protocol_id"] = f"proto_{protocol['content_sha256'][:20]}"
        protocol["content_sha256"] = sha256_json({k: v for k, v in protocol.items() if k != "content_sha256"})
        protocol["protocol_id"] = f"proto_{protocol['content_sha256'][:20]}"
        protocol["content_sha256"] = sha256_json({k: v for k, v in protocol.items() if k != "content_sha256"})
        # At this point content_sha256 includes final protocol_id. For validation
        # we define protocol_id as the prefix of the final hash; iterate to a
        # fixed deterministic representation by hashing a canonical payload that
        # excludes both derived fields, then derive both fields from that base.
        base_payload = {k: v for k, v in protocol.items() if k not in {"protocol_id", "content_sha256"}}
        base_hash = sha256_json(base_payload)
        protocol["protocol_id"] = f"proto_{base_hash[:20]}"
        content_for_hash = {k: v for k, v in protocol.items() if k != "content_sha256"}
        protocol["content_sha256"] = sha256_json(content_for_hash)
        return protocol


# Validation helper for the stable derived-field convention used above.
def validate_protocol_document(protocol: Any) -> None:  # type: ignore[no-redef]
    if not isinstance(protocol, dict):
        raise ExperimentProtocolValidationError("protocol 必须是 object")
    required = {
        "stage", "schema_version", "protocol_id", "template_id",
        "candidate_id", "status", "material_recipe", "process_parameters",
        "process_steps", "measurement_steps", "expected_outputs",
        "safety_limits", "validation", "content_sha256",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ExperimentProtocolValidationError(f"protocol 缺少字段: {missing}")
    if protocol["stage"] != PROTOCOL_STAGE:
        raise ExperimentProtocolValidationError("protocol.stage 错误")
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ExperimentProtocolValidationError("protocol.schema_version 错误")
    if protocol["status"] not in PROTOCOL_STATUSES:
        raise ExperimentProtocolValidationError("protocol.status 错误")
    validation = protocol.get("validation")
    if not isinstance(validation, dict) or not isinstance(validation.get("issues"), list):
        raise ExperimentProtocolValidationError("protocol.validation 结构错误")
    issues = validation["issues"]
    if protocol["status"] == "READY" and issues:
        raise ExperimentProtocolValidationError("READY protocol 不能带 blocking issues")
    if protocol["status"] == "BLOCKED" and not issues:
        raise ExperimentProtocolValidationError("BLOCKED protocol 必须包含 blocking issue")

    base_payload = {k: v for k, v in protocol.items() if k not in {"protocol_id", "content_sha256"}}
    base_hash = sha256_json(base_payload)
    expected_id = f"proto_{base_hash[:20]}"
    if protocol.get("protocol_id") != expected_id:
        raise ExperimentProtocolValidationError("protocol_id 校验失败")
    content_for_hash = {k: v for k, v in protocol.items() if k != "content_sha256"}
    expected_content_hash = sha256_json(content_for_hash)
    if protocol.get("content_sha256") != expected_content_hash:
        raise ExperimentProtocolValidationError("protocol content_sha256 校验失败")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


class ExperimentProtocolStore:
    """Immutable protocol persistence for T27.

    READY and BLOCKED protocols are both persisted for auditability. Persisting a
    BLOCKED protocol does not authorize execution; T28/T29 must require READY.
    """

    def __init__(self, runtime_root: str | Path = ".runtime") -> None:
        self.root = Path(runtime_root) / "v030" / "protocols"

    def protocol_dir(self, protocol_id: str) -> Path:
        return self.root / protocol_id

    def save(self, protocol: dict[str, Any]) -> dict[str, Any]:
        validate_protocol_document(protocol)
        directory = self.protocol_dir(protocol["protocol_id"])
        protocol_path = directory / "protocol.json"
        manifest_path = directory / "manifest.json"

        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
            validate_protocol_document(existing)
            if existing != protocol:
                raise ExperimentProtocolConflictError(
                    f"protocol_id 已存在但内容冲突: {protocol['protocol_id']}"
                )
            return {
                "idempotent_replay": True,
                "protocol_path": str(protocol_path),
                "manifest_path": str(manifest_path),
                "protocol": existing,
            }

        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(protocol_path, protocol)
        manifest = {
            "stage": PROTOCOL_STAGE,
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "protocol_id": protocol["protocol_id"],
            "candidate_id": protocol["candidate_id"],
            "status": protocol["status"],
            "content_sha256": protocol["content_sha256"],
            "persisted_at": utc_now_iso(),
            "execution_authorized": protocol["status"] == "READY",
        }
        _atomic_json(manifest_path, manifest)
        return {
            "idempotent_replay": False,
            "protocol_path": str(protocol_path),
            "manifest_path": str(manifest_path),
            "protocol": deepcopy(protocol),
        }

    def load(self, protocol_id: str) -> dict[str, Any]:
        path = self.protocol_dir(protocol_id) / "protocol.json"
        if not path.exists():
            raise ExperimentProtocolValidationError(
                f"protocol 不存在: {protocol_id}"
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        validate_protocol_document(doc)
        return doc
