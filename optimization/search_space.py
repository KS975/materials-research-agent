from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


VARIABLE_KINDS = {"continuous", "integer", "categorical"}
SEVERITIES = {"HARD", "SOFT"}
OPERATORS = {"==", "!=", "<", "<=", ">", ">=", "in", "not_in"}
CONSTRAINT_TYPES = {
    "scalar",
    "weighted_sum",
    "forbidden_combination",
}


class SearchSpaceError(ValueError):
    """Raised when a V0.1.4 search-space definition is invalid."""


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()
    unit: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class ConstraintSpec:
    constraint_id: str
    constraint_type: str
    severity: str
    payload: dict[str, Any]
    weight: float = 1.0
    message: str | None = None


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SearchSpaceError(f"{path} 必须是 JSON object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SearchSpaceError(f"{path} 必须是 JSON array")
    return value


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_number(value: Any, path: str) -> float:
    if not _is_number(value):
        raise SearchSpaceError(f"{path} 必须是有限数值")
    return float(value)


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "in":
        if not isinstance(right, (list, tuple, set)):
            raise SearchSpaceError("'in' 的右值必须是列表")
        return left in right
    if operator == "not_in":
        if not isinstance(right, (list, tuple, set)):
            raise SearchSpaceError("'not_in' 的右值必须是列表")
        return left not in right

    if not _is_number(left) or not _is_number(right):
        return False

    a = float(left)
    b = float(right)

    if operator == "<":
        return a < b
    if operator == "<=":
        return a <= b
    if operator == ">":
        return a > b
    if operator == ">=":
        return a >= b

    raise SearchSpaceError(f"不支持的 operator: {operator}")


def _numeric_violation(left: float, operator: str, right: float, tolerance: float = 0.0) -> float:
    """Return absolute violation magnitude. 0 means satisfied."""
    if operator == "<=":
        return max(0.0, left - right)
    if operator == "<":
        return max(0.0, left - right + 1e-12)
    if operator == ">=":
        return max(0.0, right - left)
    if operator == ">":
        return max(0.0, right - left + 1e-12)
    if operator == "==":
        return max(0.0, abs(left - right) - tolerance)
    if operator == "!=":
        return 1.0 if abs(left - right) <= tolerance else 0.0
    raise SearchSpaceError(
        f"数值约束不支持 operator={operator!r}"
    )


def _normalized_penalty(magnitude: float, reference: float, weight: float) -> float:
    scale = max(abs(reference), 1.0)
    return float(weight) * (float(magnitude) / scale)


class SearchSpace:
    def __init__(
        self,
        *,
        stage: str,
        project_id: int | None,
        name: str,
        variables: list[VariableSpec],
        constraints: list[ConstraintSpec],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.project_id = project_id
        self.name = name
        self.variables = tuple(variables)
        self.constraints = tuple(constraints)
        self.metadata = dict(metadata or {})
        self.variable_map = {item.name: item for item in variables}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchSpace":
        root = _require_mapping(data, "search_space")

        stage = str(root.get("stage") or "").strip()
        if stage != "V0.1.4-T14_search_space":
            raise SearchSpaceError(
                "stage 必须是 'V0.1.4-T14_search_space'"
            )

        project_id_raw = root.get("project_id")
        project_id = None
        if project_id_raw is not None:
            if isinstance(project_id_raw, bool) or not isinstance(project_id_raw, int):
                raise SearchSpaceError("project_id 必须是整数或 null")
            project_id = project_id_raw

        name = str(root.get("name") or "").strip()
        if not name:
            raise SearchSpaceError("name 不能为空")

        variable_docs = _require_list(root.get("variables"), "variables")
        if not variable_docs:
            raise SearchSpaceError("variables 不能为空")

        variables: list[VariableSpec] = []
        seen_names: set[str] = set()

        for index, raw in enumerate(variable_docs):
            doc = _require_mapping(raw, f"variables[{index}]")
            var_name = str(doc.get("name") or "").strip()

            if not var_name:
                raise SearchSpaceError(f"variables[{index}].name 不能为空")
            if var_name in seen_names:
                raise SearchSpaceError(f"变量重复: {var_name}")
            seen_names.add(var_name)

            kind = str(doc.get("kind") or "").strip().lower()
            if kind not in VARIABLE_KINDS:
                raise SearchSpaceError(
                    f"{var_name}.kind 必须是 {sorted(VARIABLE_KINDS)}"
                )

            unit = doc.get("unit")
            if unit is not None:
                unit = str(unit)

            role = doc.get("role")
            if role is not None:
                role = str(role)

            if kind in {"continuous", "integer"}:
                minimum = _require_number(doc.get("min"), f"{var_name}.min")
                maximum = _require_number(doc.get("max"), f"{var_name}.max")

                if minimum > maximum:
                    raise SearchSpaceError(
                        f"{var_name}: min 不能大于 max"
                    )

                if kind == "integer":
                    if not float(minimum).is_integer() or not float(maximum).is_integer():
                        raise SearchSpaceError(
                            f"{var_name}: integer 变量 min/max 必须是整数"
                        )
                    minimum = int(minimum)
                    maximum = int(maximum)

                step = doc.get("step")
                if step is not None:
                    step_value = _require_number(step, f"{var_name}.step")
                    if step_value <= 0:
                        raise SearchSpaceError(
                            f"{var_name}: step 必须 > 0"
                        )
                    if kind == "integer":
                        if not step_value.is_integer():
                            raise SearchSpaceError(
                                f"{var_name}: integer step 必须是整数"
                            )
                        step = int(step_value)
                    else:
                        step = float(step_value)

                variables.append(
                    VariableSpec(
                        name=var_name,
                        kind=kind,
                        minimum=minimum,
                        maximum=maximum,
                        step=step,
                        unit=unit,
                        role=role,
                    )
                )
            else:
                choices = _require_list(doc.get("choices"), f"{var_name}.choices")
                if not choices:
                    raise SearchSpaceError(
                        f"{var_name}.choices 不能为空"
                    )
                if len({jsonable_key(v) for v in choices}) != len(choices):
                    raise SearchSpaceError(
                        f"{var_name}.choices 不能包含重复值"
                    )

                variables.append(
                    VariableSpec(
                        name=var_name,
                        kind=kind,
                        choices=tuple(choices),
                        unit=unit,
                        role=role,
                    )
                )

        constraints_docs = root.get("constraints", [])
        constraints_docs = _require_list(constraints_docs, "constraints")

        constraints: list[ConstraintSpec] = []
        seen_ids: set[str] = set()

        for index, raw in enumerate(constraints_docs):
            doc = _require_mapping(raw, f"constraints[{index}]")
            cid = str(doc.get("id") or "").strip()

            if not cid:
                raise SearchSpaceError(
                    f"constraints[{index}].id 不能为空"
                )
            if cid in seen_ids:
                raise SearchSpaceError(f"约束 id 重复: {cid}")
            seen_ids.add(cid)

            ctype = str(doc.get("type") or "").strip()
            if ctype not in CONSTRAINT_TYPES:
                raise SearchSpaceError(
                    f"{cid}.type 必须是 {sorted(CONSTRAINT_TYPES)}"
                )

            severity = str(doc.get("severity") or "").strip().upper()
            if severity not in SEVERITIES:
                raise SearchSpaceError(
                    f"{cid}.severity 必须是 HARD 或 SOFT"
                )

            weight = doc.get("weight", 1.0)
            weight = _require_number(weight, f"{cid}.weight")
            if weight < 0:
                raise SearchSpaceError(
                    f"{cid}.weight 不能小于 0"
                )

            payload = dict(doc)
            cls._validate_constraint_payload(
                cid,
                ctype,
                payload,
                seen_names,
            )

            constraints.append(
                ConstraintSpec(
                    constraint_id=cid,
                    constraint_type=ctype,
                    severity=severity,
                    payload=payload,
                    weight=weight,
                    message=(
                        str(doc["message"])
                        if doc.get("message") is not None
                        else None
                    ),
                )
            )

        metadata = root.get("metadata") or {}
        metadata = _require_mapping(metadata, "metadata")

        return cls(
            stage=stage,
            project_id=project_id,
            name=name,
            variables=variables,
            constraints=constraints,
            metadata=metadata,
        )

    @staticmethod
    def _validate_constraint_payload(
        cid: str,
        ctype: str,
        doc: dict[str, Any],
        variable_names: set[str],
    ) -> None:
        if ctype == "scalar":
            variable = str(doc.get("variable") or "").strip()
            if variable not in variable_names:
                raise SearchSpaceError(
                    f"{cid}: 未知变量 {variable!r}"
                )
            operator = str(doc.get("operator") or "").strip()
            if operator not in OPERATORS:
                raise SearchSpaceError(
                    f"{cid}: operator 非法"
                )
            if "value" not in doc:
                raise SearchSpaceError(
                    f"{cid}: 缺少 value"
                )
            return

        if ctype == "weighted_sum":
            operator = str(doc.get("operator") or "").strip()
            if operator not in {"==", "<=", ">=", "<", ">"}:
                raise SearchSpaceError(
                    f"{cid}: weighted_sum operator 仅支持 ==, <=, >=, <, >"
                )

            _require_number(doc.get("value"), f"{cid}.value")

            tolerance = doc.get("tolerance", 0.0)
            tolerance = _require_number(tolerance, f"{cid}.tolerance")
            if tolerance < 0:
                raise SearchSpaceError(
                    f"{cid}.tolerance 不能小于 0"
                )

            terms = _require_list(doc.get("terms"), f"{cid}.terms")
            if not terms:
                raise SearchSpaceError(
                    f"{cid}.terms 不能为空"
                )

            seen_terms = set()
            for term_index, raw_term in enumerate(terms):
                term = _require_mapping(
                    raw_term,
                    f"{cid}.terms[{term_index}]",
                )
                variable = str(term.get("variable") or "").strip()
                if variable not in variable_names:
                    raise SearchSpaceError(
                        f"{cid}: 未知变量 {variable!r}"
                    )
                if variable in seen_terms:
                    raise SearchSpaceError(
                        f"{cid}: terms 中变量重复 {variable}"
                    )
                seen_terms.add(variable)
                _require_number(
                    term.get("weight", 1.0),
                    f"{cid}.terms[{term_index}].weight",
                )
            return

        if ctype == "forbidden_combination":
            clauses = _require_list(
                doc.get("clauses"),
                f"{cid}.clauses",
            )
            if len(clauses) < 2:
                raise SearchSpaceError(
                    f"{cid}: forbidden_combination 至少需要 2 个 clauses"
                )

            for clause_index, raw_clause in enumerate(clauses):
                clause = _require_mapping(
                    raw_clause,
                    f"{cid}.clauses[{clause_index}]",
                )
                variable = str(clause.get("variable") or "").strip()
                if variable not in variable_names:
                    raise SearchSpaceError(
                        f"{cid}: 未知变量 {variable!r}"
                    )
                operator = str(clause.get("operator") or "").strip()
                if operator not in OPERATORS:
                    raise SearchSpaceError(
                        f"{cid}: clause operator 非法"
                    )
                if "value" not in clause:
                    raise SearchSpaceError(
                        f"{cid}: clause 缺少 value"
                    )
            return

        raise SearchSpaceError(f"不支持的 constraint type: {ctype}")

    def validate_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise SearchSpaceError("candidate 必须是 JSON object")

        candidate_name = str(candidate.get("sample_name") or "").strip() or None
        raw_features = candidate.get("features", candidate)
        raw_features = _require_mapping(raw_features, "candidate.features")

        expected = set(self.variable_map)
        provided = set(raw_features)
        allowed_meta = {"sample_name", "notes", "metadata"}
        provided_feature_like = {
            key for key in provided
            if key not in allowed_meta
        }

        missing = sorted(expected - provided_feature_like)
        unknown = sorted(provided_feature_like - expected)

        variable_errors: list[dict[str, Any]] = []
        normalized: dict[str, Any] = {}

        if missing:
            variable_errors.append(
                {
                    "type": "missing_variables",
                    "variables": missing,
                    "message": "候选缺少搜索空间要求的变量",
                }
            )

        if unknown:
            variable_errors.append(
                {
                    "type": "unknown_variables",
                    "variables": unknown,
                    "message": "候选包含搜索空间未定义变量",
                }
            )

        for spec in self.variables:
            if spec.name not in raw_features:
                continue

            value = raw_features[spec.name]

            if spec.kind == "categorical":
                if value not in spec.choices:
                    variable_errors.append(
                        {
                            "type": "invalid_choice",
                            "variable": spec.name,
                            "value": value,
                            "choices": list(spec.choices),
                            "message": "类别变量取值不在允许 choices 中",
                        }
                    )
                else:
                    normalized[spec.name] = value
                continue

            if not _is_number(value):
                variable_errors.append(
                    {
                        "type": "non_numeric",
                        "variable": spec.name,
                        "value": value,
                        "message": "数值变量必须是有限数值",
                    }
                )
                continue

            numeric = float(value)

            if spec.kind == "integer" and not numeric.is_integer():
                variable_errors.append(
                    {
                        "type": "not_integer",
                        "variable": spec.name,
                        "value": value,
                        "message": "integer 变量必须为整数",
                    }
                )
                continue

            if numeric < float(spec.minimum) or numeric > float(spec.maximum):
                variable_errors.append(
                    {
                        "type": "out_of_bounds",
                        "variable": spec.name,
                        "value": numeric,
                        "min": spec.minimum,
                        "max": spec.maximum,
                        "message": "变量超出搜索空间范围",
                    }
                )
                continue

            if spec.step is not None:
                origin = float(spec.minimum)
                step = float(spec.step)
                quotient = (numeric - origin) / step
                if abs(quotient - round(quotient)) > 1e-8:
                    variable_errors.append(
                        {
                            "type": "off_step",
                            "variable": spec.name,
                            "value": numeric,
                            "step": spec.step,
                            "origin": spec.minimum,
                            "message": "变量不在允许的 step 网格上",
                        }
                    )
                    continue

            normalized[spec.name] = (
                int(numeric)
                if spec.kind == "integer"
                else numeric
            )

        hard_violations: list[dict[str, Any]] = []
        soft_violations: list[dict[str, Any]] = []
        soft_penalty = 0.0

        # Constraint evaluation only makes sense after variable validation.
        if not variable_errors:
            for constraint in self.constraints:
                violation = self._evaluate_constraint(
                    constraint,
                    normalized,
                )
                if violation is None:
                    continue

                if constraint.severity == "HARD":
                    hard_violations.append(violation)
                else:
                    soft_violations.append(violation)
                    soft_penalty += float(violation["penalty"])

        hard_valid = not variable_errors and not hard_violations

        if not hard_valid:
            status = "INVALID"
        elif soft_violations:
            status = "VALID_WITH_SOFT_PENALTY"
        else:
            status = "VALID"

        return {
            "sample_name": candidate_name,
            "status": status,
            "hard_valid": hard_valid,
            "soft_penalty": float(soft_penalty),
            "normalized_candidate": normalized,
            "variable_errors": variable_errors,
            "hard_violations": hard_violations,
            "soft_violations": soft_violations,
        }

    def _evaluate_constraint(
        self,
        constraint: ConstraintSpec,
        candidate: dict[str, Any],
    ) -> dict[str, Any] | None:
        doc = constraint.payload
        ctype = constraint.constraint_type
        cid = constraint.constraint_id

        if ctype == "scalar":
            variable = str(doc["variable"])
            operator = str(doc["operator"])
            expected = doc["value"]
            actual = candidate[variable]

            satisfied = _compare(actual, operator, expected)
            if satisfied:
                return None

            magnitude = 1.0
            if _is_number(actual) and _is_number(expected) and operator not in {"in", "not_in"}:
                magnitude = _numeric_violation(
                    float(actual),
                    operator,
                    float(expected),
                    float(doc.get("tolerance", 0.0)),
                )

            penalty = (
                _normalized_penalty(
                    magnitude,
                    float(expected) if _is_number(expected) else 1.0,
                    constraint.weight,
                )
                if constraint.severity == "SOFT"
                else 0.0
            )

            return {
                "constraint_id": cid,
                "type": ctype,
                "severity": constraint.severity,
                "message": constraint.message
                or f"{variable} 不满足 {operator} {expected}",
                "actual": actual,
                "operator": operator,
                "expected": expected,
                "violation_magnitude": float(magnitude),
                "penalty": float(penalty),
            }

        if ctype == "weighted_sum":
            terms = doc["terms"]
            actual_sum = 0.0

            for term in terms:
                variable = str(term["variable"])
                weight = float(term.get("weight", 1.0))
                value = candidate[variable]
                if not _is_number(value):
                    raise SearchSpaceError(
                        f"{cid}: weighted_sum 变量 {variable} 不是数值"
                    )
                actual_sum += weight * float(value)

            operator = str(doc["operator"])
            expected = float(doc["value"])
            tolerance = float(doc.get("tolerance", 0.0))
            magnitude = _numeric_violation(
                actual_sum,
                operator,
                expected,
                tolerance,
            )

            if magnitude <= 0:
                return None

            penalty = (
                _normalized_penalty(
                    magnitude,
                    expected,
                    constraint.weight,
                )
                if constraint.severity == "SOFT"
                else 0.0
            )

            return {
                "constraint_id": cid,
                "type": ctype,
                "severity": constraint.severity,
                "message": constraint.message
                or f"加权和不满足 {operator} {expected}",
                "actual": float(actual_sum),
                "operator": operator,
                "expected": expected,
                "tolerance": tolerance,
                "violation_magnitude": float(magnitude),
                "penalty": float(penalty),
            }

        if ctype == "forbidden_combination":
            matched_clauses = []

            for clause in doc["clauses"]:
                variable = str(clause["variable"])
                operator = str(clause["operator"])
                expected = clause["value"]
                actual = candidate[variable]

                if _compare(actual, operator, expected):
                    matched_clauses.append(
                        {
                            "variable": variable,
                            "actual": actual,
                            "operator": operator,
                            "expected": expected,
                        }
                    )
                else:
                    return None

            penalty = (
                float(constraint.weight)
                if constraint.severity == "SOFT"
                else 0.0
            )

            return {
                "constraint_id": cid,
                "type": ctype,
                "severity": constraint.severity,
                "message": constraint.message
                or "命中禁止组合",
                "matched_clauses": matched_clauses,
                "violation_magnitude": 1.0,
                "penalty": penalty,
            }

        raise SearchSpaceError(
            f"不支持的 constraint type: {ctype}"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "project_id": self.project_id,
            "name": self.name,
            "variable_count": len(self.variables),
            "continuous_count": sum(v.kind == "continuous" for v in self.variables),
            "integer_count": sum(v.kind == "integer" for v in self.variables),
            "categorical_count": sum(v.kind == "categorical" for v in self.variables),
            "constraint_count": len(self.constraints),
            "hard_constraint_count": sum(c.severity == "HARD" for c in self.constraints),
            "soft_constraint_count": sum(c.severity == "SOFT" for c in self.constraints),
            "variables": [
                {
                    "name": v.name,
                    "kind": v.kind,
                    "min": v.minimum,
                    "max": v.maximum,
                    "step": v.step,
                    "choices": list(v.choices),
                    "unit": v.unit,
                    "role": v.role,
                }
                for v in self.variables
            ],
            "constraints": [
                {
                    "id": c.constraint_id,
                    "type": c.constraint_type,
                    "severity": c.severity,
                    "weight": c.weight,
                    "message": c.message,
                }
                for c in self.constraints
            ],
        }


def jsonable_key(value: Any) -> str:
    try:
        import json
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def load_search_space(data: dict[str, Any]) -> SearchSpace:
    return SearchSpace.from_dict(data)
