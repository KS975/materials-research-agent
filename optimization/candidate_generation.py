from __future__ import annotations

from collections import Counter
import json
import math
import random
from typing import Any

from .search_space import SearchSpace, VariableSpec


class CandidateGenerationError(RuntimeError):
    """Raised when a requested candidate pool cannot be generated."""


def _snap_numeric(spec: VariableSpec, value: float) -> float | int | None:
    """Snap a numeric value to the variable's grid and bounds."""
    minimum = float(spec.minimum)
    maximum = float(spec.maximum)

    if spec.step is None:
        snapped = min(max(float(value), minimum), maximum)
    else:
        step = float(spec.step)
        index = round((float(value) - minimum) / step)
        snapped = minimum + index * step

    if snapped < minimum - 1e-9 or snapped > maximum + 1e-9:
        return None

    if spec.kind == "integer":
        if abs(snapped - round(snapped)) > 1e-8:
            return None
        return int(round(snapped))

    return float(snapped)


class CandidateGenerator:
    """Constraint-aware candidate generator for a V0.1.4 SearchSpace."""

    def __init__(
        self,
        search_space: SearchSpace,
        *,
        random_state: int = 42,
        id_prefix: str = "V014_T15",
    ) -> None:
        prefix = str(id_prefix or "").strip()
        if not prefix:
            raise CandidateGenerationError("id_prefix 不能为空")
        if any(ch.isspace() for ch in prefix):
            raise CandidateGenerationError("id_prefix 不能包含空白字符")

        self.search_space = search_space
        self.random = random.Random(random_state)
        self.random_state = random_state
        self.id_prefix = prefix

    def _sample_variable(self, spec: VariableSpec) -> Any:
        if spec.kind == "categorical":
            return self.random.choice(list(spec.choices))

        minimum = float(spec.minimum)
        maximum = float(spec.maximum)

        if spec.step is None:
            value = self.random.uniform(minimum, maximum)
            return int(round(value)) if spec.kind == "integer" else float(value)

        step = float(spec.step)
        count = int(math.floor((maximum - minimum) / step + 1e-9))
        index = self.random.randint(0, count)
        value = minimum + index * step

        if spec.kind == "integer":
            return int(round(value))

        return float(value)

    def _repair_hard_equalities(self, features: dict[str, Any]) -> None:
        """
        Repair simple HARD weighted_sum == constraints.

        The repair is deliberately conservative:
        - only HARD weighted_sum equality constraints;
        - one numeric term is adjusted at a time;
        - the adjusted value must land on the declared variable grid;
        - the final equality must satisfy its tolerance.

        This improves rejection-sampling efficiency without bypassing the
        SearchSpace validator. The validator remains the final authority.
        """
        for constraint in self.search_space.constraints:
            if (
                constraint.severity != "HARD"
                or constraint.constraint_type != "weighted_sum"
            ):
                continue

            doc = constraint.payload
            if str(doc.get("operator")) != "==":
                continue

            target = float(doc["value"])
            tolerance = float(doc.get("tolerance", 0.0))
            terms = list(doc["terms"])

            repair_options = []
            for term in reversed(terms):
                variable = str(term["variable"])
                spec = self.search_space.variable_map[variable]
                weight = float(term.get("weight", 1.0))
                if (
                    spec.kind in {"continuous", "integer"}
                    and abs(weight) > 1e-12
                ):
                    repair_options.append((variable, spec, weight))

            for repair_variable, repair_spec, repair_weight in repair_options:
                other_sum = 0.0
                valid = True

                for term in terms:
                    variable = str(term["variable"])
                    weight = float(term.get("weight", 1.0))
                    if variable == repair_variable:
                        continue

                    value = features.get(variable)
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        valid = False
                        break

                    other_sum += weight * float(value)

                if not valid:
                    continue

                required = (target - other_sum) / repair_weight
                snapped = _snap_numeric(repair_spec, required)

                if snapped is None:
                    continue

                old_value = features.get(repair_variable)
                features[repair_variable] = snapped

                actual = 0.0
                for term in terms:
                    variable = str(term["variable"])
                    weight = float(term.get("weight", 1.0))
                    actual += weight * float(features[variable])

                if abs(actual - target) <= tolerance + 1e-9:
                    break

                features[repair_variable] = old_value

    @staticmethod
    def _candidate_key(features: dict[str, Any]) -> str:
        return json.dumps(
            features,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def generate(
        self,
        *,
        candidate_count: int,
        max_attempts: int,
        keep_rejected_examples: int = 20,
    ) -> dict[str, Any]:
        if candidate_count <= 0:
            raise CandidateGenerationError("candidate_count 必须 > 0")
        if max_attempts < candidate_count:
            raise CandidateGenerationError(
                "max_attempts 不能小于 candidate_count"
            )

        accepted: list[dict[str, Any]] = []
        rejected_examples: list[dict[str, Any]] = []
        rejection_counts: Counter[str] = Counter()
        seen: set[str] = set()
        duplicate_count = 0
        attempts = 0

        while attempts < max_attempts and len(accepted) < candidate_count:
            attempts += 1

            features = {
                spec.name: self._sample_variable(spec)
                for spec in self.search_space.variables
            }

            self._repair_hard_equalities(features)

            report = self.search_space.validate_candidate(
                {"features": features}
            )

            if not report["hard_valid"]:
                reasons = []

                for item in report["variable_errors"]:
                    key = (
                        f"variable::{item.get('type')}::"
                        f"{item.get('variable') or ','.join(item.get('variables', []))}"
                    )
                    rejection_counts[key] += 1
                    reasons.append(key)

                for item in report["hard_violations"]:
                    key = f"constraint::{item['constraint_id']}"
                    rejection_counts[key] += 1
                    reasons.append(key)

                if len(rejected_examples) < keep_rejected_examples:
                    rejected_examples.append(
                        {
                            "features": features,
                            "reasons": reasons,
                            "report": report,
                        }
                    )
                continue

            normalized = dict(report["normalized_candidate"])
            key = self._candidate_key(normalized)

            if key in seen:
                duplicate_count += 1
                continue

            seen.add(key)

            accepted.append(
                {
                    "candidate_id": f"{self.id_prefix}_{len(accepted) + 1:05d}",
                    "features": normalized,
                    "constraint_status": report["status"],
                    "soft_penalty": float(report["soft_penalty"]),
                    "soft_violations": report["soft_violations"],
                }
            )

        complete = len(accepted) == candidate_count

        return {
            "requested_count": candidate_count,
            "generated_count": len(accepted),
            "generation_complete": complete,
            "attempts": attempts,
            "acceptance_rate": (
                len(accepted) / attempts if attempts else 0.0
            ),
            "duplicate_count": duplicate_count,
            "rejection_counts": dict(rejection_counts),
            "rejected_examples": rejected_examples,
            "candidates": accepted,
            "random_state": self.random_state,
            "id_prefix": self.id_prefix,
        }
