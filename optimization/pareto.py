from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


DIRECTIONS = {"maximize", "minimize"}
THRESHOLD_OPERATORS = {">=", ">", "<=", "<"}


class MultiObjectiveError(ValueError):
    """Raised when a multi-objective definition is invalid."""


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str
    direction: str
    threshold_operator: str | None = None
    threshold_value: float | None = None
    weight: float = 1.0


def parse_objectives(data: dict[str, Any]) -> list[ObjectiveSpec]:
    if not isinstance(data, dict):
        raise MultiObjectiveError("objective spec 必须是 JSON object")

    if data.get("stage") != "V0.1.4-T16_multiobjective_spec":
        raise MultiObjectiveError(
            "stage 必须是 V0.1.4-T16_multiobjective_spec"
        )

    raw = data.get("objectives")
    if not isinstance(raw, list) or len(raw) < 2:
        raise MultiObjectiveError("T16 至少需要 2 个 objectives")

    objectives: list[ObjectiveSpec] = []
    seen: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MultiObjectiveError(
                f"objectives[{index}] 必须是 JSON object"
            )

        metric = str(item.get("metric") or "").strip()
        if not metric:
            raise MultiObjectiveError(
                f"objectives[{index}].metric 不能为空"
            )
        if metric in seen:
            raise MultiObjectiveError(f"objective 重复: {metric}")
        seen.add(metric)

        direction = str(item.get("direction") or "").strip().lower()
        if direction not in DIRECTIONS:
            raise MultiObjectiveError(
                f"{metric}.direction 必须是 maximize 或 minimize"
            )

        weight = item.get("weight", 1.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise MultiObjectiveError(
                f"{metric}.weight 必须是 >=0 的有限数值"
            )

        threshold = item.get("threshold")
        op = None
        value = None

        if threshold is not None:
            if not isinstance(threshold, dict):
                raise MultiObjectiveError(
                    f"{metric}.threshold 必须是 JSON object"
                )

            op = str(threshold.get("operator") or "").strip()
            if op not in THRESHOLD_OPERATORS:
                raise MultiObjectiveError(
                    f"{metric}.threshold.operator 非法"
                )

            raw_value = threshold.get("value")
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
            ):
                raise MultiObjectiveError(
                    f"{metric}.threshold.value 必须是有限数值"
                )

            value = float(raw_value)

        objectives.append(
            ObjectiveSpec(
                metric=metric,
                direction=direction,
                threshold_operator=op,
                threshold_value=value,
                weight=float(weight),
            )
        )

    return objectives


def threshold_pass(value: float, objective: ObjectiveSpec) -> bool:
    op = objective.threshold_operator

    if op is None:
        return True

    threshold = float(objective.threshold_value)

    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold

    raise MultiObjectiveError(f"不支持 threshold operator: {op}")


def dominates(
    left: dict[str, float],
    right: dict[str, float],
    objectives: Iterable[ObjectiveSpec],
) -> bool:
    """
    Standard Pareto dominance:
    left is no worse in every objective and strictly better in >=1.
    """
    no_worse = True
    strictly_better = False

    for objective in objectives:
        a = float(left[objective.metric])
        b = float(right[objective.metric])

        if objective.direction == "maximize":
            if a < b:
                no_worse = False
                break
            if a > b:
                strictly_better = True
        else:
            if a > b:
                no_worse = False
                break
            if a < b:
                strictly_better = True

    return no_worse and strictly_better


def pareto_front_indices(
    prediction_rows: list[dict[str, float]],
    objectives: list[ObjectiveSpec],
) -> list[int]:
    front = []

    for i, row_i in enumerate(prediction_rows):
        dominated_flag = False

        for j, row_j in enumerate(prediction_rows):
            if i == j:
                continue
            if dominates(row_j, row_i, objectives):
                dominated_flag = True
                break

        if not dominated_flag:
            front.append(i)

    return front


def non_dominated_sort(
    prediction_rows: list[dict[str, float]],
    objectives: list[ObjectiveSpec],
) -> list[int]:
    """
    Return 1-based Pareto rank for every row.
    rank=1 is the Pareto front.
    """
    remaining = list(range(len(prediction_rows)))
    ranks = [0] * len(prediction_rows)
    rank = 1

    while remaining:
        local_rows = [prediction_rows[i] for i in remaining]
        local_front = pareto_front_indices(local_rows, objectives)
        global_front = [remaining[i] for i in local_front]

        for index in global_front:
            ranks[index] = rank

        front_set = set(global_front)
        remaining = [
            index for index in remaining
            if index not in front_set
        ]
        rank += 1

    return ranks


def normalized_utilities(
    prediction_rows: list[dict[str, float]],
    objectives: list[ObjectiveSpec],
) -> list[float]:
    if not prediction_rows:
        return []

    ranges: dict[str, tuple[float, float]] = {}

    for objective in objectives:
        values = [
            float(row[objective.metric])
            for row in prediction_rows
        ]
        ranges[objective.metric] = (min(values), max(values))

    total_weight = sum(obj.weight for obj in objectives)
    if total_weight <= 0:
        total_weight = float(len(objectives))

    output = []

    for row in prediction_rows:
        score = 0.0

        for objective in objectives:
            lo, hi = ranges[objective.metric]
            value = float(row[objective.metric])

            if hi - lo <= 1e-12:
                normalized = 1.0
            elif objective.direction == "maximize":
                normalized = (value - lo) / (hi - lo)
            else:
                normalized = (hi - value) / (hi - lo)

            weight = (
                objective.weight
                if sum(obj.weight for obj in objectives) > 0
                else 1.0
            )
            score += weight * normalized

        output.append(score / total_weight)

    return output


def _standardized_feature_vectors(
    candidates: list[dict[str, Any]],
    feature_columns: list[str],
) -> list[list[float]]:
    if not candidates:
        return []

    matrix = [
        [float(candidate["features"][col]) for col in feature_columns]
        for candidate in candidates
    ]

    columns = list(zip(*matrix))
    means = [
        sum(col) / len(col)
        for col in columns
    ]
    stds = []

    for col, mean_value in zip(columns, means):
        variance = sum(
            (value - mean_value) ** 2
            for value in col
        ) / len(col)
        std = math.sqrt(variance)
        stds.append(std if std > 1e-12 else 1.0)

    return [
        [
            (value - means[j]) / stds[j]
            for j, value in enumerate(row)
        ]
        for row in matrix
    ]


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(
        sum((x - y) ** 2 for x, y in zip(a, b))
    )


def diverse_select(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    feature_columns: list[str],
    utility_key: str = "adjusted_utility",
    diversity_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Deterministic greedy diversity selection.

    First item: highest utility.
    Later items: blend utility with min distance to already selected designs.
    """
    if count <= 0 or not candidates:
        return []

    diversity_weight = min(max(float(diversity_weight), 0.0), 1.0)
    vectors = _standardized_feature_vectors(
        candidates,
        feature_columns,
    )

    utilities = [
        float(candidate[utility_key])
        for candidate in candidates
    ]

    u_min = min(utilities)
    u_max = max(utilities)

    if u_max - u_min <= 1e-12:
        norm_utilities = [1.0] * len(utilities)
    else:
        norm_utilities = [
            (value - u_min) / (u_max - u_min)
            for value in utilities
        ]

    first = max(
        range(len(candidates)),
        key=lambda i: (
            norm_utilities[i],
            str(candidates[i].get("candidate_id", "")),
        ),
    )

    selected = [first]
    remaining = {
        i for i in range(len(candidates))
        if i != first
    }

    max_possible_distance = math.sqrt(
        max(len(feature_columns), 1)
    ) * 4.0

    while remaining and len(selected) < count:
        best_index = None
        best_score = None

        for i in sorted(
            remaining,
            key=lambda x: str(
                candidates[x].get("candidate_id", "")
            ),
        ):
            min_distance = min(
                _distance(vectors[i], vectors[j])
                for j in selected
            )

            distance_score = min(
                min_distance / max_possible_distance,
                1.0,
            )

            score = (
                (1.0 - diversity_weight) * norm_utilities[i]
                + diversity_weight * distance_score
            )

            tie = (
                score,
                norm_utilities[i],
                min_distance,
                str(candidates[i].get("candidate_id", "")),
            )

            if best_score is None or tie > best_score:
                best_score = tie
                best_index = i

        selected.append(best_index)
        remaining.remove(best_index)

    return [
        candidates[i]
        for i in selected
    ]
