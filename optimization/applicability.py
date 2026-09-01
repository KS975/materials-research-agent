from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


EDGE_FRACTION = 0.05
NN_IN_QUANTILE = 0.90
NN_OUT_QUANTILE = 0.99
CENTROID_IN_QUANTILE = 0.95
CENTROID_OUT_QUANTILE = 0.99


class ApplicabilityDomainError(RuntimeError):
    """Raised when applicability-domain calibration cannot be built."""


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


class ApplicabilityDomainCalibrator:
    """
    Reusable V0.1.3-compatible applicability-domain evaluator.

    The logic mirrors T13:
    - observed per-feature min/max;
    - standardized distance to the training centroid;
    - standardized nearest-neighbor distance;
    - empirical quantile thresholds calibrated from training data.
    """

    def __init__(
        self,
        *,
        feature_columns: list[str],
        X: Any,
        dropped_rows: int,
    ) -> None:
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ApplicabilityDomainError(
                f"ML dependencies missing: {exc}"
            ) from exc

        self.np = np
        self.NearestNeighbors = NearestNeighbors
        self.feature_columns = list(feature_columns)
        self.X = np.asarray(X, dtype=float)
        self.dropped_rows = int(dropped_rows)

        if len(self.X) < 10:
            raise ApplicabilityDomainError(
                "适用域校准至少需要 10 条完整数值训练样本"
            )

        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.safe_stds = np.where(self.stds > 1e-12, self.stds, 1.0)
        self.Xz = (self.X - self.means) / self.safe_stds

        self.mins = self.X.min(axis=0)
        self.maxs = self.X.max(axis=0)

        training_centroid_distances = np.linalg.norm(self.Xz, axis=1)
        self.centroid_in_threshold = float(
            np.quantile(
                training_centroid_distances,
                CENTROID_IN_QUANTILE,
            )
        )
        self.centroid_out_threshold = float(
            np.quantile(
                training_centroid_distances,
                CENTROID_OUT_QUANTILE,
            )
        )

        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(self.Xz)
        training_distances, _ = nn.kneighbors(self.Xz)
        training_loo_nn = training_distances[:, 1]

        self.nn_in_threshold = float(
            np.quantile(training_loo_nn, NN_IN_QUANTILE)
        )
        self.nn_out_threshold = float(
            np.quantile(training_loo_nn, NN_OUT_QUANTILE)
        )

        self.sample_nn = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
        )
        self.sample_nn.fit(self.Xz)

    @classmethod
    def from_csv(
        cls,
        dataset_csv: str | Path,
        *,
        feature_columns: list[str],
    ) -> "ApplicabilityDomainCalibrator":
        path = Path(dataset_csv)

        if not path.exists():
            raise ApplicabilityDomainError(
                f"训练数据不存在: {path}"
            )

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ApplicabilityDomainError(
                    "训练 CSV 没有表头"
                )

            missing = [
                col for col in feature_columns
                if col not in reader.fieldnames
            ]
            if missing:
                raise ApplicabilityDomainError(
                    f"训练数据缺少模型特征: {missing}"
                )

            rows = list(reader)

        X_rows = []
        dropped = 0

        for row in rows:
            vector = [
                _parse_float(row.get(col))
                for col in feature_columns
            ]

            if any(value is None for value in vector):
                dropped += 1
                continue

            X_rows.append([float(value) for value in vector])

        return cls(
            feature_columns=feature_columns,
            X=X_rows,
            dropped_rows=dropped,
        )

    def evaluate(self, features: dict[str, Any]) -> dict[str, Any]:
        np = self.np

        missing = [
            col for col in self.feature_columns
            if col not in features
        ]
        if missing:
            raise ApplicabilityDomainError(
                f"候选缺少模型特征: {missing}"
            )

        values = []
        for col in self.feature_columns:
            value = _parse_float(features.get(col))
            if value is None:
                raise ApplicabilityDomainError(
                    f"候选模型特征不是有限数值: {col}"
                )
            values.append(value)

        x = np.asarray(values, dtype=float)
        xz = (x - self.means) / self.safe_stds

        range_checks = []
        outside_features = []
        edge_features = []

        for idx, feature in enumerate(self.feature_columns):
            value = float(x[idx])
            min_value = float(self.mins[idx])
            max_value = float(self.maxs[idx])
            width = max_value - min_value

            if width <= 1e-12:
                in_range = abs(value - min_value) <= 1e-12
                edge_fraction = 0.0 if in_range else None
            else:
                in_range = min_value <= value <= max_value
                edge_fraction = (
                    min(
                        (value - min_value) / width,
                        (max_value - value) / width,
                    )
                    if in_range
                    else None
                )

            if not in_range:
                outside_features.append(feature)
            elif (
                edge_fraction is not None
                and edge_fraction < EDGE_FRACTION
            ):
                edge_features.append(feature)

            range_checks.append(
                {
                    "feature": feature,
                    "value": value,
                    "train_min": min_value,
                    "train_max": max_value,
                    "in_range": bool(in_range),
                    "edge_fraction": (
                        float(edge_fraction)
                        if edge_fraction is not None
                        else None
                    ),
                }
            )

        centroid_distance = float(np.linalg.norm(xz))

        distances, indices = self.sample_nn.kneighbors(
            xz.reshape(1, -1)
        )
        nn_distance = float(distances[0, 0])
        nearest_index = int(indices[0, 0])

        reasons = []

        if outside_features:
            status = "OUT_OF_DOMAIN"
            risk = "HIGH"
            reasons.extend(
                f"{feature} 超出训练数据单特征范围"
                for feature in outside_features
            )
        elif (
            nn_distance > self.nn_out_threshold
            or centroid_distance > self.centroid_out_threshold
        ):
            status = "OUT_OF_DOMAIN"
            risk = "HIGH"

            if nn_distance > self.nn_out_threshold:
                reasons.append(
                    "与最近训练样本的标准化距离超过 99% 训练近邻距离阈值"
                )

            if centroid_distance > self.centroid_out_threshold:
                reasons.append(
                    "到训练分布中心的标准化距离超过 99% 训练分布阈值"
                )
        elif (
            edge_features
            or nn_distance > self.nn_in_threshold
            or centroid_distance > self.centroid_in_threshold
        ):
            status = "BORDERLINE"
            risk = "MEDIUM"

            if edge_features:
                reasons.append(
                    "部分特征处于训练范围边缘 5% 区域："
                    + ", ".join(edge_features)
                )

            if nn_distance > self.nn_in_threshold:
                reasons.append(
                    "与最近训练样本距离高于训练近邻距离的 90% 分位阈值"
                )

            if centroid_distance > self.centroid_in_threshold:
                reasons.append(
                    "到训练分布中心距离高于训练样本的 95% 分位阈值"
                )
        else:
            status = "IN_DOMAIN"
            risk = "LOW"
            reasons.append(
                "所有模型特征均位于训练数据主要覆盖区域"
            )

        return {
            "status": status,
            "risk": risk,
            "reasons": reasons,
            "outside_features": outside_features,
            "edge_features": edge_features,
            "range_checks": range_checks,
            "nearest_neighbor_distance": nn_distance,
            "nearest_neighbor_in_threshold": self.nn_in_threshold,
            "nearest_neighbor_out_threshold": self.nn_out_threshold,
            "nearest_training_row_index": nearest_index,
            "centroid_distance": centroid_distance,
            "centroid_in_threshold": self.centroid_in_threshold,
            "centroid_out_threshold": self.centroid_out_threshold,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "usable_training_rows": int(len(self.X)),
            "dropped_training_rows": self.dropped_rows,
            "feature_columns": self.feature_columns,
            "edge_fraction_threshold": EDGE_FRACTION,
            "nearest_neighbor": {
                "in_quantile": NN_IN_QUANTILE,
                "out_quantile": NN_OUT_QUANTILE,
                "in_threshold": self.nn_in_threshold,
                "out_threshold": self.nn_out_threshold,
            },
            "centroid_distance": {
                "in_quantile": CENTROID_IN_QUANTILE,
                "out_quantile": CENTROID_OUT_QUANTILE,
                "in_threshold": self.centroid_in_threshold,
                "out_threshold": self.centroid_out_threshold,
            },
        }
