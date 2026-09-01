from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


ACQUISITIONS = {"EI", "PI", "UCB"}
DIRECTIONS = {"maximize", "minimize"}


class BayesianOptimizationError(RuntimeError):
    """Raised when the Bayesian optimization workflow is invalid."""


@dataclass(frozen=True)
class BOConfig:
    acquisition: str = "EI"
    direction: str = "maximize"
    xi: float = 0.01
    kappa: float = 2.0
    batch_size: int = 5
    min_batch_distance: float = 0.20
    random_state: int = 42


def _validate_config(config: BOConfig) -> BOConfig:
    acquisition = str(config.acquisition).upper().strip()
    if acquisition not in ACQUISITIONS:
        raise BayesianOptimizationError(
            f"acquisition 必须是 {sorted(ACQUISITIONS)}"
        )

    direction = str(config.direction).lower().strip()
    if direction not in DIRECTIONS:
        raise BayesianOptimizationError(
            "direction 必须是 maximize 或 minimize"
        )

    if config.batch_size <= 0:
        raise BayesianOptimizationError("batch_size 必须 > 0")

    if config.min_batch_distance < 0:
        raise BayesianOptimizationError(
            "min_batch_distance 不能小于 0"
        )

    if not math.isfinite(float(config.xi)) or config.xi < 0:
        raise BayesianOptimizationError("xi 必须是 >=0 的有限数值")

    if not math.isfinite(float(config.kappa)) or config.kappa < 0:
        raise BayesianOptimizationError(
            "kappa 必须是 >=0 的有限数值"
        )

    return BOConfig(
        acquisition=acquisition,
        direction=direction,
        xi=float(config.xi),
        kappa=float(config.kappa),
        batch_size=int(config.batch_size),
        min_batch_distance=float(config.min_batch_distance),
        random_state=int(config.random_state),
    )


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    flat = np.asarray(z, dtype=float).ravel()
    values = [
        0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
        for value in flat
    ]
    return np.asarray(values, dtype=float).reshape(np.asarray(z).shape)


def acquisition_values(
    mean_score: np.ndarray,
    std_score: np.ndarray,
    *,
    best_score: float,
    acquisition: str,
    xi: float = 0.01,
    kappa: float = 2.0,
) -> np.ndarray:
    """
    Compute maximize-oriented acquisition values.

    mean_score is already transformed so that "higher is better".
    """
    mu = np.asarray(mean_score, dtype=float)
    sigma = np.maximum(np.asarray(std_score, dtype=float), 0.0)
    mode = str(acquisition).upper().strip()

    if mode == "UCB":
        return mu + float(kappa) * sigma

    improvement = mu - float(best_score) - float(xi)

    safe_sigma = np.where(sigma > 1e-12, sigma, 1.0)
    z = improvement / safe_sigma

    if mode == "PI":
        pi = _normal_cdf(z)
        return np.where(sigma > 1e-12, pi, (improvement > 0).astype(float))

    if mode == "EI":
        ei = (
            improvement * _normal_cdf(z)
            + sigma * _normal_pdf(z)
        )
        return np.where(
            sigma > 1e-12,
            np.maximum(ei, 0.0),
            np.maximum(improvement, 0.0),
        )

    raise BayesianOptimizationError(
        f"不支持 acquisition={acquisition!r}"
    )


def _standardized_matrix(
    X: np.ndarray,
    *,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    safe_stds = np.where(stds > 1e-12, stds, 1.0)
    return (X - means) / safe_stds


def _min_distance_to_selected(
    point: np.ndarray,
    selected: list[np.ndarray],
) -> float:
    if not selected:
        return math.inf
    return min(
        float(np.linalg.norm(point - other))
        for other in selected
    )


def feature_key_from_vector(values: np.ndarray | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Stable numeric key used to prevent re-proposing already observed experiments."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise BayesianOptimizationError(
            "feature key contains non-finite values"
        )
    return tuple(round(float(value), 10) for value in array)


def observed_feature_key_set(X_observed: np.ndarray) -> set[tuple[float, ...]]:
    X = np.asarray(X_observed, dtype=float)
    if X.ndim != 2:
        raise BayesianOptimizationError(
            "X_observed 必须是二维矩阵"
        )
    return {
        feature_key_from_vector(row)
        for row in X
    }


def filter_already_observed_candidate_indices(
    X_candidates: np.ndarray,
    X_observed: np.ndarray,
) -> tuple[list[int], list[int]]:
    """
    Return (keep_indices, duplicate_indices).

    Exact comparison is performed on stable rounded numeric feature keys.
    This helper makes the "never repeat an already completed experiment"
    rule independently testable.
    """
    Xc = np.asarray(X_candidates, dtype=float)
    Xo = np.asarray(X_observed, dtype=float)

    if Xc.ndim != 2 or Xo.ndim != 2:
        raise BayesianOptimizationError(
            "candidate/observed matrices must both be 2-D"
        )
    if Xc.shape[1] != Xo.shape[1]:
        raise BayesianOptimizationError(
            "candidate/observed feature counts differ"
        )

    observed_keys = observed_feature_key_set(Xo)

    keep: list[int] = []
    duplicates: list[int] = []

    for index, row in enumerate(Xc):
        if feature_key_from_vector(row) in observed_keys:
            duplicates.append(index)
        else:
            keep.append(index)

    return keep, duplicates


class GaussianProcessBayesianOptimizer:
    """
    Single-objective GP Bayesian Optimization for V0.1.4-T18.

    Batch selection uses Kriging Believer:
    - fit GP on observed experiments;
    - select highest acquisition candidate;
    - append its posterior mean as a fantasy observation;
    - refit and choose the next point;
    - repeat until batch_size is reached.

    This is a real sequential surrogate update, not just "top-5 acquisition".
    """

    def __init__(
        self,
        X_observed: np.ndarray,
        y_observed: np.ndarray,
        *,
        config: BOConfig,
    ) -> None:
        self.config = _validate_config(config)

        X = np.asarray(X_observed, dtype=float)
        y = np.asarray(y_observed, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise BayesianOptimizationError(
                "X_observed 必须是二维矩阵"
            )
        if len(X) != len(y):
            raise BayesianOptimizationError(
                "X_observed 与 y_observed 行数不一致"
            )
        if len(X) < 10:
            raise BayesianOptimizationError(
                "Bayesian Optimization 至少需要 10 条历史实验"
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise BayesianOptimizationError(
                "历史实验包含非有限数值"
            )

        self.X_observed = X
        self.y_observed = y
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0)
        self.safe_feature_stds = np.where(
            self.feature_stds > 1e-12,
            self.feature_stds,
            1.0,
        )

        self.Xz_observed = _standardized_matrix(
            X,
            means=self.feature_means,
            stds=self.safe_feature_stds,
        )

        self.score_observed = (
            y.copy()
            if self.config.direction == "maximize"
            else -y
        )

    def _fit_gp(
        self,
        Xz: np.ndarray,
        score: np.ndarray,
    ):
        try:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                ConstantKernel,
                Matern,
                WhiteKernel,
            )
        except ImportError as exc:
            raise BayesianOptimizationError(
                f"缺少 scikit-learn Gaussian Process 依赖: {exc}"
            ) from exc

        import warnings

        feature_count = Xz.shape[1]

        kernel = (
            ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(0.05, 20.0),
            )
            * Matern(
                length_scale=[1.0] * feature_count,
                length_scale_bounds=(0.10, 20.0),
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=0.03,
                noise_level_bounds=(1e-5, 2.0),
            )
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.config.random_state,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp.fit(Xz, score)

        return gp

    def fit_summary(self) -> dict[str, Any]:
        gp = self._fit_gp(
            self.Xz_observed,
            self.score_observed,
        )

        return {
            "observed_rows": int(len(self.X_observed)),
            "feature_count": int(self.X_observed.shape[1]),
            "direction": self.config.direction,
            "best_observed": (
                float(np.max(self.y_observed))
                if self.config.direction == "maximize"
                else float(np.min(self.y_observed))
            ),
            "fitted_kernel": str(gp.kernel_),
            "log_marginal_likelihood": float(
                gp.log_marginal_likelihood(gp.kernel_.theta)
            ),
        }

    def propose_batch(
        self,
        X_candidates: np.ndarray,
        candidate_ids: list[str],
        *,
        candidate_penalties: np.ndarray | list[float] | None = None,
        penalty_weight: float = 0.0,
    ) -> dict[str, Any]:
        Xc = np.asarray(X_candidates, dtype=float)

        if Xc.ndim != 2:
            raise BayesianOptimizationError(
                "X_candidates 必须是二维矩阵"
            )
        if Xc.shape[1] != self.X_observed.shape[1]:
            raise BayesianOptimizationError(
                "候选特征数与历史实验不一致"
            )
        if len(candidate_ids) != len(Xc):
            raise BayesianOptimizationError(
                "candidate_ids 与 X_candidates 行数不一致"
            )
        if not np.isfinite(Xc).all():
            raise BayesianOptimizationError(
                "候选包含非有限数值"
            )
        if len(Xc) < self.config.batch_size:
            raise BayesianOptimizationError(
                "候选数量少于 batch_size"
            )

        if not math.isfinite(float(penalty_weight)) or float(penalty_weight) < 0:
            raise BayesianOptimizationError(
                "penalty_weight 必须是 >=0 的有限数值"
            )

        if candidate_penalties is None:
            penalties = np.zeros(len(Xc), dtype=float)
        else:
            penalties = np.asarray(candidate_penalties, dtype=float).reshape(-1)
            if len(penalties) != len(Xc):
                raise BayesianOptimizationError(
                    "candidate_penalties 与 X_candidates 行数不一致"
                )
            if not np.isfinite(penalties).all() or np.any(penalties < 0):
                raise BayesianOptimizationError(
                    "candidate_penalties 必须是 >=0 的有限数值"
                )

        penalty_weight = float(penalty_weight)

        Xcz = _standardized_matrix(
            Xc,
            means=self.feature_means,
            stds=self.safe_feature_stds,
        )

        active = list(range(len(Xc)))
        selected_indices: list[int] = []
        selected_z: list[np.ndarray] = []

        fantasy_Xz = self.Xz_observed.copy()
        fantasy_score = self.score_observed.copy()

        rounds: list[dict[str, Any]] = []

        for round_index in range(1, self.config.batch_size + 1):
            gp = self._fit_gp(
                fantasy_Xz,
                fantasy_score,
            )

            if not active:
                break

            active_array = np.asarray(active, dtype=int)
            X_active = Xcz[active_array]

            mean_score, std_score = gp.predict(
                X_active,
                return_std=True,
            )

            best_score = float(np.max(fantasy_score))

            acq = acquisition_values(
                mean_score,
                std_score,
                best_score=best_score,
                acquisition=self.config.acquisition,
                xi=self.config.xi,
                kappa=self.config.kappa,
            )

            active_penalties = penalties[active_array]
            adjusted_acq = (
                acq
                - penalty_weight * active_penalties
            )

            ranked_local = sorted(
                range(len(active)),
                key=lambda local_idx: (
                    float(adjusted_acq[local_idx]),
                    float(acq[local_idx]),
                    float(std_score[local_idx]),
                    candidate_ids[active[local_idx]],
                ),
                reverse=True,
            )

            chosen_local = None

            for local_idx in ranked_local:
                global_idx = active[local_idx]

                distance = _min_distance_to_selected(
                    Xcz[global_idx],
                    selected_z,
                )

                if (
                    not selected_z
                    or distance >= self.config.min_batch_distance
                ):
                    chosen_local = local_idx
                    break

            # If diversity threshold is too strict, do not fail the whole
            # experiment planning run. Pick the highest acquisition remaining.
            diversity_relaxed = False
            if chosen_local is None:
                chosen_local = ranked_local[0]
                diversity_relaxed = True

            chosen_global = active[chosen_local]
            chosen_mu_score = float(mean_score[chosen_local])
            chosen_std = float(std_score[chosen_local])
            chosen_acq = float(acq[chosen_local])
            chosen_adjusted_acq = float(
                adjusted_acq[chosen_local]
            )
            chosen_penalty = float(
                active_penalties[chosen_local]
            )

            chosen_objective_mean = (
                chosen_mu_score
                if self.config.direction == "maximize"
                else -chosen_mu_score
            )

            distance = _min_distance_to_selected(
                Xcz[chosen_global],
                selected_z,
            )

            rounds.append(
                {
                    "round": round_index,
                    "candidate_index": int(chosen_global),
                    "candidate_id": candidate_ids[chosen_global],
                    "posterior_mean": float(chosen_objective_mean),
                    "posterior_std": chosen_std,
                    "acquisition_value": chosen_acq,
                    "candidate_penalty": chosen_penalty,
                    "penalty_weight": penalty_weight,
                    "adjusted_acquisition": chosen_adjusted_acq,
                    "best_score_before_round": float(best_score),
                    "min_standardized_distance_to_selected": (
                        None if math.isinf(distance) else float(distance)
                    ),
                    "diversity_threshold_relaxed": diversity_relaxed,
                    "fitted_kernel": str(gp.kernel_),
                }
            )

            selected_indices.append(chosen_global)
            selected_z.append(Xcz[chosen_global].copy())

            # Kriging Believer fantasy = current posterior mean in score space.
            fantasy_Xz = np.vstack(
                [fantasy_Xz, Xcz[chosen_global]]
            )
            fantasy_score = np.concatenate(
                [fantasy_score, [chosen_mu_score]]
            )

            active.remove(chosen_global)

        if len(selected_indices) != self.config.batch_size:
            raise BayesianOptimizationError(
                "无法生成完整 BO batch"
            )

        return {
            "batch_strategy": "kriging_believer",
            "acquisition": self.config.acquisition,
            "direction": self.config.direction,
            "xi": self.config.xi,
            "kappa": self.config.kappa,
            "batch_size": self.config.batch_size,
            "min_batch_distance": self.config.min_batch_distance,
            "penalty_weight": penalty_weight,
            "selection_score": (
                "adjusted_acquisition = raw_acquisition "
                "- penalty_weight * candidate_penalty"
            ),
            "selected_indices": selected_indices,
            "rounds": rounds,
        }
