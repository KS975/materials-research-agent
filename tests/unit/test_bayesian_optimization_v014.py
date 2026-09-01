import numpy as np

from optimization.bayesian_optimization import (
    BOConfig,
    GaussianProcessBayesianOptimizer,
    acquisition_values,
    filter_already_observed_candidate_indices,
)


def test_ei_is_nonnegative_and_rewards_improvement():
    mu = np.array([0.0, 1.0, 2.0])
    sigma = np.array([0.5, 0.5, 0.5])

    ei = acquisition_values(
        mu,
        sigma,
        best_score=1.0,
        acquisition="EI",
        xi=0.01,
    )

    assert np.all(ei >= 0)
    assert ei[2] > ei[1] > ei[0]


def test_pi_is_probability_like():
    mu = np.array([0.0, 1.0, 2.0])
    sigma = np.array([1.0, 1.0, 1.0])

    pi = acquisition_values(
        mu,
        sigma,
        best_score=1.0,
        acquisition="PI",
        xi=0.0,
    )

    assert np.all(pi >= 0)
    assert np.all(pi <= 1)
    assert pi[2] > pi[1] > pi[0]


def test_ucb_rewards_uncertainty():
    mu = np.array([1.0, 1.0])
    sigma = np.array([0.1, 1.0])

    ucb = acquisition_values(
        mu,
        sigma,
        best_score=1.0,
        acquisition="UCB",
        kappa=2.0,
    )

    assert ucb[1] > ucb[0]


def test_already_observed_candidate_is_removed_deterministically():
    observed = np.array(
        [
            [10.0, 20.0],
            [11.0, 21.0],
        ]
    )
    candidates = np.array(
        [
            [10.0, 20.0],  # exact repeat of completed experiment
            [12.0, 22.0],  # new
            [11.0, 21.0],  # another repeat
            [13.0, 23.0],  # new
        ]
    )

    keep, duplicates = filter_already_observed_candidate_indices(
        candidates,
        observed,
    )

    assert keep == [1, 3]
    assert duplicates == [0, 2]


def test_soft_penalty_changes_actual_bo_selection_order():
    """
    Same GP candidate geometry, but candidate 0 receives a very large
    engineering soft penalty. The optimizer must rank by adjusted
    acquisition, not raw acquisition.
    """
    X_obs = np.array(
        [[x] for x in np.linspace(0.0, 1.0, 20)],
        dtype=float,
    )
    y_obs = np.sin(4.0 * X_obs[:, 0])

    X_candidates = np.array(
        [
            [1.05],
            [1.10],
            [0.95],
        ],
        dtype=float,
    )
    ids = ["HIGH_PENALTY", "LOW_PENALTY", "OTHER"]

    config = BOConfig(
        acquisition="UCB",
        batch_size=1,
        kappa=1.0,
        min_batch_distance=0.0,
        random_state=42,
    )

    bo = GaussianProcessBayesianOptimizer(
        X_obs,
        y_obs,
        config=config,
    )

    no_penalty = bo.propose_batch(
        X_candidates,
        ids,
        candidate_penalties=[0.0, 0.0, 0.0],
        penalty_weight=0.0,
    )
    raw_winner = no_penalty["rounds"][0]["candidate_id"]

    penalties = [
        100.0 if candidate_id == raw_winner else 0.0
        for candidate_id in ids
    ]

    penalized = bo.propose_batch(
        X_candidates,
        ids,
        candidate_penalties=penalties,
        penalty_weight=1.0,
    )

    penalized_winner = penalized["rounds"][0]["candidate_id"]

    assert penalized_winner != raw_winner
    assert penalized["rounds"][0]["adjusted_acquisition"] <= (
        penalized["rounds"][0]["acquisition_value"]
    )


def test_gp_bo_returns_unique_kriging_believer_batch():
    rng = np.random.default_rng(42)

    X = rng.uniform(0, 1, size=(25, 2))
    y = (
        np.sin(4 * X[:, 0])
        + np.cos(3 * X[:, 1])
    )

    candidates = rng.uniform(
        0,
        1,
        size=(100, 2),
    )
    ids = [
        f"C{i:03d}"
        for i in range(len(candidates))
    ]

    penalties = np.linspace(
        0.0,
        0.2,
        len(candidates),
    )

    bo = GaussianProcessBayesianOptimizer(
        X,
        y,
        config=BOConfig(
            acquisition="EI",
            batch_size=5,
            min_batch_distance=0.05,
            random_state=42,
        ),
    )

    result = bo.propose_batch(
        candidates,
        ids,
        candidate_penalties=penalties,
        penalty_weight=0.5,
    )

    assert result["batch_strategy"] == "kriging_believer"
    assert len(result["rounds"]) == 5

    selected = [
        row["candidate_id"]
        for row in result["rounds"]
    ]
    assert len(set(selected)) == 5
    assert all(
        row["posterior_std"] >= 0
        for row in result["rounds"]
    )
    assert all(
        "adjusted_acquisition" in row
        for row in result["rounds"]
    )
