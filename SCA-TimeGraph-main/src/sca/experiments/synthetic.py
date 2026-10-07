"""Synthetic dynamic causal processes with known, time-indexed ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


SYNTHETIC_PROTOCOL_VERSION = "v2_recurring_graphs_measurement_snr"


@dataclass(frozen=True)
class SyntheticRegimeData:
    X: np.ndarray
    A_true: np.ndarray
    change_points: tuple[int, ...]
    clean_X: Optional[np.ndarray] = None


def _innovation(rng, size, scale, distribution):
    if distribution == "gaussian":
        return rng.normal(0, scale, size)
    if distribution == "student_t":
        # Unit-variance t innovations (df=5).
        return rng.standard_t(5, size) * scale / np.sqrt(5 / 3)
    if distribution == "uniform":
        return rng.uniform(-np.sqrt(3) * scale, np.sqrt(3) * scale, size)
    raise ValueError("distribution must be gaussian, student_t, or uniform")


def _graph_for_regime(n_variables, max_lag, regime, base):
    """Alternate two explicit graphs: A -> B -> A -> B.

    Returning to graph A at the second change point is the recurrence test. It
    makes memory retention measurable rather than merely adding unrelated
    perturbations at each boundary.
    """
    if regime % 2 == 0:
        return base.copy()
    graph = base.copy()
    # Replace one true edge with a previously absent, non-self lagged edge.
    removed_source, removed_target = 0, 1 % n_variables
    removed_lag = 1 + removed_target % max_lag
    graph[removed_source, removed_target, removed_lag] = 0.0
    added_source, added_target = n_variables - 1, 1 % n_variables
    if added_source == added_target:
        added_target = (added_target + 1) % n_variables
    graph[added_source, added_target, 1] = -0.35
    return graph


def generate_synthetic_regimes(
    *, n_points=5000, n_variables=4, max_lag=2, change_points=(1000, 2500, 4000),
    noise_scale=0.1, distribution="gaussian", seed=0, process_noise=0.1,
) -> SyntheticRegimeData:
    """Generate stable VAR processes with recurring changes and controlled SNR.

    Graph entries use ``[source, target, lag]`` and only positive lags, allowing
    exact comparison with a past-only discovery method.
    """
    if n_variables < 2 or max_lag < 1 or n_points <= max_lag + 10:
        raise ValueError("need >=2 variables, positive lag, and sufficient observations")
    if noise_scale < 0 or process_noise <= 0:
        raise ValueError("noise_scale must be non-negative and process_noise positive")
    points = tuple(sorted(set(int(point) for point in change_points if 0 < point < n_points)))
    rng = np.random.default_rng(seed)
    base = np.zeros((n_variables, n_variables, max_lag + 1), dtype=np.float32)
    # Sparse, stable, low-magnitude VAR coefficients.
    for target in range(n_variables):
        source = (target - 1) % n_variables
        base[source, target, 1 + target % max_lag] = 0.30
    graphs = [_graph_for_regime(n_variables, max_lag, index, base) for index in range(len(points) + 1)]
    truth = np.zeros((n_points, n_variables, n_variables, max_lag + 1), dtype=np.float32)
    clean = np.zeros((n_points, n_variables), dtype=np.float32)
    clean[:max_lag] = rng.normal(0, process_noise, size=(max_lag, n_variables))
    regime = 0
    next_change = iter(points)
    boundary = next(next_change, None)
    for t in range(max_lag, n_points):
        if boundary is not None and t >= boundary:
            regime += 1
            boundary = next(next_change, None)
        graph = graphs[regime]
        # Fixed process innovation sets the common signal scale. The requested
        # sigma is applied only afterwards as measurement innovation, so sigma
        # genuinely changes SNR instead of simply rescaling the entire VAR.
        value = rng.normal(0, process_noise, size=n_variables)
        for lag in range(1, max_lag + 1):
            value += clean[t - lag] @ graph[:, :, lag]
        clean[t] = value
        truth[t] = graph
    truth[:max_lag] = graphs[0]
    observed = clean + _innovation(rng, clean.shape, noise_scale, distribution)
    return SyntheticRegimeData(X=observed.astype(np.float32), A_true=truth, change_points=points, clean_X=clean)
