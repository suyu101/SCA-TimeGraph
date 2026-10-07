"""Matched past-only streaming linear baselines.

These estimators deliberately use the same lagged graph representation as SCA.
They are comparison methods, not variants of SCA: none uses its detector,
persistence signal, or edge-specific plasticity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.selective_adaptation import estimate_lagged_graph


def _zero_diagonal(graph: np.ndarray) -> np.ndarray:
    graph = np.asarray(graph, dtype=np.float32).copy()
    diagonal = np.arange(graph.shape[0])
    graph[diagonal, diagonal, :] = 0.0
    return graph


def _graph_from_coefficients(coefficients: np.ndarray, n_variables: int, max_lag: int) -> np.ndarray:
    """Map feature-by-target coefficients to [source, target, lag]."""
    graph = np.zeros((n_variables, n_variables, max_lag + 1), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        graph[:, :, lag] = coefficients[(lag - 1) * n_variables:lag * n_variables]
    return _zero_diagonal(graph)


def _lagged_design(values: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    features = np.concatenate(
        [values[max_lag - lag:n - lag] for lag in range(1, max_lag + 1)], axis=1,
    )
    return features, values[max_lag:]


@dataclass(frozen=True)
class KGLSConfig:
    """Local ridge estimate pulled toward the previous graph at each update."""

    regression_window: int = 200
    prior_strength: float = 1.0
    update_interval: int = 10

    def __post_init__(self) -> None:
        if self.regression_window < 3 or self.prior_strength < 0 or self.update_interval < 1:
            raise ValueError("invalid KGLS configuration")


class KGLSPreviousGraph:
    """Ridge-to-previous-weights baseline inspired by KGLS-style updates."""

    def __init__(self, max_lag: int, config: KGLSConfig | None = None):
        self.max_lag = max_lag
        self.config = config or KGLSConfig()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        out = np.zeros((n, d, d, self.max_lag + 1), dtype=np.float32)
        previous = None
        for t in range(self.max_lag, n):
            if t % self.config.update_interval == 0:
                window = X[max(0, t - self.config.regression_window + 1):t + 1]
                features, targets = _lagged_design(window, self.max_lag)
                if len(features) > 0:
                    # Standardising each local fit gives the same scale convention
                    # used by SCA's local graph estimator.
                    scale = window.std(axis=0)
                    scale[scale < 1e-12] = 1.0
                    standard = (window - window.mean(axis=0)) / scale
                    features, targets = _lagged_design(standard, self.max_lag)
                    gram = features.T @ features
                    cross = features.T @ targets
                    if previous is None:
                        beta = np.linalg.lstsq(features, targets, rcond=None)[0]
                    else:
                        prior = np.concatenate([previous[:, :, lag] for lag in range(1, self.max_lag + 1)], axis=0)
                        penalty = self.config.prior_strength * np.eye(gram.shape[0])
                        try:
                            beta = np.linalg.solve(gram + penalty, cross + self.config.prior_strength * prior)
                        except np.linalg.LinAlgError:
                            beta = np.linalg.lstsq(gram + penalty, cross + self.config.prior_strength * prior, rcond=None)[0]
                    previous = _graph_from_coefficients(beta, d, self.max_lag)
            if previous is not None:
                out[t] = previous
        return out


@dataclass(frozen=True)
class RLSConfig:
    """Full-matrix recursive least squares with exponential forgetting."""

    forgetting_factor: float = 0.99
    initial_covariance: float = 100.0
    update_interval: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.forgetting_factor <= 1 or self.initial_covariance <= 0 or self.update_interval < 1:
            raise ValueError("invalid RLS configuration")


class RecursiveLeastSquares:
    """Multivariate, past-only RLS; coefficients are emitted at fixed updates."""

    def __init__(self, max_lag: int, config: RLSConfig | None = None):
        self.max_lag = max_lag
        self.config = config or RLSConfig()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        p = d * self.max_lag + 1  # intercept plus all lagged predictors
        coefficients = np.zeros((p, d), dtype=float)
        covariance = np.eye(p, dtype=float) * self.config.initial_covariance
        out = np.zeros((n, d, d, self.max_lag + 1), dtype=np.float32)
        current = out[0]
        for t in range(self.max_lag, n):
            feature = np.concatenate(([1.0], *(X[t - lag] for lag in range(1, self.max_lag + 1))))
            gain_numerator = covariance @ feature
            denominator = self.config.forgetting_factor + feature @ gain_numerator
            gain = gain_numerator / denominator
            residual = X[t] - feature @ coefficients
            coefficients += np.outer(gain, residual)
            covariance = (covariance - np.outer(gain, feature) @ covariance) / self.config.forgetting_factor
            # Numerical symmetry is important for the high-dimensional cells.
            covariance = 0.5 * (covariance + covariance.T)
            if t % self.config.update_interval == 0:
                current = _graph_from_coefficients(coefficients[1:], d, self.max_lag)
            out[t] = current
        return out


@dataclass(frozen=True)
class GlobalEMAConfig:
    """A global (one-rate-for-all-edges) local-estimate EMA baseline."""

    regression_window: int = 200
    rate: float = 0.2
    ridge_alpha: float = 0.0
    update_interval: int = 10

    def __post_init__(self) -> None:
        if self.regression_window < 3 or not 0 < self.rate <= 1 or self.ridge_alpha < 0 or self.update_interval < 1:
            raise ValueError("invalid global EMA configuration")


class GlobalRateEMA:
    def __init__(self, max_lag: int, config: GlobalEMAConfig | None = None):
        self.max_lag = max_lag
        self.config = config or GlobalEMAConfig()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        out = np.zeros((n, d, d, self.max_lag + 1), dtype=np.float32)
        memory = None
        for t in range(self.max_lag, n):
            if t % self.config.update_interval == 0:
                local = estimate_lagged_graph(
                    X[max(0, t - self.config.regression_window + 1):t + 1],
                    self.max_lag,
                    ridge_alpha=self.config.ridge_alpha,
                )
                memory = local if memory is None else (1 - self.config.rate) * memory + self.config.rate * local
                memory = _zero_diagonal(memory)
            if memory is not None:
                out[t] = memory
        return out
