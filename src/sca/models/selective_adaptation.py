"""Dimension-agnostic, strictly past-only Selective Causal Adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class SCAConfig:
    """Tunable parameters; all windows are measured in observations."""

    regression_window: int = 100
    detection_window: int = 100
    threshold: float = 0.08
    persistence_decay: float = 0.90
    deviation_scale: float = 0.05
    min_plasticity: float = 0.02
    max_plasticity: float = 0.50
    update_interval: int = 10
    ridge_alpha: float = 0.0

    def __post_init__(self):
        if self.regression_window < 3 or self.detection_window < 3:
            raise ValueError("regression_window and detection_window must be at least 3")
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive")
        if not 0 <= self.persistence_decay <= 1:
            raise ValueError("persistence_decay must be in [0, 1]")
        if self.min_plasticity < 0 or self.max_plasticity > 1 or self.min_plasticity > self.max_plasticity:
            raise ValueError("plasticity bounds must satisfy 0 <= min <= max <= 1")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - values.mean(axis=0)) / scale


def estimate_lagged_graph(X: np.ndarray, max_lag: int, *, ridge_alpha: float = 0.0) -> np.ndarray:
    """Estimate conditional lagged effects with no contemporaneous predictors.

    The output's lag-zero plane is always zero.  This is intentional: a
    past-only online estimator cannot identify same-time directions without an
    additional structural-identification assumption.  It also prevents the
    target-leakage present in the legacy estimator.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or max_lag < 1:
        raise ValueError("X must be two-dimensional and max_lag must be positive")
    n, d = X.shape
    graph = np.zeros((d, d, max_lag + 1), dtype=np.float32)
    if n <= max_lag + 5:
        return graph
    Z = _standardize(X)
    target = Z[max_lag:]
    features = np.concatenate([Z[max_lag - lag:n - lag] for lag in range(1, max_lag + 1)], axis=1)
    design = np.column_stack([np.ones(len(features)), features])
    if ridge_alpha:
        # Standardisation above puts every lagged predictor on the same scale.
        # Do not penalise the intercept; this keeps the ridge estimate local and
        # comparable with OLS while making p ~= n windows well posed.
        penalty = np.eye(design.shape[1]) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        try:
            coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)[1:]
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(design.T @ design + penalty, design.T @ target, rcond=None)[0][1:]
    else:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0][1:]
    for lag in range(1, max_lag + 1):
        block = coefficients[(lag - 1) * d:lag * d]
        graph[:, :, lag] = block
    for variable in range(d):
        graph[variable, variable, :] = 0.0
    return graph


def _change_score(previous: np.ndarray, current: np.ndarray) -> float:
    difference = np.abs(current - previous).copy()
    diagonal = np.arange(difference.shape[0])
    difference[diagonal, diagonal, :] = 0.0
    k = min(4, difference.size)
    return float(np.partition(difference.ravel(), -k)[-k:].mean())


class SelectiveCausalAdapter:
    """Online causal-memory estimator parameterized by graph dimension and lag."""

    def __init__(self, max_lag: int, config: Optional[SCAConfig] = None):
        if max_lag < 1:
            raise ValueError("max_lag must be positive")
        self.max_lag = max_lag
        self.config = config or SCAConfig()

    def predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must have shape (time, variables)")
        n, d = X.shape
        shape = (d, d, self.max_lag + 1)
        predictions = np.zeros((n, *shape), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        detected = np.zeros(n, dtype=bool)
        plasticity = np.zeros((n, *shape), dtype=np.float32)
        memory = np.zeros(shape, dtype=np.float32)
        evidence = np.zeros(shape, dtype=np.float32)
        initialized = False

        for t in range(self.max_lag, n):
            if t % self.config.update_interval:
                predictions[t] = memory
                continue
            local_start = max(0, t - self.config.regression_window + 1)
            local = estimate_lagged_graph(X[local_start:t + 1], self.max_lag, ridge_alpha=self.config.ridge_alpha)
            if not initialized:
                memory = local
                initialized = True
                predictions[t] = memory
                continue
            if t + 1 >= 2 * self.config.detection_window:
                split = t - self.config.detection_window + 1
                previous = estimate_lagged_graph(X[t - 2 * self.config.detection_window + 1:split], self.max_lag, ridge_alpha=self.config.ridge_alpha)
                recent = estimate_lagged_graph(X[split:t + 1], self.max_lag, ridge_alpha=self.config.ridge_alpha)
                scores[t] = _change_score(previous, recent)
                detected[t] = scores[t] >= self.config.threshold
            intensity = min(1.0, scores[t] / max(self.config.threshold, 1e-12)) if detected[t] else 0.0
            difference = np.abs(local - memory)
            evidence = self.config.persistence_decay * evidence + (1 - self.config.persistence_decay) * (difference >= self.config.deviation_scale)
            edge_signal = difference / (difference + self.config.deviation_scale)
            rates = self.config.min_plasticity + (self.config.max_plasticity - self.config.min_plasticity) * intensity * edge_signal * evidence
            diagonal = np.arange(d)
            rates[diagonal, diagonal, :] = 0.0
            memory = (1 - rates) * memory + rates * local
            plasticity[t] = rates
            predictions[t] = memory
        # Carry the most recently available state across skipped timestamps.
        for t in range(1, n):
            if not np.any(predictions[t]):
                predictions[t] = predictions[t - 1]
        return predictions, scores, detected, plasticity

    def final_graph(self, X: np.ndarray) -> np.ndarray:
        """Return only the final online memory state, avoiding large histories.

        This is mathematically identical to :meth:`predict` at the last update
        and is used for validation searches on large real-world series.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must have shape (time, variables)")
        n, d = X.shape
        shape = (d, d, self.max_lag + 1)
        memory = np.zeros(shape, dtype=np.float32)
        evidence = np.zeros(shape, dtype=np.float32)
        initialized = False
        diagonal = np.arange(d)
        for t in range(self.max_lag, n):
            if t % self.config.update_interval:
                continue
            local = estimate_lagged_graph(X[max(0, t - self.config.regression_window + 1):t + 1], self.max_lag, ridge_alpha=self.config.ridge_alpha)
            if not initialized:
                memory = local
                initialized = True
                continue
            score = 0.0
            if t + 1 >= 2 * self.config.detection_window:
                split = t - self.config.detection_window + 1
                previous = estimate_lagged_graph(X[t - 2 * self.config.detection_window + 1:split], self.max_lag, ridge_alpha=self.config.ridge_alpha)
                recent = estimate_lagged_graph(X[split:t + 1], self.max_lag, ridge_alpha=self.config.ridge_alpha)
                score = _change_score(previous, recent)
            intensity = min(1.0, score / max(self.config.threshold, 1e-12)) if score >= self.config.threshold else 0.0
            difference = np.abs(local - memory)
            evidence = self.config.persistence_decay * evidence + (1 - self.config.persistence_decay) * (difference >= self.config.deviation_scale)
            signal = difference / (difference + self.config.deviation_scale)
            rates = self.config.min_plasticity + (self.config.max_plasticity - self.config.min_plasticity) * intensity * signal * evidence
            rates[diagonal, diagonal, :] = 0.0
            memory = (1 - rates) * memory + rates * local
        return memory
