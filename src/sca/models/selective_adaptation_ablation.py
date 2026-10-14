"""Ablation variants of the *final* (lag-zero-free) Selective Causal Adaptation.

The three variants share every line of the update loop with
``SelectiveCausalAdapter.predict`` in ``selective_adaptation.py`` (same local
ridge/OLS estimator, same two-window detector, same persistence update, same EMA
memory update).  They differ only in how the edgewise rate is formed:

``full``            alpha = a_min + (a_max - a_min) * I_t * s_ijl * P_ijl   (Eq. 11)
``no_persistence``  alpha = a_min + (a_max - a_min) * I_t * s_ijl           (P factor removed)
``global_only``     alpha = a_min + (a_max - a_min) * I_t                   (one rate for all edges)

These definitions mirror the legacy scripts ``sca_model_no_persistence.py`` and
``sca_model_global_only.py``, but are built on ``estimate_lagged_graph``, so the
lag-zero plane of every emitted graph is identically zero.  The ``full`` variant
is verified to reproduce ``SelectiveCausalAdapter.predict`` exactly
(``tests/test_dev_ablation_consistency.py``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .selective_adaptation import SCAConfig, _change_score, estimate_lagged_graph

GATINGS = ("full", "no_persistence", "global_only")


def predict_gated(X: np.ndarray, max_lag: int, config: Optional[SCAConfig] = None, gating: str = "full"):
    """Return ``(predictions, scores, detected, plasticity)`` for one gating variant."""
    if gating not in GATINGS:
        raise ValueError(f"gating must be one of {GATINGS}")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    config = config or SCAConfig()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (time, variables)")
    n, d = X.shape
    shape = (d, d, max_lag + 1)
    predictions = np.zeros((n, *shape), dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    plasticity = np.zeros((n, *shape), dtype=np.float32)
    memory = np.zeros(shape, dtype=np.float32)
    evidence = np.zeros(shape, dtype=np.float32)
    initialized = False

    for t in range(max_lag, n):
        if t % config.update_interval:
            predictions[t] = memory
            continue
        local_start = max(0, t - config.regression_window + 1)
        local = estimate_lagged_graph(X[local_start:t + 1], max_lag, ridge_alpha=config.ridge_alpha)
        if not initialized:
            memory = local
            initialized = True
            predictions[t] = memory
            continue
        if t + 1 >= 2 * config.detection_window:
            split = t - config.detection_window + 1
            previous = estimate_lagged_graph(X[t - 2 * config.detection_window + 1:split], max_lag, ridge_alpha=config.ridge_alpha)
            recent = estimate_lagged_graph(X[split:t + 1], max_lag, ridge_alpha=config.ridge_alpha)
            scores[t] = _change_score(previous, recent)
            detected[t] = scores[t] >= config.threshold
        intensity = min(1.0, scores[t] / max(config.threshold, 1e-12)) if detected[t] else 0.0
        difference = np.abs(local - memory)
        evidence = config.persistence_decay * evidence + (1 - config.persistence_decay) * (difference >= config.deviation_scale)
        edge_signal = difference / (difference + config.deviation_scale)
        span = config.max_plasticity - config.min_plasticity
        if gating == "full":
            rates = config.min_plasticity + span * intensity * edge_signal * evidence
        elif gating == "no_persistence":
            rates = config.min_plasticity + span * intensity * edge_signal
        else:  # global_only
            rates = np.full(shape, config.min_plasticity + span * intensity, dtype=np.float64)
        diagonal = np.arange(d)
        rates[diagonal, diagonal, :] = 0.0
        memory = (1 - rates) * memory + rates * local
        plasticity[t] = rates
        predictions[t] = memory
    # Carry the most recently available state across skipped timestamps
    # (identical to SelectiveCausalAdapter.predict).
    for t in range(1, n):
        if not np.any(predictions[t]):
            predictions[t] = predictions[t - 1]
    return predictions, scores, detected, plasticity
