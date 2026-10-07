"""Benchmark configuration system for systematic change-type evaluation.

This module defines the eight change-type benchmarks required for
IEEE Access evaluation:

  1. NO_CHANGE       -- constant graph; tests false-alarm rate
  2. EDGE_ADDITION   -- 0 -> w (adds a previously absent edge)
  3. EDGE_DELETION   -- w -> 0 (removes a present edge)
  4. WEIGHT_CHANGE   -- w1 -> w2 (changes edge strength)
  5. MULTI_CHANGE    -- several edges change simultaneously
  6. GRADUAL_CHANGE  -- edge strength changes linearly over a ramp period
  7. LOCATION_SWEEP  -- change point at different positions
  8. SNR_SWEEP       -- different measurement noise levels

Each configuration is a frozen dataclass.  Call ``generate_benchmark``
to obtain a ``SyntheticRegimeData`` object suitable for evaluation.

Evaluation convention
---------------------
All metrics use the paper-facing preregistered threshold:
    edge_threshold = PAPER_EDGE_THRESHOLD = 0.15

Reproducibility
---------------
Every benchmark is seeded.  Use a different seed value for each
experimental condition to ensure independence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import numpy as np

from .synthetic import SyntheticRegimeData


# ===========================================================
# Benchmark type identifiers
# ===========================================================

BENCHMARK_NO_CHANGE = "no_change"
BENCHMARK_EDGE_ADDITION = "edge_addition"
BENCHMARK_EDGE_DELETION = "edge_deletion"
BENCHMARK_WEIGHT_CHANGE = "weight_change"
BENCHMARK_MULTI_CHANGE = "multi_change"
BENCHMARK_GRADUAL_CHANGE = "gradual_change"
BENCHMARK_LOCATION_SWEEP = "location_sweep"
BENCHMARK_SNR_SWEEP = "snr_sweep"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark condition.

    Parameters
    ----------
    benchmark_type : str
        One of the BENCHMARK_* identifiers above.
    n_points : int
        Length of the time series.
    n_variables : int
        Number of observed variables.
    max_lag : int
        Maximum lag considered by the estimator.
    noise_scale : float
        Standard deviation of measurement noise.
    seed : int
        Random seed for reproducibility.
    change_point : int
        Time index at which the first regime change occurs.
    ramp_length : int
        For GRADUAL_CHANGE only: number of steps over which
        the edge weight is linearly interpolated.
    weight_before : float
        Edge weight before the change (for deletion, weight, gradual).
    weight_after : float
        Edge weight after the change (for addition, weight, gradual).
    """
    benchmark_type: str = BENCHMARK_NO_CHANGE
    n_points: int = 2000
    n_variables: int = 4
    max_lag: int = 2
    noise_scale: float = 0.10
    seed: int = 0
    change_point: int = 1000
    ramp_length: int = 0        # 0 = instantaneous change
    weight_before: float = 0.30
    weight_after: float = 0.00


def _sparse_base_graph(n_variables: int, max_lag: int) -> np.ndarray:
    """Create a sparse, stable VAR coefficient matrix (lagged edges only)."""
    graph = np.zeros((n_variables, n_variables, max_lag + 1), dtype=np.float32)
    for target in range(n_variables):
        source = (target - 1) % n_variables
        lag = 1 + target % max_lag
        graph[source, target, lag] = 0.30
    return graph


def _simulate_var(
    graphs: list,
    change_points: tuple,
    n_points: int,
    max_lag: int,
    n_variables: int,
    noise_scale: float,
    rng: np.random.Generator,
    ramp_length: int = 0,
    weight_before: float = 0.30,
    weight_after: float = 0.00,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a VAR process from a sequence of graphs.

    For ramp_length > 0, the transition between the first two graphs
    is linear over ``ramp_length`` steps.
    """
    n_vars = n_variables
    X = np.zeros((n_points, n_vars), dtype=np.float32)
    X[:max_lag] = rng.normal(0, noise_scale, (max_lag, n_vars))
    truth = np.zeros((n_points, n_vars, n_vars, max_lag + 1), dtype=np.float32)

    # Build regime schedule
    cp_iter = iter(change_points)
    boundary = next(cp_iter, None)
    regime = 0

    for t in range(max_lag, n_points):
        if boundary is not None and t >= boundary:
            regime = min(regime + 1, len(graphs) - 1)
            boundary = next(cp_iter, None)

        graph = graphs[regime].copy()

        # Apply linear ramp for gradual change
        if ramp_length > 0 and len(change_points) >= 1:
            cp0 = change_points[0]
            if cp0 <= t < cp0 + ramp_length:
                alpha = (t - cp0) / ramp_length
                # Linearly interpolate between graphs[0] and graphs[1]
                graph = (1.0 - alpha) * graphs[0] + alpha * graphs[1]

        truth[t] = graph
        value = rng.normal(0, noise_scale, n_vars)
        for lag in range(1, max_lag + 1):
            if t - lag >= 0:
                value = value + X[t - lag] @ graph[:, :, lag]
        X[t] = value.astype(np.float32)

    truth[:max_lag] = graphs[0]
    return X, truth


def generate_benchmark(config: BenchmarkConfig) -> SyntheticRegimeData:
    """Generate a benchmark dataset from a configuration.

    Returns a SyntheticRegimeData with X, A_true, and change_points
    suitable for evaluation.
    """
    rng = np.random.default_rng(config.seed)
    base = _sparse_base_graph(config.n_variables, config.max_lag)

    bt = config.benchmark_type

    # -------------------------------------------------------
    # 1. NO_CHANGE -- stationary; no regime shift
    # -------------------------------------------------------
    if bt == BENCHMARK_NO_CHANGE:
        graphs = [base]
        change_points = ()

    # -------------------------------------------------------
    # 2. EDGE_ADDITION -- absent edge gains weight
    # -------------------------------------------------------
    elif bt == BENCHMARK_EDGE_ADDITION:
        after = base.copy()
        # Add edge from last to first variable at lag 1
        s, tgt, lag = config.n_variables - 1, 0, 1
        after[s, tgt, lag] = config.weight_after if config.weight_after != 0 else 0.30
        graphs = [base, after]
        change_points = (config.change_point,)

    # -------------------------------------------------------
    # 3. EDGE_DELETION -- present edge drops to zero
    # -------------------------------------------------------
    elif bt == BENCHMARK_EDGE_DELETION:
        # Remove the first base edge
        after = base.copy()
        tgt = 0
        s = (tgt - 1) % config.n_variables
        lag = 1 + tgt % config.max_lag
        after[s, tgt, lag] = 0.0
        graphs = [base, after]
        change_points = (config.change_point,)

    # -------------------------------------------------------
    # 4. WEIGHT_CHANGE -- edge changes strength
    # -------------------------------------------------------
    elif bt == BENCHMARK_WEIGHT_CHANGE:
        after = base.copy()
        tgt = 0
        s = (tgt - 1) % config.n_variables
        lag = 1 + tgt % config.max_lag
        after[s, tgt, lag] = config.weight_after
        graphs = [base, after]
        change_points = (config.change_point,)

    # -------------------------------------------------------
    # 5. MULTI_CHANGE -- multiple edges change at once
    # -------------------------------------------------------
    elif bt == BENCHMARK_MULTI_CHANGE:
        after = base.copy()
        n = config.n_variables
        # Shift the base pattern: each edge weight changes
        for target in range(n):
            source = (target - 1) % n
            lag = 1 + target % config.max_lag
            after[source, target, lag] = -0.30  # flip sign
        # Also add one new edge
        new_s, new_tgt, new_lag = n - 1, 0, 1
        if base[new_s, new_tgt, new_lag] == 0:
            after[new_s, new_tgt, new_lag] = 0.25
        graphs = [base, after]
        change_points = (config.change_point,)

    # -------------------------------------------------------
    # 6. GRADUAL_CHANGE -- linear ramp over ramp_length steps
    # -------------------------------------------------------
    elif bt == BENCHMARK_GRADUAL_CHANGE:
        after = base.copy()
        tgt = 0
        s = (tgt - 1) % config.n_variables
        lag = 1 + tgt % config.max_lag
        after[s, tgt, lag] = config.weight_after
        graphs = [base, after]
        change_points = (config.change_point,)

    # -------------------------------------------------------
    # 7. LOCATION_SWEEP -- handled by varying change_point in config
    # -------------------------------------------------------
    elif bt == BENCHMARK_LOCATION_SWEEP:
        after = base.copy()
        tgt = 0
        s = (tgt - 1) % config.n_variables
        lag = 1 + tgt % config.max_lag
        after[s, tgt, lag] = 0.0
        graphs = [base, after]
        change_points = (config.change_point,)

    # -------------------------------------------------------
    # 8. SNR_SWEEP -- handled by varying noise_scale in config
    # -------------------------------------------------------
    elif bt == BENCHMARK_SNR_SWEEP:
        after = base.copy()
        tgt = 0
        s = (tgt - 1) % config.n_variables
        lag = 1 + tgt % config.max_lag
        after[s, tgt, lag] = 0.0
        graphs = [base, after]
        change_points = (config.change_point,)

    else:
        raise ValueError(f"Unknown benchmark_type: {bt!r}")

    ramp_length = config.ramp_length if bt == BENCHMARK_GRADUAL_CHANGE else 0
    X, truth = _simulate_var(
        graphs=graphs,
        change_points=change_points,
        n_points=config.n_points,
        max_lag=config.max_lag,
        n_variables=config.n_variables,
        noise_scale=config.noise_scale,
        rng=rng,
        ramp_length=ramp_length,
    )

    return SyntheticRegimeData(
        X=X,
        A_true=truth,
        change_points=change_points,
    )


# ===========================================================
# Pre-defined benchmark suites
# ===========================================================

DEFAULT_BENCHMARK_SUITE = [
    # 1. No change baseline
    BenchmarkConfig(benchmark_type=BENCHMARK_NO_CHANGE, seed=100),

    # 2. Edge addition
    BenchmarkConfig(benchmark_type=BENCHMARK_EDGE_ADDITION, seed=101),

    # 3. Edge deletion
    BenchmarkConfig(benchmark_type=BENCHMARK_EDGE_DELETION, seed=102),

    # 4. Weight change (strong)
    BenchmarkConfig(
        benchmark_type=BENCHMARK_WEIGHT_CHANGE,
        weight_after=0.10,  # 0.30 -> 0.10
        seed=103,
    ),

    # 4b. Weight change (weak)
    BenchmarkConfig(
        benchmark_type=BENCHMARK_WEIGHT_CHANGE,
        weight_after=0.25,  # 0.30 -> 0.25 (small change)
        seed=104,
    ),

    # 5. Multiple simultaneous changes
    BenchmarkConfig(benchmark_type=BENCHMARK_MULTI_CHANGE, seed=105),

    # 6. Gradual change (200-step ramp)
    BenchmarkConfig(
        benchmark_type=BENCHMARK_GRADUAL_CHANGE,
        ramp_length=200,
        weight_after=0.0,
        seed=106,
    ),

    # 7. Change-point location sweep
    BenchmarkConfig(
        benchmark_type=BENCHMARK_LOCATION_SWEEP,
        change_point=250,
        seed=107,
    ),
    BenchmarkConfig(
        benchmark_type=BENCHMARK_LOCATION_SWEEP,
        change_point=1000,
        seed=108,
    ),
    BenchmarkConfig(
        benchmark_type=BENCHMARK_LOCATION_SWEEP,
        change_point=1750,
        seed=109,
    ),

    # 8. SNR sweep
    BenchmarkConfig(
        benchmark_type=BENCHMARK_SNR_SWEEP,
        noise_scale=0.05,
        seed=110,
    ),
    BenchmarkConfig(
        benchmark_type=BENCHMARK_SNR_SWEEP,
        noise_scale=0.25,
        seed=111,
    ),
    BenchmarkConfig(
        benchmark_type=BENCHMARK_SNR_SWEEP,
        noise_scale=0.50,
        seed=112,
    ),
]
