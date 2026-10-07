"""Baseline models for causal discovery (Rolling Regression, Static Regression, PCMCI+)."""

from .rolling_regression_baseline import estimate_graph as estimate_rolling_graph
from .static_regression_baseline import estimate_static_graph
from .regime_aware import RegimeAwareBaselineUnavailable, run_rpcmci
from .streaming_linear import (
    GlobalEMAConfig,
    GlobalRateEMA,
    KGLSConfig,
    KGLSPreviousGraph,
    RLSConfig,
    RecursiveLeastSquares,
)

__all__ = [
    "estimate_rolling_graph",
    "estimate_static_graph",
    "RegimeAwareBaselineUnavailable",
    "run_rpcmci",
    "GlobalEMAConfig",
    "GlobalRateEMA",
    "KGLSConfig",
    "KGLSPreviousGraph",
    "RLSConfig",
    "RecursiveLeastSquares",
]
