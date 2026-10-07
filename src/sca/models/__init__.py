"""Selective Causal Adaptation (SCA) models and detectors."""

from .window_change_detector import detect_changes
from .selective_adaptation import SCAConfig, SelectiveCausalAdapter, estimate_lagged_graph

detect_window_changes = detect_changes

__all__ = [
    "detect_changes",
    "detect_window_changes",
    "SCAConfig",
    "SelectiveCausalAdapter",
    "estimate_lagged_graph",
]
