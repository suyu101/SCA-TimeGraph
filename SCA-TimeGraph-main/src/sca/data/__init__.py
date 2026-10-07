"""Safe public data-loading and chronological-preprocessing API."""

from .data_pipeline import (
    apply_standardizer,
    chronological_split,
    create_temporal_windows,
    fit_standardizer,
    load_timegraph,
    prepare_timegraph,
)
from .normalize_data import create_raw_splits, standardize_using_train
from .regime_dataset import load_regime_change_dataset, load_regime_dataset
from .window_builder import create_temporal_windows as build_windows

# Compatibility aliases for the package API documented in early drafts.
temporal_train_test_split = chronological_split
fit_training_normalizer = fit_standardizer
apply_normalizer = apply_standardizer
build_lagged_windows = create_temporal_windows
normalize_data = standardize_using_train
load_dataset = load_timegraph

__all__ = [
    "load_timegraph", "load_dataset", "prepare_timegraph", "chronological_split",
    "temporal_train_test_split", "fit_standardizer", "fit_training_normalizer",
    "apply_standardizer", "apply_normalizer", "create_temporal_windows",
    "build_lagged_windows", "build_windows", "standardize_using_train",
    "normalize_data", "create_raw_splits", "load_regime_change_dataset",
    "load_regime_dataset",
]
