"""Consistency checks for the regenerated development ablation (Tables 8-9, Fig. 8)."""

import importlib.util
from pathlib import Path

import numpy as np

from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter
from sca.models.selective_adaptation_ablation import predict_gated

ROOT = Path(__file__).resolve().parents[1]


def _dev_series():
    spec = importlib.util.spec_from_file_location("gen_rc", ROOT / "scripts" / "generate_regime_change.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = module.generate_regime_change_data()
    return frame[["X1", "X2", "X3", "X4"]].to_numpy(dtype=np.float32)


def test_full_variant_reproduces_final_adapter_exactly():
    X = _dev_series()[:1500]
    config = SCAConfig()
    reference = SelectiveCausalAdapter(2, config).predict(X)
    candidate = predict_gated(X, 2, config, "full")
    for ref, cand in zip(reference, candidate):
        assert np.array_equal(ref, cand)


def test_lag_zero_plane_is_identically_zero_for_every_variant():
    X = _dev_series()[:1500]
    for gating in ("full", "no_persistence", "global_only"):
        predictions, _, _, _ = predict_gated(X, 2, SCAConfig(), gating)
        assert not np.any(predictions[..., 0]), gating


def test_global_only_assigns_one_rate_to_all_off_diagonal_edges():
    X = _dev_series()[:1500]
    _, _, _, plasticity = predict_gated(X, 2, SCAConfig(), "global_only")
    off_diagonal = ~np.eye(4, dtype=bool)
    for t in np.flatnonzero(plasticity.any(axis=(1, 2, 3))):
        values = plasticity[t][off_diagonal]
        assert np.allclose(values, values.flat[0])
