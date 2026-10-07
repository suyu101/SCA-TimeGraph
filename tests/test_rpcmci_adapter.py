"""Unit tests for native RPCMCI output conversion without optional dependencies."""

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rpcmci.py"
    spec = importlib.util.spec_from_file_location("evaluate_rpcmci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rpcmci_graph_conversion_preserves_native_source_target_orientation():
    module = _module()
    graph = np.empty((2, 2, 2), dtype="<U3")
    graph.fill("")
    graph[0, 1, 1] = "-->"  # variable 0 at t-1 causes variable 1 at t
    result = {
        "regimes": np.array([[1, 1, 0], [0, 0, 1]], dtype=float),
        "causal_results": {0: {"graph": graph}, 1: {"graph": graph}},
    }
    prediction, flags = module._regime_predictions(result, n_variables=2, max_lag=1)
    assert np.all(prediction[:, 0, 1, 1] == 1)
    assert np.all(prediction[:, 1, 0, 1] == 0)
    assert flags.tolist() == [False, False, True]
