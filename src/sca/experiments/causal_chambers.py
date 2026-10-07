"""Chronology-safe Causal Chambers wind-tunnel benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..evaluation.evaluation import precision_recall_f1, structural_hamming_distance
from ..models.selective_adaptation import SCAConfig, SelectiveCausalAdapter


ADMIN_COLUMNS = {"timestamp", "config", "counter", "flag", "intervention"}


def load_wt_walks(csv_path, *, reference_graph=None):
    """Load a Causal Chambers experiment without shuffle or outcome leakage."""
    frame = pd.read_csv(csv_path).sort_values("timestamp", kind="stable")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be unique and strictly chronological")
    if reference_graph is None:
        try:
            from causalchamber.ground_truth import graph
            reference_graph = graph("wt", "standard")
        except ImportError as error:
            raise RuntimeError("Install causalchamber==0.2.8 to load the official reference graph.") from error
    variables = [name for name in reference_graph.index if name in frame.columns and name not in ADMIN_COLUMNS]
    if not variables:
        raise ValueError("No official graph variables were found in the CSV")
    values = frame[variables].apply(pd.to_numeric, errors="raise")
    if values.isna().any().any():
        raise ValueError("Missing signal values require an explicit, forward-only imputation protocol")
    truth = reference_graph.loc[variables, variables].to_numpy(dtype=np.float32)
    return values.to_numpy(dtype=np.float32), truth, variables, frame["timestamp"].to_numpy()


def _collapsed_tensor(graph):
    """Map weighted time-lag estimates to a directed adjacency, preserving sign magnitude."""
    return np.max(np.abs(graph[:, :, 1:]), axis=2) if graph.shape[-1] > 1 else np.abs(graph[:, :, 0])


def _score_collapsed(graph, truth, threshold):
    prediction = _collapsed_tensor(graph)
    return {
        **precision_recall_f1(prediction[:, :, None], truth[:, :, None], threshold),
        "shd": structural_hamming_distance(prediction[:, :, None], truth[:, :, None], threshold),
    }


def run_sca_wt_walks(X, truth, *, max_lag=10, threshold=0.05):
    """Fit SCA on the full chronology and score final lag-collapsed recovery."""
    # 500 samples is required for a stable 32-variable, 10-lag regression.
    config = SCAConfig(regression_window=500, detection_window=500, threshold=threshold, update_interval=50)
    start = perf_counter()
    prediction, scores, flags, plasticity = SelectiveCausalAdapter(max_lag, config).predict(X)
    elapsed = perf_counter() - start
    return prediction, {**_score_collapsed(prediction[-1], truth, threshold), "runtime_seconds": elapsed, "updates": int(np.count_nonzero(np.any(plasticity, axis=(1, 2, 3))) )}


def run_pcmci_wt_walks(X, variables, truth, *, max_lag=10, alpha_level=0.01):
    """Run official-style PCMCI+ and score lag-collapsed directed recovery."""
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as error:
        return None, {"status": "unavailable", "reason": str(error)}
    start = perf_counter()
    result = PCMCI(dataframe=pp.DataFrame(X, var_names=variables), cond_ind_test=ParCorr(significance="analytic"), verbosity=0).run_pcmciplus(tau_min=1, tau_max=max_lag, pc_alpha=alpha_level)
    elapsed = perf_counter() - start
    graph = result["graph"]
    d = len(variables)
    tensor = np.zeros((d, d, max_lag + 1), dtype=np.float32)
    for target in range(d):
        for source in range(d):
            for lag in range(1, max_lag + 1):
                # Tigramite stores graph[target, source, lag].  For a lagged
                # entry ``-->`` means source(t-lag) -> target(t).  Earlier
                # code inverted this marker and could report an empty graph.
                if graph[target, source, lag] == "-->":
                    tensor[source, target, lag] = 1.0
    return tensor, {**_score_collapsed(tensor, truth, 0.0), "runtime_seconds": elapsed, "status": "completed"}


def write_report(output_dir, *, variables, timestamps, metrics, protocol):
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
    report = {"variables": variables, "n_variables": len(variables), "n_observations": len(timestamps), "median_sampling_interval": float(np.median(np.diff(timestamps))), "protocol": protocol}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
