"""Checkpointed CD-NOD benchmark on the canonical dynamic synthetic setting.

CD-NOD is a nonstationary causal-discovery baseline.  To retain the project
time-series target, each observation is lag-expanded and the chronological
sample index is supplied as CD-NOD's documented domain-index input.  Any
adjacency from a past node to a present node is directed forward in time when
scored; contemporaneous and backward-in-time edges are excluded.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.experiments.synthetic import generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence


def _atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _lag_expand(X, max_lag):
    """Return [present, lag 1, ..., lag L] rows and their time indices."""
    n, _ = X.shape
    rows = [X[max_lag:]]
    rows.extend(X[max_lag - lag:n - lag] for lag in range(1, max_lag + 1))
    return np.concatenate(rows, axis=1), np.arange(max_lag, n, dtype=float)[:, None]


def _cdnod_prediction(X, max_lag, alpha):
    try:
        from causallearn.search.ConstraintBased.CDNOD import cdnod
    except ImportError as error:
        raise RuntimeError(
            "CD-NOD requires causal-learn; install the dynamic-baselines extra."
        ) from error
    expanded, time_index = _lag_expand(np.asarray(X, dtype=float), max_lag)
    graph = np.asarray(
        cdnod(
            expanded, time_index, alpha=float(alpha), indep_test="fisherz",
            stable=True, verbose=False, show_progress=False,
        ).G.graph
    )
    d = X.shape[1]
    tensor = np.zeros((d, d, max_lag + 1), dtype=np.float32)
    # CD-NOD learns adjacencies in the augmented data.  Temporal precedence is
    # known, so an adjacency between a lagged source and current target is
    # scored as source(t-lag) -> target(t), regardless of CPDAG ambiguity.
    for lag in range(1, max_lag + 1):
        for source in range(d):
            for target in range(d):
                if source == target:
                    continue
                source_index, target_index = lag * d + source, target
                if graph[source_index, target_index] != 0 or graph[target_index, source_index] != 0:
                    tensor[source, target, lag] = 1.0
    return np.broadcast_to(tensor, (len(X), *tensor.shape)).copy()


def main():
    parser = argparse.ArgumentParser(description="Run the native CD-NOD dynamic-baseline benchmark")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="results/validation/cdnod_raw.csv")
    args = parser.parse_args()
    if args.seeds < 20:
        parser.error("At least 20 seeds are required for the manuscript baseline.")
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "status", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points", "alpha", "precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate", "runtime_seconds", "error"]
    existing = pd.read_csv(target) if target.exists() else pd.DataFrame(columns=columns)
    rows = existing.to_dict("records")
    completed = set(existing.loc[existing.status == "completed", "seed"])
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if seed in completed:
            continue
        row = {"method": "CD-NOD", "seed": seed, "n_variables": 4, "max_lag": 2, "noise_scale": 0.1, "distribution": "gaussian", "n_points": args.n_points, "alpha": args.alpha}
        print(f"Starting native CD-NOD seed {seed} (alpha={args.alpha}).", flush=True)
        try:
            data = generate_synthetic_regimes(n_points=args.n_points, n_variables=4, max_lag=2, noise_scale=0.1, distribution="gaussian", seed=seed)
            started = time.perf_counter()
            prediction = _cdnod_prediction(data.X, max_lag=2, alpha=args.alpha)
            metrics = _evaluate_sequence(prediction, data.A_true, data.change_points)
            rows.append({**row, "status": "completed", **metrics, "runtime_seconds": time.perf_counter() - started, "error": ""})
        except Exception as error:
            rows.append({**row, "status": "failed", **{key: np.nan for key in columns[9:-2]}, "runtime_seconds": np.nan, "error": repr(error)})
        _atomic_csv(pd.DataFrame(rows, columns=columns), target)
        print(f"Checkpoint saved: CD-NOD seed {seed}.", flush=True)


if __name__ == "__main__":
    main()
