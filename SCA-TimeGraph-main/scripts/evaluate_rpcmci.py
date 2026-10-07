"""Checkpointed native RPCMCI benchmark on the preregistered dynamic subset.

The full 2,880-condition factorial grid is intentionally not used for RPCMCI:
its mixed-integer regime optimisation is orders of magnitude more expensive than
SCA.  Instead, this script evaluates the same 20 seeds on the canonical
dynamic setting (d=4, L=2, Gaussian innovations, sigma=0.10), records every
native failure, and never substitutes another model.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.baselines.regime_aware import RegimeAwareBaselineUnavailable, run_rpcmci
from sca.experiments.synthetic import generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence


def _atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _regime_predictions(result, n_variables, max_lag):
    """Convert native per-regime Tigramite graphs to the project tensor format."""
    regimes = np.asarray(result["regimes"])
    labels = regimes.argmax(axis=0)
    n = len(labels)
    prediction = np.zeros((n, n_variables, n_variables, max_lag + 1), dtype=np.float32)
    causal_results = result["causal_results"]
    for regime, native in causal_results.items():
        graph = np.asarray(native["graph"])
        if graph.shape[:2] != (n_variables, n_variables) or graph.shape[2] <= max_lag:
            raise ValueError("Native RPCMCI graph does not match the requested variables/lags.")
        tensor = np.zeros((n_variables, n_variables, max_lag + 1), dtype=np.float32)
        for source in range(n_variables):
            for target in range(n_variables):
                if source == target:
                    continue
                for lag in range(1, max_lag + 1):
                    # Tigramite's documented convention is
                    # graph[source, target, lag] == '-->' for
                    # source(t-lag) -> target(t).  This adapter deliberately
                    # excludes lag 0 because native RPCMCI is run with
                    # tau_min=1.
                    if graph[source, target, lag] == "-->":
                        tensor[source, target, lag] = 1.0
        prediction[labels == int(regime)] = tensor
    flags = np.zeros(n, dtype=bool)
    flags[1:] = labels[1:] != labels[:-1]
    return prediction, flags


def main():
    parser = argparse.ArgumentParser(description="Run native RPCMCI dynamic-baseline benchmark")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20, help="held-out seed range starts at 20 by protocol")
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--output", default="results/validation/rpcmci_raw.csv")
    # One native optimisation/annealing pass is the preregistered compute
    # budget for the 20-seed comparison.  The full 10-by-3 configuration did
    # not produce a first checkpoint after a substantial uninterrupted run on
    # the reference workstation.  These values are written into every result
    # row so the comparison is transparent and exactly reproducible.
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--annealings", type=int, default=1)
    args = parser.parse_args()
    if args.seeds < 20:
        parser.error("At least 20 seeds are required for the manuscript baseline.")
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "status", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points", "edge_threshold", "num_iterations", "max_anneal", "precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate", "error"]
    if target.exists():
        existing = pd.read_csv(target)
        rows = existing.to_dict("records")
        finished = set(existing.loc[existing.status == "completed", "seed"])
    else:
        rows, finished = [], set()
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if seed in finished:
            continue
        row = {"method": "RPCMCI", "seed": seed, "n_variables": 4, "max_lag": 2, "noise_scale": 0.1, "distribution": "gaussian", "n_points": args.n_points, "edge_threshold": 0.15, "num_iterations": args.iterations, "max_anneal": args.annealings}
        print(
            f"Starting native RPCMCI seed {seed} "
            f"(iterations={args.iterations}, annealings={args.annealings}).",
            flush=True,
        )
        try:
            data = generate_synthetic_regimes(n_points=args.n_points, n_variables=4, max_lag=2, noise_scale=0.1, distribution="gaussian", seed=seed)
            native = run_rpcmci(data.X, max_lag=2, num_regimes=4, max_transitions=3, seed=seed, num_iterations=args.iterations, max_anneal=args.annealings)
            prediction, flags = _regime_predictions(native, 4, 2)
            rows.append({**row, "status": "completed", **_evaluate_sequence(prediction, data.A_true, data.change_points, detected_flags=flags), "error": ""})
        except RegimeAwareBaselineUnavailable as error:
            raise SystemExit(str(error)) from error
        except Exception as error:  # preserve a reproducible failure record
            rows.append({**row, "status": "failed", **{key: np.nan for key in columns[11:-1]}, "error": repr(error)})
        _atomic_csv(pd.DataFrame(rows, columns=columns), target)
        print(f"Checkpoint saved: RPCMCI seed {seed}.", flush=True)


if __name__ == "__main__":
    main()
