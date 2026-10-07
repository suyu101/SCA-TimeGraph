"""Tune matched streaming baselines only on seeds 0--19.

Configurations are selected independently for every (d, L) using the same
canonical dynamic tuning cell and the same AUPRC-first deterministic rule used
for Phase-1 SCA tuning.  The fixed edge decision threshold is never tuned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.baselines.streaming_linear import (GlobalEMAConfig, GlobalRateEMA, KGLSConfig, KGLSPreviousGraph, RLSConfig, RecursiveLeastSquares)
from sca.experiments.synthetic import generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence


def _window(d: int, lag: int, multiplier: float) -> int:
    return max(200, int(np.ceil(multiplier * d * lag)))


def candidates(d: int, lag: int):
    for multiplier in (2.5, 5.0):
        for strength in (0.1, 1.0):
            yield "KGLS ridge-to-previous", {"regression_window": _window(d, lag, multiplier), "prior_strength": strength}
    for forgetting in (0.97, 0.99):
        yield "RLS forgetting", {"forgetting_factor": forgetting, "initial_covariance": 100.0}
    for multiplier in (2.5, 5.0):
        for rate in (0.05, 0.25):
            yield "Global-rate EMA", {"regression_window": _window(d, lag, multiplier), "rate": rate, "ridge_alpha": 1.0}


def build(method: str, lag: int, params: dict):
    if method == "KGLS ridge-to-previous":
        return KGLSPreviousGraph(lag, KGLSConfig(**params))
    if method == "RLS forgetting":
        return RecursiveLeastSquares(lag, RLSConfig(**params))
    if method == "Global-rate EMA":
        return GlobalRateEMA(lag, GlobalEMAConfig(**params))
    raise ValueError(method)


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune streaming baselines on seeds 0--19 only")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[4, 10, 20, 50])
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--output-dir", default="results/phase1/baseline_tuning")
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_start + args.seeds > 20:
        parser.error("tuning is restricted to seeds 0--19")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw.csv"
    columns = ["method", "seed", "n_variables", "max_lag", "parameters", "n_points", "precision", "recall", "f1", "shd", "auroc", "auprc"]
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=columns)
    completed = set(raw[["method", "seed", "n_variables", "max_lag", "parameters", "n_points"]].itertuples(index=False, name=None)) if not raw.empty else set()
    rows = raw.to_dict("records")
    for d in args.dimensions:
        for lag in args.lags:
            for method, params in candidates(d, lag):
                encoded = json.dumps(params, sort_keys=True)
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    key = (method, seed, d, lag, encoded, args.n_points)
                    if key in completed:
                        continue
                    data = generate_synthetic_regimes(n_points=args.n_points, n_variables=d, max_lag=lag, noise_scale=0.1, distribution="gaussian", seed=seed)
                    metrics = _evaluate_sequence(build(method, lag, params).predict(data.X), data.A_true, data.change_points)
                    rows.append({"method": method, "seed": seed, "n_variables": d, "max_lag": lag, "parameters": encoded, "n_points": args.n_points, **{name: metrics[name] for name in columns if name in metrics}})
                    completed.add(key)
                    _atomic(pd.DataFrame(rows, columns=columns), raw_path)
                    print(f"checkpoint {method}, d={d}, L={lag}, seed={seed}", flush=True)
    frame = pd.DataFrame(rows, columns=columns)
    summary = frame.groupby(["method", "n_variables", "max_lag", "parameters"], as_index=False).agg(auprc_mean=("auprc", "mean"), f1_mean=("f1", "mean"), shd_mean=("shd", "mean"))
    summary = summary.sort_values(["method", "n_variables", "max_lag", "auprc_mean", "f1_mean", "shd_mean"], ascending=[True, True, True, False, False, True])
    summary.to_csv(output / "summary.csv", index=False)
    selected = summary.groupby(["method", "n_variables", "max_lag"], as_index=False).first()
    selected.to_json(output / "selected.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
