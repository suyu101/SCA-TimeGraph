"""Joint, null-calibrated SCA tuning for Phase 1.

This runner is deliberately separate from the historical one-factor sensitivity
study. It uses only tuning seeds, calibrates each candidate's detector from
stationary data, and records every candidate rather than silently selecting a
single favourable factor setting.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.experiments.synthetic import generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter


def dimension_aware_config(d, lag, *, window_multiplier, ridge_alpha, persistence_decay, deviation_scale):
    """Use at least ``window_multiplier * d * L`` observations for each OLS/Ridge fit."""
    regression_window = max(200, int(np.ceil(window_multiplier * d * lag)))
    return SCAConfig(
        regression_window=regression_window,
        detection_window=max(50, regression_window // 2),
        threshold=1.0,  # replaced by calibration before dynamic evaluation
        persistence_decay=persistence_decay,
        deviation_scale=deviation_scale,
        ridge_alpha=ridge_alpha,
    )


def null_scores(*, d, lag, seeds, n_points, config):
    """Return only score opportunities where the detector is eligible to fire."""
    scores = []
    for seed in seeds:
        data = generate_synthetic_regimes(
            n_points=n_points, n_variables=d, max_lag=lag, change_points=(),
            noise_scale=0.1, distribution="gaussian", seed=int(seed),
        )
        _, values, _, _ = SelectiveCausalAdapter(lag, config).predict(data.X)
        times = np.arange(len(values))
        eligible = (times % config.update_interval == 0) & (times + 1 >= 2 * config.detection_window)
        scores.extend(values[eligible].tolist())
    return np.asarray(scores, dtype=float)


def threshold_at_far(scores, target_far=0.05):
    """Choose a conservative threshold for `score >= threshold` detection."""
    if len(scores) == 0:
        raise ValueError("No eligible null detector scores were generated.")
    ordered = np.sort(scores)
    # Strictly above the empirical 95th percentile prevents ties from
    # exceeding the requested FAR with the estimator's >= decision rule.
    index = min(len(ordered) - 1, int(np.ceil((1.0 - target_far) * len(ordered))))
    return float(np.nextafter(ordered[index], np.inf))


def candidate_grid():
    """A fractional factorial joint screen; unregularised OLS is the comparator.

    The high-dimensional OLS candidate is intentionally excluded from this
    screen because it is mathematically ill-posed at d=50, L=4. It is run once
    as the documented before-condition, not repeatedly as a tuning candidate.
    """
    yield from (
        (2.5, 0.10, 0.90, 0.05), (2.5, 0.10, 0.97, 0.10),
        (2.5, 1.00, 0.90, 0.10), (2.5, 1.00, 0.97, 0.05),
        (5.0, 0.10, 0.90, 0.10), (5.0, 0.10, 0.97, 0.05),
        (5.0, 1.00, 0.90, 0.05), (5.0, 1.00, 0.97, 0.10),
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 1 joint tuning on seeds 0-19 only")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[50])
    parser.add_argument("--lags", type=int, nargs="+", default=[4])
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--output-dir", default="results/phase1/tuning")
    parser.add_argument("--target-far", type=float, default=0.05)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_start + args.seeds > 20:
        parser.error("Phase 1 tuning is restricted to seeds 0-19.")
    if not 0 < args.target_far < 1:
        parser.error("--target-far must be in (0, 1).")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    raw_path = output / "joint_tuning_raw.csv"
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records")
    candidate_keys = ["n_variables", "max_lag", "window_multiplier", "ridge_alpha", "persistence_decay", "deviation_scale", "seed"]
    completed = set(tuple(row[key] for key in candidate_keys) for row in rows) if rows else set()
    for d in args.dimensions:
        for lag in args.lags:
            for multiplier, ridge_alpha, persistence, deviation in candidate_grid():
                base = dimension_aware_config(
                    d, lag, window_multiplier=multiplier, ridge_alpha=ridge_alpha,
                    persistence_decay=persistence, deviation_scale=deviation,
                )
                scores = null_scores(d=d, lag=lag, seeds=seeds, n_points=args.n_points, config=base)
                threshold = threshold_at_far(scores, args.target_far)
                config = SCAConfig(**{**asdict(base), "threshold": threshold})
                null_far = float(np.mean(scores >= threshold))
                for seed in seeds:
                    key = (d, lag, multiplier, ridge_alpha, persistence, deviation, seed)
                    if key in completed:
                        continue
                    data = generate_synthetic_regimes(
                        n_points=args.n_points, n_variables=d, max_lag=lag,
                        noise_scale=0.1, distribution="gaussian", seed=seed,
                    )
                    prediction, _, flags, _ = SelectiveCausalAdapter(lag, config).predict(data.X)
                    metrics = _evaluate_sequence(prediction, data.A_true, data.change_points, detected_flags=flags)
                    rows.append({
                        "seed": seed, "n_variables": d, "max_lag": lag,
                        "window_multiplier": multiplier, "ridge_alpha": ridge_alpha,
                        "persistence_decay": persistence, "deviation_scale": deviation,
                        "regression_window": config.regression_window,
                        "detection_window": config.detection_window,
                        "threshold": threshold, "null_far": null_far, **metrics,
                    })
                    completed.add(key)
                    # Preserve completed seeds if a long high-dimensional run
                    # is interrupted; duplicate-free aggregation happens below.
                    pd.DataFrame(rows).to_csv(raw_path, index=False)
                frame = pd.DataFrame(rows)
                frame.to_csv(raw_path, index=False)
                print(f"checkpoint d={d}, L={lag}, cW={multiplier}, ridge={ridge_alpha}", flush=True)
    frame = pd.DataFrame(rows)
    keys = ["n_variables", "max_lag", "window_multiplier", "ridge_alpha", "persistence_decay", "deviation_scale", "regression_window", "detection_window", "threshold", "null_far"]
    summary = frame.groupby(keys, as_index=False).agg(
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
        shd_mean=("shd", "mean"), shd_sd=("shd", "std"),
    )
    summary = summary[summary.null_far <= args.target_far].sort_values(
        ["n_variables", "max_lag", "auprc_mean", "f1_mean", "shd_mean"],
        ascending=[True, True, False, False, True],
    )
    summary.to_csv(output / "joint_tuning_summary.csv", index=False)
    selected = summary.groupby(["n_variables", "max_lag"], as_index=False).first()
    selected.to_json(output / "selected_by_dimension.json", orient="records", indent=2)
    (output / "protocol.json").write_text(json.dumps({
        "tuning_seeds": list(seeds), "target_null_far": args.target_far,
        "primary_endpoint": "AUPRC; F1 then SHD are deterministic tie-breakers",
        "candidate_count_per_dimension_lag": 8,
        "screen_design": "fractional factorial over window multiplier, ridge, persistence, and deviation scale",
        "dynamic_tuning_cell": {"noise_scale": 0.1, "distribution": "gaussian"},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
