"""Null (no-change) experiment for SCA.

This script evaluates SCA on stationary data — data where the
causal graph never changes.  A good method should have a low
false-alarm rate (FAR) in this setting.

Purpose
-------
The null experiment serves two roles:
  1. Sanity check: FAR should be < 5% for the preregistered threshold
  2. Upper bound: reported metrics bound the worst-case mismatch cost

The false-alarm rate is the fraction of time steps at which a change
is spuriously detected.

Usage
-----
    python scripts/run_null_experiment.py [--seeds N] [--n-points T]

Output
------
    results/null_experiment.csv
        FAR per seed and its mean/std summary
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter
from sca.experiments.benchmark_configs import (
    generate_benchmark,
    BenchmarkConfig,
    BENCHMARK_NO_CHANGE,
)
from sca.evaluation.adaptation_metrics import PAPER_EDGE_THRESHOLD


def run_null_experiment(
    seeds,
    n_points,
    n_variables,
    max_lag,
    noise_scale,
    config: SCAConfig,
):
    """Run SCA on stationary (no-change) data for each seed.

    Returns a list of dicts with per-seed results.
    """
    rows = []
    for seed in seeds:
        benchmark_cfg = BenchmarkConfig(
            benchmark_type=BENCHMARK_NO_CHANGE,
            n_points=n_points,
            n_variables=n_variables,
            max_lag=max_lag,
            noise_scale=noise_scale,
            seed=seed,
        )
        data = generate_benchmark(benchmark_cfg)

        # Reset adapter for each seed (no state carries over)
        adapter = SelectiveCausalAdapter(max_lag, config)
        _, scores, detected, _ = adapter.predict(data.X)

        far = float(np.mean(detected))
        rows.append({
            "seed": seed,
            "n_points": n_points,
            "n_variables": n_variables,
            "max_lag": max_lag,
            "noise_scale": noise_scale,
            "false_alarm_rate": far,
            "n_detected": int(np.sum(detected)),
            "threshold": config.threshold,
            "detection_window": config.detection_window,
            "regression_window": config.regression_window,
        })
        print(f"  seed={seed}: FAR={far:.4f} ({int(np.sum(detected))}/{n_points} detections)")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Null (no-change) experiment for SCA")
    parser.add_argument("--seeds", type=int, default=20, help="number of random seeds")
    parser.add_argument("--seed-start", type=int, default=100,
                        help="first seed (kept separate from validation seeds 0-39)")
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--n-variables", type=int, default=4)
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=PAPER_EDGE_THRESHOLD,
                        help="edge threshold (default: preregistered paper threshold 0.15)")
    parser.add_argument("--detection-window", type=int, default=100)
    parser.add_argument("--regression-window", type=int, default=200)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config = SCAConfig(
        threshold=args.threshold,
        detection_window=args.detection_window,
        regression_window=args.regression_window,
    )

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    print(f"Null experiment: {args.seeds} seeds, n_points={args.n_points}, "
          f"threshold={args.threshold:.3f}")

    rows = run_null_experiment(
        seeds=seeds,
        n_points=args.n_points,
        n_variables=args.n_variables,
        max_lag=args.max_lag,
        noise_scale=args.noise_scale,
        config=config,
    )

    # Compute summary
    fars = [r["false_alarm_rate"] for r in rows]
    mean_far = float(np.mean(fars))
    std_far = float(np.std(fars, ddof=1))
    print(f"\nNull experiment summary:")
    print(f"  Seeds: {args.seed_start}..{args.seed_start + args.seeds - 1}")
    print(f"  FAR mean +/- std: {mean_far:.4f} +/- {std_far:.4f}")
    print(f"  Max FAR: {max(fars):.4f}")

    # Pass/fail at 5% (strict) and 20% (lenient)
    strict_ok = mean_far < 0.05
    lenient_ok = mean_far < 0.20
    print(f"  Passes 5% FAR criterion: {'YES' if strict_ok else 'NO (WARNING)'}")
    print(f"  Passes 20% FAR criterion: {'YES' if lenient_ok else 'NO (FAIL)'}")

    # Write CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "null_experiment.csv"

    # Add summary row
    rows.append({
        "seed": "MEAN",
        "n_points": args.n_points,
        "n_variables": args.n_variables,
        "max_lag": args.max_lag,
        "noise_scale": args.noise_scale,
        "false_alarm_rate": mean_far,
        "n_detected": "",
        "threshold": args.threshold,
        "detection_window": args.detection_window,
        "regression_window": args.regression_window,
    })
    rows.append({
        "seed": "STD",
        "n_points": args.n_points,
        "n_variables": args.n_variables,
        "max_lag": args.max_lag,
        "noise_scale": args.noise_scale,
        "false_alarm_rate": std_far,
        "n_detected": "",
        "threshold": args.threshold,
        "detection_window": args.detection_window,
        "regression_window": args.regression_window,
    })

    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {output}")

    if not lenient_ok:
        raise SystemExit(
            f"FAR={mean_far:.4f} exceeds 20% criterion. "
            f"Review SCAConfig threshold settings."
        )


if __name__ == "__main__":
    main()
