"""Run reproducible multi-seed validation for the paper's empirical section."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.experiments.validation import run_multiseed_validation, run_sensitivity_analysis


def main():
    parser = argparse.ArgumentParser(description="Run SCA empirical validation")
    parser.add_argument("--study", choices=("synthetic", "sensitivity", "all"), default="all")
    parser.add_argument("--seeds", type=int, default=20, help="number of independent seeds (minimum 20 for paper tables)")
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--seed-start", type=int, default=None, help="first seed; defaults to 20 for held-out synthetic testing and 0 for sensitivity tuning")
    parser.add_argument("--quick", action="store_true", help="small smoke test; never use its output in the manuscript")
    args = parser.parse_args()
    if args.seeds < 20 and not args.quick:
        parser.error("--seeds must be at least 20 for a statistical paper run; use --quick only for smoke tests")
    # Tuning and final evaluation must use disjoint seeds even when both
    # studies are requested in one invocation.  An explicit --seed-start is
    # retained for small development runs, not for the paper protocol.
    synthetic_seed_start = 20 if args.seed_start is None else args.seed_start
    sensitivity_seed_start = 0 if args.seed_start is None else args.seed_start
    if args.quick:
        dimensions, lags, sigmas, distributions = (4,), (1, 2), (0.1,), ("gaussian",)
    else:
        dimensions, lags, sigmas, distributions = (4, 10, 20, 50), (1, 2, 4), (0.05, 0.1, 0.25, 0.5), ("gaussian", "student_t", "uniform")
    if args.study in ("synthetic", "all"):
        _, summary = run_multiseed_validation(seeds=range(synthetic_seed_start, synthetic_seed_start + args.seeds), dimensions=dimensions, lags=lags, noise_scales=sigmas, distributions=distributions, n_points=args.n_points, output_dir=args.output_dir)
        print(f"Wrote {len(summary)} mean +/- standard-deviation rows to {args.output_dir}")
    if args.study in ("sensitivity", "all"):
        _, summary = run_sensitivity_analysis(seeds=range(sensitivity_seed_start, sensitivity_seed_start + args.seeds), n_points=args.n_points, output_dir=args.output_dir)
        print(f"Wrote {len(summary)} hyperparameter-sensitivity rows to {args.output_dir}")


if __name__ == "__main__":
    main()
