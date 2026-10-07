"""Fail closed unless every manuscript experiment artifact is complete and coherent."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path("results")
PROTOCOL = "v2_recurring_graphs_measurement_snr"


def check(condition, message, findings):
    if not condition:
        findings.append(message)


def read_csv(path, findings):
    if not path.is_file():
        findings.append(f"Missing required artifact: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:
        findings.append(f"Unreadable CSV {path}: {error}")
        return None


def main():
    findings = []
    synthetic = read_csv(ROOT / "validation" / "synthetic_raw.csv", findings)
    sensitivity = read_csv(ROOT / "validation" / "sensitivity_raw.csv", findings)
    if synthetic is not None:
        required = {"method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "synthetic_protocol"}
        check(required.issubset(synthetic), "Synthetic raw file has an incomplete schema.", findings)
        if required.issubset(synthetic):
            keys = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution"]
            check(set(synthetic.seed) == set(range(20, 40)), "Synthetic held-out seeds must be exactly 20--39.", findings)
            check(set(synthetic.n_variables) == {4, 10, 20, 50}, "Synthetic dimensions must be 4, 10, 20, 50.", findings)
            check(set(synthetic.max_lag) == {1, 2, 4}, "Synthetic lags must be 1, 2, 4.", findings)
            check(set(synthetic.noise_scale) == {0.05, 0.1, 0.25, 0.5}, "Synthetic noise scales are incomplete.", findings)
            check(set(synthetic.distribution) == {"gaussian", "student_t", "uniform"}, "Synthetic distributions are incomplete.", findings)
            check(set(synthetic.synthetic_protocol) == {PROTOCOL}, "Synthetic results use an incompatible protocol.", findings)
            expected = 20 * 4 * 3 * 4 * 3 * 2
            check(len(synthetic) == expected, f"Synthetic raw requires {expected} method rows; found {len(synthetic)}.", findings)
            check(not synthetic.duplicated(keys).any(), "Synthetic raw contains duplicate method-condition rows.", findings)
            check(set(synthetic.method) >= {"SCA", "Rolling OLS"}, "Synthetic comparison is missing SCA or Rolling OLS.", findings)
    if sensitivity is not None:
        required = {"parameter", "value", "seed", "synthetic_protocol"}
        check(required.issubset(sensitivity), "Sensitivity raw file has an incomplete schema.", findings)
        if required.issubset(sensitivity):
            check(set(sensitivity.seed) == set(range(20)), "Sensitivity tuning seeds must be exactly 0--19.", findings)
            check(set(sensitivity.synthetic_protocol) == {PROTOCOL}, "Sensitivity results use an incompatible protocol.", findings)
            check(len(sensitivity) == 300, f"Sensitivity requires 300 rows; found {len(sensitivity)}.", findings)
    for artifact in (
        ROOT / "manuscript_table_ii.csv", ROOT / "manuscript_table_iii.csv",
        ROOT / "manuscript_table_iv_dynamic_baseline.csv",
        ROOT / "significance_tests.csv", ROOT / "publication_figures" / "figure_scaling_f1.svg",
        ROOT / "publication_figures" / "figure_hyperparameter_sensitivity.svg",
        ROOT / "causal_chambers" / "wt_walks_v1" / "tuning" / "selection.json",
        ROOT / "causal_chambers" / "wt_walks_v1" / "tuning" / "held_out_metrics.csv",
    ):
        check(artifact.is_file(), f"Missing required artifact: {artifact}", findings)
    cdnod = read_csv(ROOT / "validation" / "cdnod_raw.csv", findings)
    if cdnod is not None:
        required = {"method", "status", "seed"}
        check(required.issubset(cdnod), "CD-NOD result file has an incomplete schema.", findings)
        if required.issubset(cdnod):
            completed = cdnod[(cdnod.method == "CD-NOD") & (cdnod.status == "completed")]
            check(set(completed.seed) == set(range(20, 40)), "CD-NOD requires 20 completed held-out benchmark seeds (20--39).", findings)
    if findings:
        print("MANUSCRIPT READINESS: NOT READY")
        print(*[f"- {finding}" for finding in findings], sep="\n")
        return 1
    print("MANUSCRIPT READINESS: READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
