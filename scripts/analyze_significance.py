"""Seed-level paired randomization tests with Holm correction and bootstrap CIs.

The factorial cells within a seed share the same random realization and are not
independent replicates.  We therefore average each method across the complete
factorial design first, then compare the resulting 20 paired seed summaries.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p values in original row order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_ci(values, seed=20260916, samples=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)))


def paired_sign_flip_test(differences, seed=20260916, samples=100_000):
    """Two-sided paired randomization p-value under random sign assignment."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        return float("nan"), float("nan")
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(differences)))
    null_means = (signs * differences).mean(axis=1)
    p_value = (np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1) / (samples + 1)
    return observed, float(p_value)


def main():
    raw = pd.read_csv("results/validation/synthetic_raw.csv")
    required_protocol = "v2_recurring_graphs_measurement_snr"
    keys = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points"]
    expected_rows = 20 * 4 * 3 * 4 * 3 * 2
    condition_sizes = raw.groupby(["method", "n_variables", "max_lag", "noise_scale", "distribution"]).size()
    if (
        raw.seed.nunique() != 20
        or set(raw.seed) != set(range(20, 40))
        or "edge_threshold" not in raw
        or "synthetic_protocol" not in raw
        or set(raw.synthetic_protocol.dropna()) != {required_protocol}
        or len(raw) != expected_rows
        or raw.duplicated(keys).any()
        or not (condition_sizes == 20).all()
    ):
        raise SystemExit("Need held-out seeds 20--39 from the corrected thresholded protocol.")
    rows = []
    for metric in ("precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate"):
        # One independent paired observation per held-out seed.  All factorial
        # cells are equally weighted within a method/seed summary.
        seed_means = raw.groupby(["seed", "method"], as_index=False)[metric].mean()
        wide = seed_means.pivot(index="seed", columns="method", values=metric).dropna()
        # A baseline without a change detector has no valid paired samples for
        # detector-only metrics.  Do not manufacture a test from empty data.
        if not {"SCA", "Rolling OLS"}.issubset(wide.columns) or wide.empty:
            continue
        difference = wide["SCA"] - wide["Rolling OLS"]
        statistic, p_value = paired_sign_flip_test(difference, seed=20260916 + len(rows))
        effect = difference.mean() / difference.std(ddof=1) if difference.std(ddof=1) else float("nan")
        ci_low, ci_high = bootstrap_ci(difference)
        rows.append({"comparison": "SCA vs Rolling OLS", "metric": metric, "n_pairs": len(difference), "mean_paired_difference_sca_minus_comparator": difference.mean(), "ci_95_low": ci_low, "ci_95_high": ci_high, "test": "paired sign-flip randomization (100,000 draws)", "statistic": statistic, "p_value_uncorrected": p_value, "effect_size": effect, "effect_size_type": "Cohen dz"})
    # CD-NOD is evaluated only on the pre-specified canonical dynamic cell.
    # Retain the seed-level pairing; do not pool it with the factorial OLS
    # comparison, which has a different estimand.
    cdnod_path = Path("results/validation/cdnod_raw.csv")
    if cdnod_path.is_file():
        cdnod = pd.read_csv(cdnod_path)
        canonical = raw[
            (raw.n_variables == 4) & (raw.max_lag == 2)
            & (raw.noise_scale == 0.1) & (raw.distribution == "gaussian")
            & (raw.method == "SCA")
        ]
        cdnod = cdnod[(cdnod.method == "CD-NOD") & (cdnod.status == "completed")]
        for metric in ("precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation"):
            paired = canonical[["seed", metric]].merge(
                cdnod[["seed", metric]], on="seed", suffixes=("_sca", "_cdnod")
            ).dropna()
            if len(paired) != 20:
                continue
            difference = paired[f"{metric}_sca"] - paired[f"{metric}_cdnod"]
            statistic, p_value = paired_sign_flip_test(difference, seed=20261000 + len(rows))
            effect = difference.mean() / difference.std(ddof=1) if difference.std(ddof=1) else float("nan")
            ci_low, ci_high = bootstrap_ci(difference, seed=20261000 + len(rows))
            rows.append({"comparison": "SCA vs CD-NOD (canonical dynamic cell)", "metric": metric, "n_pairs": len(difference), "mean_paired_difference_sca_minus_comparator": difference.mean(), "ci_95_low": ci_low, "ci_95_high": ci_high, "test": "paired sign-flip randomization (100,000 draws)", "statistic": statistic, "p_value_uncorrected": p_value, "effect_size": effect, "effect_size_type": "Cohen dz"})
    if not rows:
        raise SystemExit("No complete SCA/Rolling OLS pairs were found.")
    table = pd.DataFrame(rows)
    table["p_value_holm"] = holm_adjust(table["p_value_uncorrected"])
    output = Path("results/significance_tests.csv")
    table.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
