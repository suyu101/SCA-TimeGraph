"""Seed-level paired sign-flip tests for the Phase 1 d=50, L=4 comparison."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sca.experiments.validation import _ranking_metrics  # import validates project path


def sign_flip(diff, seed, draws=100_000):
    rng = np.random.default_rng(seed)
    observed = float(np.mean(diff))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(diff)))
    p = (np.count_nonzero(np.abs((signs * diff).mean(axis=1)) >= abs(observed)) + 1) / (draws + 1)
    boot = rng.choice(diff, size=(10_000, len(diff)), replace=True).mean(axis=1)
    return observed, float(p), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def holm(values):
    values = np.asarray(values); order = np.argsort(values); result = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index]); result[index] = min(1.0, running)
    return result


def main():
    raw = pd.read_csv("results/phase1/heldout_d50_l4/heldout_raw.csv")
    expected = set(range(20, 40))
    if set(raw.seed) != expected:
        raise SystemExit("Requires exactly held-out seeds 20-39.")
    metrics = ["precision", "recall", "f1", "shd", "stable_edge_recall", "per_edge_delay", "per_edge_delay_censor_rate", "auroc", "auprc"]
    rows = []
    for comparator in ("SCA legacy OLS", "Rolling OLS (W=250, U=10)"):
        for number, metric in enumerate(metrics):
            wide = raw[raw.method.isin(["SCA dimension-aware ridge", comparator])].pivot(index="seed", columns="method", values=metric)
            difference = (wide["SCA dimension-aware ridge"] - wide[comparator]).to_numpy()
            estimate, p, low, high = sign_flip(difference, 20261006 + number)
            rows.append({"comparison": f"ridge SCA minus {comparator}", "metric": metric, "n_pairs": len(difference), "mean_paired_difference": estimate, "ci_95_low": low, "ci_95_high": high, "p_value_uncorrected": p})
    result = pd.DataFrame(rows); result["p_value_holm"] = holm(result.p_value_uncorrected)
    result.to_csv("results/phase1/heldout_d50_l4/paired_stats.csv", index=False)
    print("results/phase1/heldout_d50_l4/paired_stats.csv")


if __name__ == "__main__":
    main()
