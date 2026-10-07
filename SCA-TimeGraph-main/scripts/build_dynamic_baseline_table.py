"""Build a manuscript table for the seed-matched native CD-NOD comparison."""

from pathlib import Path

import pandas as pd


METRICS = ("precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate")
CANONICAL = {"n_variables": 4, "max_lag": 2, "noise_scale": 0.1, "distribution": "gaussian"}


def plus_minus(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def main():
    synthetic_path = Path("results/validation/synthetic_raw.csv")
    cdnod_path = Path("results/validation/cdnod_raw.csv")
    if not synthetic_path.is_file() or not cdnod_path.is_file():
        raise SystemExit(
            "Run the completed held-out synthetic study and native CD-NOD benchmark "
            "before building Table IV."
        )
    synthetic = pd.read_csv(synthetic_path)
    cdnod = pd.read_csv(cdnod_path)
    subset = synthetic.copy()
    for field, value in CANONICAL.items():
        subset = subset[subset[field] == value]
    subset = subset[subset.seed.isin(range(20, 40))]
    cdnod = cdnod[(cdnod.method == "CD-NOD") & (cdnod.status == "completed") & (cdnod.seed.isin(range(20, 40)))].copy()
    required_seeds = set(range(20, 40))
    for name, frame in {"canonical SCA/Rolling subset": subset, "CD-NOD": cdnod}.items():
        if set(frame.seed) != required_seeds:
            raise SystemExit(f"Refusing dynamic table: {name} must contain exactly held-out seeds 20--39.")
    rows = []
    for method, frame in [("SCA", subset[subset.method == "SCA"]), ("Rolling OLS", subset[subset.method == "Rolling OLS"]), ("CD-NOD", cdnod)]:
        if set(frame.seed) != required_seeds:
            raise SystemExit(f"Refusing dynamic table: {method} has incomplete seed coverage.")
        result = {"Method": method, "d": 4, "L": 2, "Noise": 0.1, "Distribution": "Gaussian"}
        for metric in METRICS:
            values = frame[metric].dropna()
            result[metric.replace("_", " ").title()] = plus_minus(values.mean(), values.std(ddof=1)) if len(values) else "N/A"
        rows.append(result)
    target = Path("results/manuscript_table_iv_dynamic_baseline.csv")
    pd.DataFrame(rows).to_csv(target, index=False)
    print(target)


if __name__ == "__main__":
    main()
