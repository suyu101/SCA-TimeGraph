"""Held-out d=50, L=4 before/after evaluation for Phase 1.

Reads the configuration frozen by ``phase1_joint_tune.py`` and evaluates only
seeds 20--39. The legacy configuration is included solely as the documented
before-condition; it is not retuned here.
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

from sca.experiments.synthetic import generate_synthetic_regimes
from sca.experiments.validation import _evaluate_sequence, _rolling_predictions
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Held-out Phase 1 d=50, L=4 evaluation")
    parser.add_argument("--selection", default="results/phase1/tuning_d50_l4/selected_by_dimension.json")
    parser.add_argument("--output-dir", default="results/phase1/heldout_d50_l4")
    parser.add_argument("--n-points", type=int, default=5000)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if len(selection) != 1 or selection[0]["n_variables"] != 50 or selection[0]["max_lag"] != 4:
        raise SystemExit("Selection must contain exactly the frozen d=50, L=4 tuning result.")
    picked = selection[0]
    selected = SCAConfig(
        regression_window=int(picked["regression_window"]),
        detection_window=int(picked["detection_window"]),
        threshold=float(picked["threshold"]),
        persistence_decay=float(picked["persistence_decay"]),
        deviation_scale=float(picked["deviation_scale"]),
        ridge_alpha=float(picked["ridge_alpha"]),
    )
    legacy = SCAConfig(
        regression_window=200, detection_window=50, threshold=0.30,
        persistence_decay=0.97, deviation_scale=0.10, ridge_alpha=0.0,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "heldout_raw.csv"
    columns = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points", "regression_window", "detection_window", "threshold", "persistence_decay", "deviation_scale", "ridge_alpha", "precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_recall", "per_edge_delay", "per_edge_delay_censor_rate", "auroc", "auprc", "change_detection_rate", "false_alarm_rate"]
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=columns)
    rows = existing.to_dict("records")
    completed = set(zip(existing.get("method", []), existing.get("seed", [])))
    methods = {"SCA legacy OLS": legacy, "SCA dimension-aware ridge": selected}
    for seed in range(20, 40):
        data = generate_synthetic_regimes(n_points=args.n_points, n_variables=50, max_lag=4, noise_scale=0.1, distribution="gaussian", seed=seed)
        for label, config in methods.items():
            if (label, seed) in completed:
                continue
            prediction, _, flags, _ = SelectiveCausalAdapter(4, config).predict(data.X)
            rows.append({"method": label, "seed": seed, "n_variables": 50, "max_lag": 4, "noise_scale": 0.1, "distribution": "gaussian", "n_points": args.n_points, "regression_window": config.regression_window, "detection_window": config.detection_window, "threshold": config.threshold, "persistence_decay": config.persistence_decay, "deviation_scale": config.deviation_scale, "ridge_alpha": config.ridge_alpha, **_evaluate_sequence(prediction, data.A_true, data.change_points, detected_flags=flags)})
            atomic_csv(pd.DataFrame(rows, columns=columns), raw_path)
            print(f"checkpoint method={label!r}, seed={seed}", flush=True)
        if ("Rolling OLS (W=250, U=10)", seed) not in completed:
            prediction = _rolling_predictions(data.X, 4, window=250, interval=10)
            rows.append({"method": "Rolling OLS (W=250, U=10)", "seed": seed, "n_variables": 50, "max_lag": 4, "noise_scale": 0.1, "distribution": "gaussian", "n_points": args.n_points, "regression_window": 250, "detection_window": np.nan, "threshold": np.nan, "persistence_decay": np.nan, "deviation_scale": np.nan, "ridge_alpha": 0.0, **_evaluate_sequence(prediction, data.A_true, data.change_points)})
            atomic_csv(pd.DataFrame(rows, columns=columns), raw_path)
            print(f"checkpoint method='Rolling OLS (W=250, U=10)', seed={seed}", flush=True)
    raw = pd.DataFrame(rows, columns=columns)
    metrics = ["precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_recall", "per_edge_delay", "per_edge_delay_censor_rate", "auroc", "auprc", "change_detection_rate", "false_alarm_rate"]
    summary = raw.groupby("method")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(output / "heldout_summary.csv", index=False)


if __name__ == "__main__":
    main()
