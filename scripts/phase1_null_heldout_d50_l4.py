"""Held-out stationary null control for the frozen d=50, L=4 SCA configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca.experiments.synthetic import generate_synthetic_regimes
from sca.models.selective_adaptation import SCAConfig, SelectiveCausalAdapter


def main():
    parser = argparse.ArgumentParser(description="Held-out null control for frozen Phase 1 SCA")
    parser.add_argument("--selection", default="results/phase1/tuning_d50_l4/selected_by_dimension.json")
    parser.add_argument("--output", default="results/phase1/heldout_d50_l4/null_heldout.csv")
    parser.add_argument("--n-points", type=int, default=5000)
    args = parser.parse_args()
    picked = json.loads(Path(args.selection).read_text(encoding="utf-8"))[0]
    config = SCAConfig(
        regression_window=int(picked["regression_window"]), detection_window=int(picked["detection_window"]),
        threshold=float(picked["threshold"]), persistence_decay=float(picked["persistence_decay"]),
        deviation_scale=float(picked["deviation_scale"]), ridge_alpha=float(picked["ridge_alpha"]),
    )
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(target) if target.exists() else pd.DataFrame()
    done = set(old.get("seed", []))
    rows = old.to_dict("records")
    for seed in range(20, 40):
        if seed in done:
            continue
        data = generate_synthetic_regimes(n_points=args.n_points, n_variables=50, max_lag=4, change_points=(), noise_scale=0.1, distribution="gaussian", seed=seed)
        _, scores, flags, _ = SelectiveCausalAdapter(4, config).predict(data.X)
        time = np.arange(args.n_points)
        eligible = (time % config.update_interval == 0) & (time + 1 >= 2 * config.detection_window)
        rows.append({"seed": seed, "false_alarm_rate_all_times": float(flags.mean()), "false_alarm_rate_eligible_updates": float(flags[eligible].mean()), "n_eligible_updates": int(eligible.sum()), "threshold": config.threshold, "regression_window": config.regression_window, "detection_window": config.detection_window, "ridge_alpha": config.ridge_alpha})
        pd.DataFrame(rows).to_csv(target, index=False)
        print(f"checkpoint seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {"seed": "MEAN", "false_alarm_rate_all_times": frame.false_alarm_rate_all_times.mean(), "false_alarm_rate_eligible_updates": frame.false_alarm_rate_eligible_updates.mean()}
    frame.loc[len(frame)] = {"seed": "SD", "false_alarm_rate_all_times": frame.iloc[:-1].false_alarm_rate_all_times.std(ddof=1), "false_alarm_rate_eligible_updates": frame.iloc[:-1].false_alarm_rate_eligible_updates.std(ddof=1)}
    frame.to_csv(target, index=False)


if __name__ == "__main__":
    main()
