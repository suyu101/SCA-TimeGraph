"""Freeze the held-out SCA configuration from the completed sensitivity study."""

import json
from pathlib import Path

import pandas as pd


def main():
    source = Path("results/validation/sensitivity_mean_std.csv")
    raw_source = Path("results/validation/sensitivity_raw.csv")
    if not source.is_file() or not raw_source.is_file():
        raise SystemExit("Completed sensitivity raw and summary CSVs are required before configuration selection.")
    summary = pd.read_csv(source)
    raw = pd.read_csv(raw_source)
    if (
        set(raw.get("seed", [])) != set(range(20))
        or set(raw.get("synthetic_protocol", [])) != {"v2_recurring_graphs_measurement_snr"}
        or len(raw) != 300
    ):
        raise SystemExit("Refusing configuration selection: sensitivity must contain the completed v2 20-seed protocol.")
    # Primary endpoint is F1. For the change threshold, constrain the mean
    # false-alarm rate to <= 5% before choosing the highest-F1 setting.
    choices = {}
    for parameter in ("detection_window", "regression_window", "persistence_decay", "deviation_scale"):
        candidates = summary[summary.parameter.eq(parameter)]
        if candidates.empty:
            raise SystemExit(f"Missing sensitivity results for {parameter}.")
        row = candidates.sort_values(["f1_mean", "shd_mean"], ascending=[False, True]).iloc[0]
        choices[parameter] = row["value"].item()
    threshold = summary[summary.parameter.eq("threshold")]
    eligible = threshold[threshold.false_alarm_rate_mean <= .05]
    if eligible.empty:
        raise SystemExit("No threshold candidate meets the preregistered false-alarm constraint.")
    row = eligible.sort_values(["f1_mean", "shd_mean"], ascending=[False, True]).iloc[0]
    choices["threshold"] = row["value"].item()
    report = {
        "selection_data": str(source),
        "tuning_seeds": list(range(20)),
        "protocol": "v2_recurring_graphs_measurement_snr",
        "primary_endpoint": "mean F1; threshold additionally constrained to mean false-alarm rate <= 0.05",
        "selected_config": choices,
        "held_out_seeds": list(range(20, 40)),
    }
    output = Path("results/validation/synthetic_config_selection.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
