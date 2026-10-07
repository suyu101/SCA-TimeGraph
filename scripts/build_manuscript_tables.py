"""Build Tables II and III only from completed, validated 20-seed outputs."""

from pathlib import Path
import math
import pandas as pd


def plus_minus(mean, std, digits=3):
    # Rolling OLS has no change-point detector; its detection and false-alarm
    # metrics are intentionally undefined, rather than zero.  Render this
    # explicitly in the manuscript table instead of emitting ``nan ± nan``.
    if not (math.isfinite(float(mean)) and math.isfinite(float(std))):
        return "N/A"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main():
    source = Path("results/validation/synthetic_raw.csv")
    raw = pd.read_csv(source)
    count = raw["seed"].nunique()
    required = {"edge_threshold", "adaptation_delay", "stable_edge_preservation", "false_alarm_rate"}
    keys = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points"]
    expected_rows = 20 * 4 * 3 * 4 * 3 * 2
    condition_sizes = raw.groupby(["method", "n_variables", "max_lag", "noise_scale", "distribution"]).size()
    if (
        count != 20
        or set(raw["seed"]) != set(range(20, 40))
        or not required.issubset(raw.columns)
        or len(raw) != expected_rows
        or raw.duplicated(keys).any()
        or set(raw["n_variables"]) != {4, 10, 20, 50}
        or set(raw["max_lag"]) != {1, 2, 4}
        or set(raw["noise_scale"]) != {0.05, 0.1, 0.25, 0.5}
        or set(raw["distribution"]) != {"gaussian", "student_t", "uniform"}
        or set(raw["method"]) != {"SCA", "Rolling OLS"}
        or not (condition_sizes == 20).all()
    ):
        raise SystemExit("Refusing to create Table II: require held-out seeds 20--39 from the corrected protocol.")
    if raw["synthetic_protocol"].nunique() != 1 or raw["synthetic_protocol"].iloc[0] != "v2_recurring_graphs_measurement_snr":
        raise SystemExit("Refusing to create Table II: synthetic protocol is not the validated recurring/SNR protocol.")
    dimensions = ["method", "n_variables", "max_lag", "noise_scale", "distribution"]
    aggregate = raw.groupby(dimensions, as_index=False).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        shd_mean=("shd", "mean"), shd_std=("shd", "std"),
        detection_mean=("change_detection_rate", "mean"), detection_std=("change_detection_rate", "std"),
        delay_mean=("adaptation_delay", "mean"), delay_std=("adaptation_delay", "std"),
        stable_mean=("stable_edge_preservation", "mean"), stable_std=("stable_edge_preservation", "std"),
        false_alarm_mean=("false_alarm_rate", "mean"), false_alarm_std=("false_alarm_rate", "std"),
    )
    output = pd.DataFrame({
        "Method": aggregate["method"], "d": aggregate["n_variables"], "L": aggregate["max_lag"],
        "Noise": aggregate["noise_scale"], "Distribution": aggregate["distribution"],
        "Precision": [plus_minus(a, b) for a, b in zip(aggregate.precision_mean, aggregate.precision_std)],
        "Recall": [plus_minus(a, b) for a, b in zip(aggregate.recall_mean, aggregate.recall_std)],
        "F1": [plus_minus(a, b) for a, b in zip(aggregate.f1_mean, aggregate.f1_std)],
        "SHD": [plus_minus(a, b) for a, b in zip(aggregate.shd_mean, aggregate.shd_std)],
        "Detection rate": [plus_minus(a, b) for a, b in zip(aggregate.detection_mean, aggregate.detection_std)],
        "Adaptation delay": [plus_minus(a, b) for a, b in zip(aggregate.delay_mean, aggregate.delay_std)],
        "Stable preservation": [plus_minus(a, b) for a, b in zip(aggregate.stable_mean, aggregate.stable_std)],
        "False-alarm rate": [plus_minus(a, b) for a, b in zip(aggregate.false_alarm_mean, aggregate.false_alarm_std)],
    })
    target = Path("results/manuscript_table_ii.csv")
    output.to_csv(target, index=False)
    sensitivity = pd.read_csv("results/validation/sensitivity_mean_std.csv")
    sensitivity_raw = pd.read_csv("results/validation/sensitivity_raw.csv")
    needed = {"parameter", "value", "f1_mean", "f1_std", "shd_mean", "shd_std", "adaptation_delay_mean", "adaptation_delay_std", "stable_edge_preservation_mean", "stable_edge_preservation_std", "change_detection_rate_mean", "change_detection_rate_std", "false_alarm_rate_mean", "false_alarm_rate_std"}
    if not needed.issubset(sensitivity.columns):
        raise SystemExit("Sensitivity CSV does not use the corrected protocol.")
    if sensitivity_raw.seed.nunique() != 20 or set(sensitivity_raw.seed) != set(range(20)):
        raise SystemExit("Refusing to create Table III: require tuning seeds 0--19.")
    if sensitivity_raw.synthetic_protocol.nunique() != 1 or sensitivity_raw.synthetic_protocol.iloc[0] != "v2_recurring_graphs_measurement_snr":
        raise SystemExit("Refusing to create Table III: sensitivity protocol does not match Table II.")
    table_iii = pd.DataFrame({
        "Parameter": sensitivity.parameter,
        "Value": sensitivity.value,
        "F1": [plus_minus(a, b) for a, b in zip(sensitivity.f1_mean, sensitivity.f1_std)],
        "SHD": [plus_minus(a, b) for a, b in zip(sensitivity.shd_mean, sensitivity.shd_std)],
        "Adaptation delay": [plus_minus(a, b) for a, b in zip(sensitivity.adaptation_delay_mean, sensitivity.adaptation_delay_std)],
        "Stable preservation": [plus_minus(a, b) for a, b in zip(sensitivity.stable_edge_preservation_mean, sensitivity.stable_edge_preservation_std)],
        "Detection rate": [plus_minus(a, b) for a, b in zip(sensitivity.change_detection_rate_mean, sensitivity.change_detection_rate_std)],
        "False-alarm rate": [plus_minus(a, b) for a, b in zip(sensitivity.false_alarm_rate_mean, sensitivity.false_alarm_rate_std)],
    })
    table_iii.to_csv("results/manuscript_table_iii.csv", index=False)
    print(target)
    print("results/manuscript_table_iii.csv")


if __name__ == "__main__":
    main()
