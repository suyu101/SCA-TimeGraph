"""Statistical validation runners used to produce paper-ready CSV tables."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import numpy as np
import pandas as pd

from ..evaluation.evaluation import precision_recall_f1, structural_hamming_distance
from ..models.selective_adaptation import SCAConfig, SelectiveCausalAdapter, estimate_lagged_graph
from .synthetic import SYNTHETIC_PROTOCOL_VERSION, generate_synthetic_regimes


# Selected solely from tuning seeds 0--19. The held-out factorial assessment
# uses seeds 20--39 and must not alter these values.
SELECTED_SYNTHETIC_CONFIG = SCAConfig(
    detection_window=50,
    regression_window=200,
    threshold=0.30,
    persistence_decay=0.97,
    deviation_scale=0.10,
)


def _rolling_predictions(X, max_lag, window=250, interval=10):
    n, d = X.shape
    out = np.zeros((n, d, d, max_lag + 1), dtype=np.float32)
    current = out[0]
    for t in range(max_lag, n):
        if t % interval == 0:
            current = estimate_lagged_graph(X[max(0, t - window + 1):t + 1], max_lag)
        out[t] = current
    return out


def _ranking_metrics(scores, labels):
    """Threshold-free AUROC and average precision without optional dependencies."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    # Average ranks handle tied coefficient magnitudes correctly for AUROC.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    auroc = (ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    descending = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[descending]
    cumulative = np.cumsum(ordered_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    auprc = float(precision[ordered_labels].sum() / positives)
    return float(auroc), auprc


def _evaluate_sequence(
    prediction, truth, change_points, *, detected_flags=None, edge_threshold=0.15,
):
    """Evaluate thresholded structure over regimes and recovery after each shift.

    A non-zero coefficient is not evidence for an edge: ordinary least squares
    yields almost no exact zeroes.  ``edge_threshold`` therefore defines the
    preregistered graph decision rule for both predictions and truth.
    """
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have the same shape")
    n = len(truth)
    # Regular temporal sampling prevents a final graph from masking poor
    # transient adaptation, while including each regime boundary explicitly.
    times = set(range(0, n, max(1, n // 50)))
    times.update(max(0, point - 1) for point in change_points)
    times.update(min(n - 1, point + 100) for point in change_points)
    scores = [precision_recall_f1(prediction[t], truth[t], threshold=edge_threshold) for t in sorted(times)]
    delays = []
    stable_recall = []
    per_edge_delays = []
    censored_changed_edges = 0
    detections = []
    positive_detection_region = np.zeros(n, dtype=bool)
    for index, point in enumerate(change_points):
        next_point = change_points[index + 1] if index + 1 < len(change_points) else n
        horizon = min(next_point, point + 500)
        positive_detection_region[point:horizon] = True
        before = np.abs(truth[point - 1]) > edge_threshold
        after = np.abs(truth[point]) > edge_threshold
        stable = before & after
        # Require a sustained correct post-shift graph, not a one-step spike.
        recovery_time = None
        for t in range(point, horizon):
            predicted = np.abs(prediction[t]) > edge_threshold
            if np.array_equal(predicted, after):
                recovery_time = t - point
                break
        delays.append(float(recovery_time if recovery_time is not None else horizon - point))
        changed = before != after
        for source, target, lag in zip(*np.where(changed)):
            wanted_present = bool(after[source, target, lag])
            recovered = None
            for t in range(point, horizon):
                present = abs(prediction[t, source, target, lag]) > edge_threshold
                if present == wanted_present:
                    recovered = t - point
                    break
            if recovered is None:
                recovered = horizon - point
                censored_changed_edges += 1
            per_edge_delays.append(float(recovered))
        if stable.any():
            stable_recall.append(float((np.abs(prediction[point:horizon])[:, stable] > edge_threshold).mean()))
        if detected_flags is not None:
            indices = np.flatnonzero(detected_flags[point:horizon])
            detections.append(float(len(indices) > 0))
    false_alarm_rate = np.nan
    if detected_flags is not None and (~positive_detection_region).any():
        false_alarm_rate = float(np.mean(detected_flags[~positive_detection_region]))
    lagged_mask = np.ones(truth.shape[1:], dtype=bool)
    lagged_mask[:, :, 0] = False
    diagonal = np.arange(lagged_mask.shape[0])
    lagged_mask[diagonal, diagonal, :] = False
    ranking_auroc, ranking_auprc = _ranking_metrics(
        np.abs(prediction[sorted(times)][:, lagged_mask]).ravel(),
        (np.abs(truth[sorted(times)][:, lagged_mask]) > 0).ravel(),
    )
    stable_value = float(np.mean(stable_recall)) if stable_recall else np.nan
    return {
        "precision": float(np.mean([score["precision"] for score in scores])),
        "recall": float(np.mean([score["recall"] for score in scores])),
        "f1": float(np.mean([score["f1"] for score in scores])),
        "shd": float(np.mean([structural_hamming_distance(prediction[t], truth[t], threshold=edge_threshold) for t in sorted(times)])),
        "adaptation_delay": float(np.mean(delays)) if delays else np.nan,
        "stable_edge_recall": stable_value,
        # Backward-compatible field name. New reports must label this metric
        # stable-edge recall rather than implying a distinct preservation score.
        "stable_edge_preservation": stable_value,
        "per_edge_delay": float(np.mean(per_edge_delays)) if per_edge_delays else np.nan,
        "per_edge_delay_censor_rate": float(censored_changed_edges / len(per_edge_delays)) if per_edge_delays else np.nan,
        "auroc": ranking_auroc,
        "auprc": ranking_auprc,
        "change_detection_rate": float(np.mean(detections)) if detections else np.nan,
        "false_alarm_rate": false_alarm_rate,
    }
def _summary(frame):
    if frame.empty:
        return pd.DataFrame()
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column not in {"seed", "n_variables", "max_lag", "noise_scale", "n_points", "edge_threshold"}]
    groups = frame.groupby([column for column in ["method", "n_variables", "max_lag", "noise_scale", "distribution"] if column in frame], dropna=False)
    result = groups[numeric].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in result.columns]
    return result


def _sensitivity_summary(frame):
    """Aggregate sensitivity rows, including the valid empty development case."""
    if frame.empty:
        return pd.DataFrame()
    metrics = ["precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate"]
    result = frame.groupby(["parameter", "value"])[metrics].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in result.columns]
    return result


def _atomic_csv(frame, path):
    """Write a CSV atomically so Ctrl+C cannot leave a corrupt checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_checkpoint(path, columns):
    """Load a compatible checkpoint or begin a fresh table."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        # Results from the pre-checkpoint runner cannot be safely matched to a
        # run configuration, so leave them untouched and start a new schema.
        backup = path.with_name(path.stem + "_legacy" + path.suffix)
        if not backup.exists():
            os.replace(path, backup)
        return pd.DataFrame(columns=columns)
    return frame[columns]


def run_multiseed_validation(
    *, seeds=range(20), dimensions=(4, 10, 20, 50), lags=(1, 2, 4), noise_scales=(0.05, 0.1, 0.25, 0.5),
    distributions=("gaussian", "student_t", "uniform"), n_points=5000, output_dir="results/validation", config=None,
):
    """Run the factorial synthetic benchmark and write raw + mean±SD CSV files."""
    config = config or SELECTED_SYNTHETIC_CONFIG
    seeds = tuple(seeds)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "synthetic_raw.csv"
    summary_path = directory / "synthetic_mean_std.csv"
    columns = ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points", "synthetic_protocol", "edge_threshold", "precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate"]
    raw = _load_checkpoint(raw_path, columns)
    completed = set(tuple(row) for row in raw[["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points"]].itertuples(index=False, name=None))
    total = len(seeds) * len(dimensions) * len(lags) * len(noise_scales) * len(distributions)
    done = len(completed) // 2
    rows = raw.to_dict("records")
    print(
        f"Resuming synthetic validation: {done}/{total} conditions already checkpointed.",
        flush=True,
    )
    for seed in seeds:
        for d in dimensions:
            for lag in lags:
                for sigma in noise_scales:
                    for distribution in distributions:
                        condition = (int(seed), d, lag, sigma, distribution, n_points)
                        expected = {("SCA", *condition), ("Rolling OLS", *condition)}
                        if expected.issubset(completed):
                            continue
                        data = generate_synthetic_regimes(n_points=n_points, n_variables=d, max_lag=lag, noise_scale=sigma, distribution=distribution, seed=int(seed))
                        sca_prediction, _, sca_flags, _ = SelectiveCausalAdapter(lag, config).predict(data.X)
                        methods = {"SCA": (sca_prediction, sca_flags), "Rolling OLS": (_rolling_predictions(data.X, lag), None)}
                        for method, (prediction, flags) in methods.items():
                            row = {"method": method, "seed": int(seed), "n_variables": d, "max_lag": lag, "noise_scale": sigma, "distribution": distribution, "n_points": n_points, "synthetic_protocol": SYNTHETIC_PROTOCOL_VERSION, "edge_threshold": 0.15, **_evaluate_sequence(prediction, data.A_true, data.change_points, detected_flags=flags)}
                            rows.append(row)
                            completed.add(tuple(row[key] for key in ["method", "seed", "n_variables", "max_lag", "noise_scale", "distribution", "n_points"]))
                        raw = pd.DataFrame(rows, columns=columns)
                        _atomic_csv(raw, raw_path)
                        _atomic_csv(_summary(raw), summary_path)
                        done += 1
                        print(f"Checkpoint saved: {done}/{total} synthetic conditions completed.", flush=True)
    raw = pd.DataFrame(rows, columns=columns)
    summary = _summary(raw)
    _atomic_csv(raw, raw_path)
    _atomic_csv(summary, summary_path)
    return raw, summary


def run_sensitivity_analysis(*, seeds=range(20), n_points=5000, output_dir="results/validation", base_config=None):
    """One-factor-at-a-time sensitivity analysis for D, W, theta, delta, epsilon."""
    base = base_config or SCAConfig()
    seeds = tuple(seeds)
    grids = {
        "detection_window": (50, 100, 200), "regression_window": (50, 100, 200),
        # Scores are regression-coefficient changes on standardized windows;
        # this range brackets their empirical null/shift overlap.  The former
        # 0.04--0.16 grid saturated the detector and was not informative.
        "threshold": (0.20, 0.25, 0.30), "persistence_decay": (0.7, 0.9, 0.97),
        "deviation_scale": (0.025, 0.05, 0.1),
    }
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "sensitivity_raw.csv"
    summary_path = directory / "sensitivity_mean_std.csv"
    columns = ["parameter", "value", "seed", "n_points", "synthetic_protocol", "edge_threshold", "precision", "recall", "f1", "shd", "adaptation_delay", "stable_edge_preservation", "change_detection_rate", "false_alarm_rate"]
    raw = _load_checkpoint(raw_path, columns)
    completed = set(tuple(row) for row in raw[["parameter", "value", "seed", "n_points"]].itertuples(index=False, name=None))
    rows = raw.to_dict("records")
    total = sum(len(values) for values in grids.values()) * len(seeds)
    print(
        f"Resuming sensitivity analysis: {len(completed)}/{total} evaluations already checkpointed.",
        flush=True,
    )
    for parameter, values in grids.items():
        for value in values:
            config = replace(base, **{parameter: value})
            for seed in seeds:
                if (parameter, value, int(seed), n_points) in completed:
                    continue
                data = generate_synthetic_regimes(n_points=n_points, seed=int(seed))
                prediction, _, flags, _ = SelectiveCausalAdapter(2, config).predict(data.X)
                metrics = _evaluate_sequence(prediction, data.A_true, data.change_points, detected_flags=flags)
                rows.append({"parameter": parameter, "value": value, "seed": int(seed), "n_points": n_points, "synthetic_protocol": SYNTHETIC_PROTOCOL_VERSION, "edge_threshold": 0.15, **metrics})
                raw = pd.DataFrame(rows, columns=columns)
                _atomic_csv(raw, raw_path)
                summary = _sensitivity_summary(raw)
                _atomic_csv(summary, summary_path)
                print(f"Checkpoint saved: sensitivity {parameter}={value}, seed {seed}.", flush=True)
    raw = pd.DataFrame(rows, columns=columns)
    summary = _sensitivity_summary(raw)
    _atomic_csv(raw, raw_path)
    _atomic_csv(summary, summary_path)
    return raw, summary
