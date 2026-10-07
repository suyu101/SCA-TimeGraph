"""Legacy threshold tuning, exposed as an explicit callable experiment."""

import numpy as np
import pandas as pd

from ..baselines.regression_baseline import DATA_PATH, MAX_LAG, VARIABLES, WINDOW_SIZE, estimate_graph
from ..ground_truth.regime_ground_truth_tensor import build_time_indexed_ground_truth
from .evaluate_model import evaluate_static_graph


THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
VALIDATION_TIMES = (3500, 3700, 3900, 4100, 4249)


def tune_thresholds(data_path=DATA_PATH, thresholds=THRESHOLDS, validation_times=VALIDATION_TIMES):
    """Evaluate candidate thresholds and return a summary plus the selected row."""
    frame = pd.read_csv(data_path)
    missing = sorted(set(VARIABLES).difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required variables: {missing}")
    observations = frame.loc[:, VARIABLES].to_numpy(dtype=np.float64)
    truth, _, _ = build_time_indexed_ground_truth()
    rows = []
    for threshold in thresholds:
        scores = []
        for time in validation_times:
            if time >= len(observations) or time >= len(truth):
                raise ValueError(f"Validation time {time} is outside the supplied sequence.")
            prediction = estimate_graph(observations, time, WINDOW_SIZE, MAX_LAG, threshold)
            scores.append(evaluate_static_graph(prediction, truth[time], threshold=0.0))
        rows.append({
            "threshold": threshold,
            "avg_f1": float(np.mean([item["f1"] for item in scores])),
            "avg_precision": float(np.mean([item["precision"] for item in scores])),
            "avg_recall": float(np.mean([item["recall"] for item in scores])),
            "avg_shd": float(np.mean([item["shd"] for item in scores])),
        })
    summary = pd.DataFrame(rows)
    selected = summary.sort_values(["avg_f1", "avg_shd"], ascending=[False, True]).iloc[0].to_dict()
    return summary, selected


if __name__ == "__main__":
    table, best = tune_thresholds()
    print(table.to_string(index=False))
    print("Selected threshold:", best["threshold"])
