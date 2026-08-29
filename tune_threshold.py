import numpy as np
import pandas as pd

from regression_baseline import (
    estimate_graph,
    MAX_LAG,
    WINDOW_SIZE,
    VARIABLES,
    DATA_PATH,
)

from regime_ground_truth_tensor import (
    build_time_indexed_ground_truth,
)

from evaluate_model import evaluate_static_graph


# ==================================================
# Configuration
# ==================================================

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

VALIDATION_TIMES = [
    3500,
    3700,
    3900,
    4100,
    4249,
]


# ==================================================
# Load data
# ==================================================

df = pd.read_csv(DATA_PATH)

X = df[VARIABLES].to_numpy(
    dtype=np.float64
)


# ==================================================
# Ground truth
# ==================================================

A_true, _, _ = (
    build_time_indexed_ground_truth()
)


print("===== Multi-Time Validation Threshold Tuning =====")

print(
    "Validation times:",
    VALIDATION_TIMES,
)


# ==================================================
# Evaluate each threshold
# ==================================================

all_results = []

for threshold in THRESHOLDS:

    f1_scores = []
    precision_scores = []
    recall_scores = []
    shd_scores = []

    print(
        f"\n--- Threshold {threshold:.2f} ---"
    )

    for validation_time in VALIDATION_TIMES:

        A_pred = estimate_graph(
            X=X,
            estimate_time=validation_time,
            window_size=WINDOW_SIZE,
            max_lag=MAX_LAG,
            threshold=threshold,
        )

        scores = evaluate_static_graph(
            A_pred,
            A_true[validation_time],
            threshold=0.0,
        )

        f1_scores.append(
            scores["f1"]
        )

        precision_scores.append(
            scores["precision"]
        )

        recall_scores.append(
            scores["recall"]
        )

        shd_scores.append(
            scores["shd"]
        )

        print(
            f"t={validation_time} | "
            f"F1={scores['f1']:.3f} | "
            f"Precision={scores['precision']:.3f} | "
            f"Recall={scores['recall']:.3f} | "
            f"SHD={scores['shd']}"
        )

    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(
        precision_scores
    )
    avg_recall = np.mean(
        recall_scores
    )
    avg_shd = np.mean(
        shd_scores
    )

    all_results.append(
        {
            "threshold": threshold,
            "avg_f1": avg_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_shd": avg_shd,
        }
    )


# ==================================================
# Print summary
# ==================================================

print("\n===== Validation Summary =====")

for result in all_results:

    print(
        f"threshold={result['threshold']:.2f} | "
        f"avg_F1={result['avg_f1']:.3f} | "
        f"avg_precision={result['avg_precision']:.3f} | "
        f"avg_recall={result['avg_recall']:.3f} | "
        f"avg_SHD={result['avg_shd']:.3f}"
    )


# ==================================================
# Select best threshold
#
# Primary criterion:
#     highest average F1
#
# Tie-breaker:
#     lowest average SHD
# ==================================================

best = sorted(
    all_results,
    key=lambda x: (
        -x["avg_f1"],
        x["avg_shd"],
    ),
)[0]


print("\n===== Selected Threshold =====")

print(
    "Threshold:",
    best["threshold"],
)

print(
    "Average F1:",
    best["avg_f1"],
)

print(
    "Average Precision:",
    best["avg_precision"],
)

print(
    "Average Recall:",
    best["avg_recall"],
)

print(
    "Average SHD:",
    best["avg_shd"],
)
