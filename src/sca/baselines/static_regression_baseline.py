"""Static regression baseline.

This module deliberately has no import-time I/O: importing :mod:`sca.baselines`
must be safe in notebooks, tests, and command-line tools.
"""

import numpy as np

from ..models.selective_adaptation import estimate_lagged_graph


def estimate_static_graph(X, *, max_lag=2, window_size=None, threshold=0.20):
    """Estimate a graph at the final observation using only preceding samples."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (time, variables)")
    if len(X) <= max_lag:
        return np.zeros((X.shape[1], X.shape[1], max_lag + 1), dtype=np.float32)
    history = X[-(window_size or len(X)):]
    graph = estimate_lagged_graph(history, max_lag)
    graph[np.abs(graph) < threshold] = 0.0
    return graph


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from ..evaluation.evaluate_model import evaluate_static_graph as evaluate

    parser = argparse.ArgumentParser(description="Fit the static regression baseline")
    parser.add_argument("csv_path")
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    columns = [column for column in df.columns if column != "time"]
    prediction = estimate_static_graph(
        df[columns].to_numpy(), max_lag=args.max_lag,
        window_size=args.window_size, threshold=args.threshold,
    )
    print(prediction)

"""



# ==================================================
# Configuration
# ==================================================

VARIABLES = ["X1", "X2", "X3", "X4"]

MAX_LAG = 2

WINDOW_SIZE = 500

THRESHOLD = 0.20

DATA_PATH = (
    "Datasets/A1/Gaussian/4 variable/Lag 2/"
    "linear_ts_n5000_vars4_lag2.csv"
)


# ==================================================
# Load data
# ==================================================

df = pd.read_csv(DATA_PATH)

X = df[VARIABLES].to_numpy(
    dtype=np.float64
)

print("===== Static A1 Baseline =====")

print(
    "Dataset shape:",
    X.shape,
)

print(
    "Threshold:",
    THRESHOLD,
)

print(
    "Window size:",
    WINDOW_SIZE,
)


# ==================================================
# Estimate graph at end of dataset
# ==================================================

estimate_time = len(X) - 1

A_pred = estimate_graph(
    X=X,
    estimate_time=estimate_time,
    window_size=WINDOW_SIZE,
    max_lag=MAX_LAG,
    threshold=THRESHOLD,
)


# ==================================================
# Print predicted edges
# ==================================================

print("\n===== Predicted Graph =====")

found = False

for source in range(len(VARIABLES)):

    for target in range(len(VARIABLES)):

        for lag in range(
            MAX_LAG + 1
        ):

            weight = A_pred[
                source,
                target,
                lag,
            ]

            if abs(weight) >= THRESHOLD:

                found = True

                print(
                    f"{VARIABLES[source]} -> "
                    f"{VARIABLES[target]} "
                    f"(lag={lag}) "
                    f"weight={weight:.4f}"
                )

if not found:

    print("No edges detected.")


# ==================================================
# Ground truth
# ==================================================

A_true = np.zeros(
    (
        len(VARIABLES),
        len(VARIABLES),
        MAX_LAG + 1,
    ),
    dtype=np.float32,
)

# X1(t-2) -> X4(t)
A_true[0, 3, 2] = 0.25

# X4(t) -> X3(t)
A_true[3, 2, 0] = 0.35

# X3(t-1) -> X2(t)
A_true[2, 1, 1] = 0.30

# X2(t) -> X1(t)
A_true[1, 0, 0] = 0.40


# ==================================================
# Evaluate
# ==================================================

from ..evaluation.evaluate_model import evaluate_static_graph


results = evaluate_static_graph(
    A_pred,
    A_true,
    threshold=0.0,
)


print(
    "\n===== Static Performance ====="
)

print(
    "Precision:",
    f"{results['precision']:.4f}",
)

print(
    "Recall:",
    f"{results['recall']:.4f}",
)

print(
    "F1:",
    f"{results['f1']:.4f}",
)

print(
    "SHD:",
    results["shd"],
)
"""
