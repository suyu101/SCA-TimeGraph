import numpy as np
import pandas as pd

from regression_baseline import estimate_graph


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

from evaluate_model import evaluate_static_graph

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
