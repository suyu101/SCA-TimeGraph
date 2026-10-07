import pandas as pd
import numpy as np


# --------------------------------------------------
# Load generated regime-change data
# --------------------------------------------------

path = "Datasets/regime_change_a1.csv"

df = pd.read_csv(path)

X = df[["X1", "X2", "X3", "X4"]].to_numpy(
    dtype=np.float64
)

CHANGE_POINT = 2500


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


# --------------------------------------------------
# BEFORE regime
# --------------------------------------------------

# X4(t) -> X3(t)
before_x4_x3 = correlation(
    X[100:CHANGE_POINT, 3],
    X[100:CHANGE_POINT, 2],
)

# X2(t-1) -> X3(t)
before_x2_lag1_x3 = correlation(
    X[101:CHANGE_POINT, 1],
    X[100:CHANGE_POINT - 1, 2],
)


# --------------------------------------------------
# AFTER regime
# --------------------------------------------------

# X4(t) -> X3(t)
after_x4_x3 = correlation(
    X[CHANGE_POINT:, 3],
    X[CHANGE_POINT:, 2],
)

# X2(t-1) -> X3(t)
after_x2_lag1_x3 = correlation(
    X[CHANGE_POINT + 1:, 1],
    X[CHANGE_POINT:-1, 2],
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("===== Regime Change Sanity Check =====")

print("\nChange point:", CHANGE_POINT)

print("\nRelationship: X4(t) -> X3(t)")

print(
    "Before:",
    before_x4_x3,
)

print(
    "After :",
    after_x4_x3,
)

print("\nRelationship: X2(t-1) -> X3(t)")

print(
    "Before:",
    before_x2_lag1_x3,
)

print(
    "After :",
    after_x2_lag1_x3,
)
