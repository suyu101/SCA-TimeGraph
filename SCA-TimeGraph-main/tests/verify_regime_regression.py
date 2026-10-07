import pandas as pd
import numpy as np


# --------------------------------------------------
# Load data
# --------------------------------------------------

path = "Datasets/regime_change_a1.csv"

df = pd.read_csv(path)

X = df[["X1", "X2", "X3", "X4"]].to_numpy(
    dtype=np.float64
)

CHANGE_POINT = 2500


# --------------------------------------------------
# Simple linear regression helper
# --------------------------------------------------

def regression_coefficients(X_target, predictors):
    """
    Fit:

        target = beta_0 + beta_1*x1 + beta_2*x2 + ...

    Returns coefficients.
    """

    X_design = np.column_stack(
        [np.ones(len(X_target)), *predictors]
    )

    beta, *_ = np.linalg.lstsq(
        X_design,
        X_target,
        rcond=None,
    )

    return beta


# --------------------------------------------------
# BEFORE
# --------------------------------------------------

# Valid times: 2 ... CHANGE_POINT-1

t_before = np.arange(2, CHANGE_POINT)

target_before = X[t_before, 2]       # X3(t)
x4_before = X[t_before, 3]           # X4(t)
x2_lag1_before = X[t_before - 1, 1]  # X2(t-1)

beta_before = regression_coefficients(
    target_before,
    [
        x4_before,
        x2_lag1_before,
    ],
)


# --------------------------------------------------
# AFTER
# --------------------------------------------------

t_after = np.arange(
    CHANGE_POINT,
    len(X),
)

target_after = X[t_after, 2]          # X3(t)
x4_after = X[t_after, 3]              # X4(t)
x2_lag1_after = X[t_after - 1, 1]     # X2(t-1)

beta_after = regression_coefficients(
    target_after,
    [
        x4_after,
        x2_lag1_after,
    ],
)


# --------------------------------------------------
# Print
# --------------------------------------------------

print("===== Regime Regression Check =====")

print("\nEquation being tested:")
print("X3(t) = beta0 + beta1*X4(t) + beta2*X2(t-1)")

print("\nExpected BEFORE:")
print("beta1 ≈ 0.35")
print("beta2 ≈ 0.00")

print("\nEstimated BEFORE:")
print("Intercept:", beta_before[0])
print("X4(t) coefficient:", beta_before[1])
print("X2(t-1) coefficient:", beta_before[2])

print("\nExpected AFTER:")
print("beta1 ≈ 0.00")
print("beta2 ≈ 0.35")

print("\nEstimated AFTER:")
print("Intercept:", beta_after[0])
print("X4(t) coefficient:", beta_after[1])
print("X2(t-1) coefficient:", beta_after[2])
