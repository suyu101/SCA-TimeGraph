import numpy as np
import pandas as pd


# --------------------------------------------
# Load dataset
# --------------------------------------------

csv_path = (
    "Datasets/A1/Gaussian/4 variable/Lag 2/"
    "linear_ts_n500_vars4_lag2.csv"
)

df = pd.read_csv(csv_path)

variables = ["X1", "X2", "X3", "X4"]

X = df[variables].to_numpy(dtype=np.float32)
time = df["time"].to_numpy()


# --------------------------------------------
# Maximum lag
# --------------------------------------------

max_lag = 2


# --------------------------------------------
# Pick one time point
# --------------------------------------------

t = 10


# --------------------------------------------
# Create historical window
# --------------------------------------------

start = t - max_lag
end = t + 1

window = X[start:end]


print("===== Temporal Window Example =====")

print("\nTarget time:", t)

print("\nWindow times:")
print(time[start:end])

print("\nWindow shape:")
print(window.shape)

print("\nWindow values:")

for i, current_time in enumerate(time[start:end]):

    print(
        f"time={current_time}: "
        f"{window[i]}"
    )


# --------------------------------------------
# Show lag interpretation
# --------------------------------------------

print("\n===== Lag Interpretation =====")

print(
    "X1(t-2) corresponds to:",
    X[t - 2, 0]
)

print(
    "X3(t-1) corresponds to:",
    X[t - 1, 2]
)

print(
    "X4(t) corresponds to:",
    X[t, 3]
)

print(
    "X2(t) corresponds to:",
    X[t, 1]
)
