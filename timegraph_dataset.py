import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Load TimeGraph data
# --------------------------------------------------

DATA_PATH = (
    "Datasets/A1/Gaussian/4 variable/Lag 2/"
    "linear_ts_n500_vars4_lag2.csv"
)

df = pd.read_csv(DATA_PATH)

VARIABLES = ["X1", "X2", "X3", "X4"]

X = df[VARIABLES].to_numpy(dtype=np.float32)
time = df["time"].to_numpy()


# --------------------------------------------------
# 2. Ground-truth causal links
# --------------------------------------------------

TRUE_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X4",  0, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2",  0, "X1"): 0.40,
}


# --------------------------------------------------
# 3. Convert links to A_true tensor
# --------------------------------------------------

n_vars = len(VARIABLES)
max_lag = 2

A_true = np.zeros(
    (n_vars, n_vars, max_lag + 1),
    dtype=np.float32
)

variable_to_idx = {
    name: idx
    for idx, name in enumerate(VARIABLES)
}

for (source, lag, target), weight in TRUE_LINKS.items():

    source_idx = variable_to_idx[source]
    target_idx = variable_to_idx[target]

    # Convert TimeGraph lag convention:
    #  0  -> 0
    # -1  -> 1
    # -2  -> 2
    lag_idx = abs(lag)

    A_true[source_idx, target_idx, lag_idx] = weight


# --------------------------------------------------
# 4. Print results
# --------------------------------------------------

print("===== TimeGraph Dataset =====")

print("\nX shape:")
print(X.shape)

print("\nA_true shape:")
print(A_true.shape)

print("\nVariables:")
print(VARIABLES)

print("\nNon-zero entries in A_true:")

for source in range(n_vars):
    for target in range(n_vars):
        for lag in range(max_lag + 1):

            weight = A_true[source, target, lag]

            if weight != 0:
                print(
                    f"{VARIABLES[source]} -> "
                    f"{VARIABLES[target]} "
                    f"(lag={lag}) "
                    f"weight={weight}"
                )
