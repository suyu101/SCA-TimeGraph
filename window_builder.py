import pandas as pd
import numpy as np


def create_temporal_windows(
    X,
    time,
    max_lag,
    train_end,
    val_end,
):
    """
    Create chronological temporal windows and assign each
    window according to its target time.

    Parameters
    ----------
    X : np.ndarray
        Shape: (n_time, n_vars)

    time : np.ndarray
        Shape: (n_time,)

    max_lag : int
        Number of historical time steps required.

    train_end : int
        Last time index belonging to training.

    val_end : int
        Last time index belonging to validation.

    Returns
    -------
    dict
        train, validation, and test windows.
    """

    train_windows = []
    val_windows = []
    test_windows = []

    train_targets = []
    val_targets = []
    test_targets = []

    for t in range(max_lag, len(X)):

        window = X[t - max_lag : t + 1]

        target_time = time[t]

        if target_time < train_end:

            train_windows.append(window)
            train_targets.append(target_time)

        elif target_time < val_end:

            val_windows.append(window)
            val_targets.append(target_time)

        else:

            test_windows.append(window)
            test_targets.append(target_time)

    return {
        "X_train": np.asarray(train_windows, dtype=np.float32),
        "X_val": np.asarray(val_windows, dtype=np.float32),
        "X_test": np.asarray(test_windows, dtype=np.float32),
        "time_train": np.asarray(train_targets),
        "time_val": np.asarray(val_targets),
        "time_test": np.asarray(test_targets),
    }


if __name__ == "__main__":

    # --------------------------------------------
    # Load dataset
    # --------------------------------------------

    csv_path = (
        "Datasets/A1/Gaussian/4 variable/Lag 2/"
        "linear_ts_n500_vars4_lag2.csv"
    )

    variables = ["X1", "X2", "X3", "X4"]

    df = pd.read_csv(csv_path)

    X = df[variables].to_numpy(dtype=np.float32)
    time = df["time"].to_numpy()

    max_lag = 2

    # --------------------------------------------
    # Define chronological split
    # --------------------------------------------

    n = len(X)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # --------------------------------------------
    # Create windows
    # --------------------------------------------

    splits = create_temporal_windows(
        X=X,
        time=time,
        max_lag=max_lag,
        train_end=train_end,
        val_end=val_end,
    )

    # --------------------------------------------
    # Display results
    # --------------------------------------------

    print("===== Windowed Temporal Dataset =====")

    print("\nTrain:")
    print("Shape:", splits["X_train"].shape)
    print(
        "Target times:",
        splits["time_train"][0],
        "->",
        splits["time_train"][-1],
    )

    print("\nValidation:")
    print("Shape:", splits["X_val"].shape)
    print(
        "Target times:",
        splits["time_val"][0],
        "->",
        splits["time_val"][-1],
    )

    print("\nTest:")
    print("Shape:", splits["X_test"].shape)
    print(
        "Target times:",
        splits["time_test"][0],
        "->",
        splits["time_test"][-1],
    )

    # --------------------------------------------
    # Inspect boundary samples
    # --------------------------------------------

    print("\n===== Boundary Check =====")

    print(
        "Last train target:",
        splits["time_train"][-1],
    )

    print(
        "First validation target:",
        splits["time_val"][0],
    )

    print(
        "Last validation target:",
        splits["time_val"][-1],
    )

    print(
        "First test target:",
        splits["time_test"][0],
    )

    # --------------------------------------------
    # Inspect first validation window
    # --------------------------------------------

    print("\n===== First Validation Window =====")

    print(
        "Target time:",
        splits["time_val"][0],
    )

    print(
        "Window:",
        splits["X_val"][0],
    )

print("\n===== Last Training Window =====")

print(
    "Target time:",
    splits["time_train"][-1],
)

print(
    "Expected window times:",
    [
        splits["time_train"][-1] - 2,
        splits["time_train"][-1] - 1,
        splits["time_train"][-1],
    ],
)

print(
    "Window:",
    splits["X_train"][-1],
)
