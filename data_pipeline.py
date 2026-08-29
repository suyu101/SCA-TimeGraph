import pandas as pd
import numpy as np

from ground_truth import (
    get_linear_equations,
    extract_linear_links,
)


def load_timegraph(
    csv_path,
    variables,
    n_vars,
    max_lag,
):
    """
    Load TimeGraph observations and construct A_true.
    """

    # Load CSV
    df = pd.read_csv(csv_path)

    # Observations
    X = df[variables].to_numpy(dtype=np.float32)

    # Time index
    time = df["time"].to_numpy()

    # Structural equations
    equations = get_linear_equations(
        n_vars=n_vars,
        max_lag=max_lag,
    )

    if not equations:
        raise ValueError(
            f"No equations found for "
            f"n_vars={n_vars}, max_lag={max_lag}"
        )

    # Ground-truth causal links
    true_links = extract_linear_links(equations)

    # Ground-truth tensor
    A_true = np.zeros(
        (n_vars, n_vars, max_lag + 1),
        dtype=np.float32,
    )

    variable_to_idx = {
        name: idx
        for idx, name in enumerate(variables)
    }

    for (source, lag, target), weight in true_links.items():

        source_idx = variable_to_idx[source]
        target_idx = variable_to_idx[target]

        lag_idx = abs(lag)

        A_true[
            source_idx,
            target_idx,
            lag_idx,
        ] = weight

    return {
        "X": X,
        "time": time,
        "A_true": A_true,
        "true_links": true_links,
        "variables": variables,
        "equations": equations,
    }


def chronological_split(
    X,
    time,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Split observations chronologically.

    No shuffling.
    """

    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "X_train_raw": X[:train_end],
        "X_val_raw": X[train_end:val_end],
        "X_test_raw": X[val_end:],

        "time_train_raw": time[:train_end],
        "time_val_raw": time[train_end:val_end],
        "time_test_raw": time[val_end:],
    }


def fit_standardizer(X_train):
    """
    Calculate normalization statistics using
    training data only.
    """

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Prevent division by zero
    std = np.where(std == 0, 1.0, std)

    return mean, std


def apply_standardizer(X, mean, std):
    """
    Apply an already-fitted standardizer.
    """

    return (X - mean) / std


def create_temporal_windows(
    X,
    time,
    max_lag,
    target_start,
    target_end,
):
    """
    Create temporal windows whose target times fall in
    [target_start, target_end).

    Example with max_lag=2:

        target t
        window = [t-2, t-1, t]
    """

    windows = []
    target_times = []

    for t in range(max_lag, len(X)):

        current_time = time[t]

        if target_start <= current_time < target_end:

            window = X[t - max_lag : t + 1]

            windows.append(window)
            target_times.append(current_time)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(target_times),
    )


def prepare_timegraph(
    csv_path,
    variables,
    n_vars,
    max_lag,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Complete TimeGraph preprocessing pipeline.

    Steps:

    1. Load observations
    2. Load ground truth
    3. Chronological split
    4. Fit scaler on training data only
    5. Normalize all splits
    6. Create temporal windows
    """

    # --------------------------------------------
    # 1. Load
    # --------------------------------------------

    dataset = load_timegraph(
        csv_path=csv_path,
        variables=variables,
        n_vars=n_vars,
        max_lag=max_lag,
    )

    X = dataset["X"]
    time = dataset["time"]

    # --------------------------------------------
    # 2. Find split boundaries
    # --------------------------------------------

    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # --------------------------------------------
    # 3. Raw chronological splits
    # --------------------------------------------

    splits = chronological_split(
        X,
        time,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    X_train_raw = splits["X_train_raw"]
    X_val_raw = splits["X_val_raw"]
    X_test_raw = splits["X_test_raw"]

    # --------------------------------------------
    # 4. Fit scaler ONLY on training data
    # --------------------------------------------

    train_mean, train_std = fit_standardizer(
        X_train_raw
    )

    # --------------------------------------------
    # 5. Normalize all splits using
    #    training statistics
    # --------------------------------------------

    X_train_scaled = apply_standardizer(
        X_train_raw,
        train_mean,
        train_std,
    )

    X_val_scaled = apply_standardizer(
        X_val_raw,
        train_mean,
        train_std,
    )

    X_test_scaled = apply_standardizer(
        X_test_raw,
        train_mean,
        train_std,
    )

    # --------------------------------------------
    # 6. Reconstruct scaled full sequence
    #
    # Needed so validation/test windows can use
    # their legitimate historical observations.
    # --------------------------------------------

    X_scaled = apply_standardizer(
        X,
        train_mean,
        train_std,
    )

    # --------------------------------------------
    # 7. Create windows by target time
    # --------------------------------------------

    X_train_windows, time_train = create_temporal_windows(
        X_scaled,
        time,
        max_lag,
        target_start=0,
        target_end=train_end,
    )

    X_val_windows, time_val = create_temporal_windows(
        X_scaled,
        time,
        max_lag,
        target_start=train_end,
        target_end=val_end,
    )

    X_test_windows, time_test = create_temporal_windows(
        X_scaled,
        time,
        max_lag,
        target_start=val_end,
        target_end=n,
    )

    # --------------------------------------------
    # 8. Return complete dataset package
    # --------------------------------------------

    return {
        "X_train": X_train_windows,
        "X_val": X_val_windows,
        "X_test": X_test_windows,

        "time_train": time_train,
        "time_val": time_val,
        "time_test": time_test,

        "A_true": dataset["A_true"],
        "true_links": dataset["true_links"],
        "variables": dataset["variables"],
        "equations": dataset["equations"],

        "train_mean": train_mean,
        "train_std": train_std,
    }


# ==================================================
# Test the complete pipeline
# ==================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python data_pipeline.py <sample_size>")
        print("Example: python data_pipeline.py 500")
        raise SystemExit(1)

    sample_size = int(sys.argv[1])

    variables = ["X1", "X2", "X3", "X4"]
    n_vars = 4
    max_lag = 2

    csv_path = (
        f"Datasets/A1/Gaussian/4 variable/Lag 2/"
        f"linear_ts_n{sample_size}_vars4_lag2.csv"
    )

    data = prepare_timegraph(
        csv_path=csv_path,
        variables=variables,
        n_vars=n_vars,
        max_lag=max_lag,
    )

    print("===== TimeGraph Pipeline =====")
    print("Sample size:", sample_size)

    print("\nTrain:")
    print(
        data["X_train"].shape,
        data["time_train"][0],
        "->",
        data["time_train"][-1],
    )

    print("\nValidation:")
    print(
        data["X_val"].shape,
        data["time_val"][0],
        "->",
        data["time_val"][-1],
    )

    print("\nTest:")
    print(
        data["X_test"].shape,
        data["time_test"][0],
        "->",
        data["time_test"][-1],
    )

    print("\nA_true:")
    print(data["A_true"].shape)
