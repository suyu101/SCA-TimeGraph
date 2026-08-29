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
    Load a TimeGraph dataset and automatically construct
    the ground-truth causal graph.

    Parameters
    ----------
    csv_path : str
        Path to the TimeGraph CSV file.

    variables : list[str]
        Names of observed variables.

    n_vars : int
        Number of variables.

    max_lag : int
        Maximum causal lag.

    Returns
    -------
    dict
        Contains:
        X          -> observed time-series data
        time       -> time indices
        A_true     -> ground-truth causal graph tensor
        true_links -> ground-truth causal links
        variables  -> variable names
        equations  -> structural equations
    """

    # --------------------------------------------------
    # 1. Load CSV
    # --------------------------------------------------

    df = pd.read_csv(csv_path)

    # --------------------------------------------------
    # 2. Extract observations
    # --------------------------------------------------

    X = df[variables].to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # 3. Extract time
    # --------------------------------------------------

    time = df["time"].to_numpy()

    # --------------------------------------------------
    # 4. Get structural equations
    # --------------------------------------------------

    equations = get_linear_equations(
        n_vars=n_vars,
        max_lag=max_lag,
    )

    if not equations:
        raise ValueError(
            f"No equations found for n_vars={n_vars}, "
            f"max_lag={max_lag}"
        )

    # --------------------------------------------------
    # 5. Extract true causal links
    # --------------------------------------------------

    true_links = extract_linear_links(equations)

    # --------------------------------------------------
    # 6. Build A_true tensor
    #
    # Shape:
    # (source, target, lag)
    #
    # lag index:
    # 0 = contemporaneous
    # 1 = one-step lag
    # 2 = two-step lag
    # --------------------------------------------------

    A_true = np.zeros(
        (n_vars, n_vars, max_lag + 1),
        dtype=np.float32,
    )

    variable_to_idx = {
        name: idx
        for idx, name in enumerate(variables)
    }

    for (source, lag, target), weight in true_links.items():

        if source not in variable_to_idx:
            raise ValueError(
                f"Unknown source variable: {source}"
            )

        if target not in variable_to_idx:
            raise ValueError(
                f"Unknown target variable: {target}"
            )

        source_idx = variable_to_idx[source]
        target_idx = variable_to_idx[target]

        # TimeGraph represents past lags as negative values.
        # Convert -1 -> 1, -2 -> 2, etc.
        lag_idx = abs(lag)

        if lag_idx > max_lag:
            raise ValueError(
                f"Lag {lag} exceeds max_lag={max_lag}"
            )

        A_true[
            source_idx,
            target_idx,
            lag_idx,
        ] = weight

    # --------------------------------------------------
    # 7. Basic validation
    # --------------------------------------------------

    if len(X) != len(time):
        raise ValueError(
            "Number of observations does not match "
            "number of time points."
        )

    if X.shape[1] != n_vars:
        raise ValueError(
            f"Expected {n_vars} variables, "
            f"but found {X.shape[1]}."
        )

    # --------------------------------------------------
    # 8. Return everything
    # --------------------------------------------------

    return {
        "X": X,
        "time": time,
        "A_true": A_true,
        "true_links": true_links,
        "variables": variables,
        "equations": equations,
    }


def temporal_split(
    X,
    time,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Split time-series data chronologically.

    No shuffling is performed.

    Example for 5000 observations:

        Train:      0 - 3499
        Validation: 3500 - 4249
        Test:       4250 - 4999
    """

    if not (0 < train_ratio < 1):
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not (0 < val_ratio < 1):
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1."
        )

    if len(X) != len(time):
        raise ValueError(
            "X and time must contain the same number "
            "of observations."
        )

    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    time_train = time[:train_end]
    time_val = time[train_end:val_end]
    time_test = time[val_end:]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "time_train": time_train,
        "time_val": time_val,
        "time_test": time_test,
    }


# ==================================================
# Test the complete loader
# ==================================================

if __name__ == "__main__":

    import sys

    # --------------------------------------------------
    # Read sample size from command line
    # --------------------------------------------------

    if len(sys.argv) != 2:
        print(
            "Usage: python timegraph_loader.py <sample_size>"
        )
        print(
            "Example: python timegraph_loader.py 500"
        )
        raise SystemExit(1)

    try:
        sample_size = int(sys.argv[1])
    except ValueError:
        print("Sample size must be an integer.")
        raise SystemExit(1)

    # --------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------

    variables = [
        "X1",
        "X2",
        "X3",
        "X4",
    ]

    n_vars = 4
    max_lag = 2

    csv_path = (
        f"Datasets/A1/Gaussian/4 variable/Lag 2/"
        f"linear_ts_n{sample_size}_vars4_lag2.csv"
    )

    # --------------------------------------------------
    # Check that the requested file exists
    # --------------------------------------------------

    from pathlib import Path

    if not Path(csv_path).exists():
        print(f"Dataset file not found:")
        print(csv_path)
        raise SystemExit(1)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    dataset = load_timegraph(
        csv_path=csv_path,
        variables=variables,
        n_vars=n_vars,
        max_lag=max_lag,
    )

    # --------------------------------------------------
    # Display dataset information
    # --------------------------------------------------

    print("===== TimeGraph Loader =====")

    print("\nSample size:")
    print(sample_size)

    print("\nX shape:")
    print(dataset["X"].shape)

    print("\ntime shape:")
    print(dataset["time"].shape)

    print("\nA_true shape:")
    print(dataset["A_true"].shape)

    print("\nVariables:")
    print(dataset["variables"])

    print("\nTrue causal links:")

    for link, weight in dataset["true_links"].items():
        print(link, "=>", weight)

    # --------------------------------------------------
    # Temporal train/validation/test split
    # --------------------------------------------------

    splits = temporal_split(
        dataset["X"],
        dataset["time"],
    )

    print("\n===== Temporal Split =====")

    print(
        "Train:",
        splits["X_train"].shape,
        f"time {splits['time_train'][0]}"
        f" -> "
        f"{splits['time_train'][-1]}",
    )

    print(
        "Validation:",
        splits["X_val"].shape,
        f"time {splits['time_val'][0]}"
        f" -> "
        f"{splits['time_val'][-1]}",
    )

    print(
        "Test:",
        splits["X_test"].shape,
        f"time {splits['time_test'][0]}"
        f" -> "
        f"{splits['time_test'][-1]}",
    )

