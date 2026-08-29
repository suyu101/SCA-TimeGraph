import pandas as pd
import numpy as np

from regime_ground_truth_tensor import (
    build_time_indexed_ground_truth,
    VARIABLES,
    CHANGE_POINT,
    MAX_LAG,
)


def load_regime_change_dataset(csv_path):
    """
    Load the synthetic regime-change dataset together
    with its time-indexed ground truth.
    """

    # --------------------------------------------
    # Load observations
    # --------------------------------------------

    df = pd.read_csv(csv_path)

    X = df[VARIABLES].to_numpy(
        dtype=np.float32
    )

    time = df["time"].to_numpy()

    # --------------------------------------------
    # Load time-indexed ground truth
    # --------------------------------------------

    A_true, A_before, A_after = (
        build_time_indexed_ground_truth()
    )

    # --------------------------------------------
    # Sanity checks
    # --------------------------------------------

    if len(X) != len(A_true):
        raise ValueError(
            "Number of observations does not match "
            "number of ground-truth time points."
        )

    if len(time) != len(A_true):
        raise ValueError(
            "Number of timestamps does not match "
            "ground truth."
        )

    # --------------------------------------------
    # Return
    # --------------------------------------------

    return {
        "X": X,
        "time": time,
        "A_true": A_true,
        "A_before": A_before,
        "A_after": A_after,
        "change_point": CHANGE_POINT,
        "variables": VARIABLES,
        "max_lag": MAX_LAG,
    }


if __name__ == "__main__":

    path = "Datasets/regime_change_a1.csv"

    data = load_regime_change_dataset(path)

    print("===== Regime-Change Dataset =====")

    print("\nX shape:")
    print(data["X"].shape)

    print("\ntime shape:")
    print(data["time"].shape)

    print("\nA_true shape:")
    print(data["A_true"].shape)

    print("\nChange point:")
    print(data["change_point"])

    print("\nVariables:")
    print(data["variables"])

    print("\nMaximum lag:")
    print(data["max_lag"])

    print("\nGraph at t=2499:")
    print(data["A_true"][2499])

    print("\nGraph at t=2500:")
    print(data["A_true"][2500])
