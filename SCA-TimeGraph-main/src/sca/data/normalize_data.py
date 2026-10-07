import pandas as pd
import numpy as np


def standardize_using_train(
    X_train,
    X_val,
    X_test,
):
    """
    Standardize data using statistics calculated
    from the training set only.
    """

    # --------------------------------------------
    # Calculate training statistics
    # --------------------------------------------

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    # --------------------------------------------
    # Apply training statistics to all splits
    # --------------------------------------------

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        mean,
        std,
    )


def create_raw_splits(X, time):
    """
    Create chronological train/validation/test splits.
    """

    n = len(X)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    time_train = time[:train_end]
    time_val = time[train_end:val_end]
    time_test = time[val_end:]

    return (
        X_train,
        X_val,
        X_test,
        time_train,
        time_val,
        time_test,
    )


if __name__ == "__main__":

    # --------------------------------------------
    # Load TimeGraph data
    # --------------------------------------------

    csv_path = (
        "Datasets/A1/Gaussian/4 variable/Lag 2/"
        "linear_ts_n500_vars4_lag2.csv"
    )

    variables = [
        "X1",
        "X2",
        "X3",
        "X4",
    ]

    df = pd.read_csv(csv_path)

    X = df[variables].to_numpy(dtype=np.float32)
    time = df["time"].to_numpy()

    # --------------------------------------------
    # Split chronologically
    # --------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        time_train,
        time_val,
        time_test,
    ) = create_raw_splits(X, time)

    # --------------------------------------------
    # Standardize using TRAIN only
    # --------------------------------------------

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        train_mean,
        train_std,
    ) = standardize_using_train(
        X_train,
        X_val,
        X_test,
    )

    # --------------------------------------------
    # Display results
    # --------------------------------------------

    print("===== Normalization Check =====")

    print("\nTraining mean:")
    print(train_mean)

    print("\nTraining std:")
    print(train_std)

    print("\nScaled training mean:")
    print(X_train_scaled.mean(axis=0))

    print("\nScaled training std:")
    print(X_train_scaled.std(axis=0))

    print("\nScaled validation mean:")
    print(X_val_scaled.mean(axis=0))

    print("\nScaled test mean:")
    print(X_test_scaled.mean(axis=0))

    print("\nShapes:")
    print("Train:", X_train_scaled.shape)
    print("Validation:", X_val_scaled.shape)
    print("Test:", X_test_scaled.shape)
