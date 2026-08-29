import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from evaluate_model import evaluate_model
from regime_ground_truth_tensor import (
    build_time_indexed_ground_truth,
)


# ==================================================
# Configuration
# ==================================================

VARIABLES = ["X1", "X2", "X3", "X4"]

N_VARS = len(VARIABLES)

MAX_LAG = 2

WINDOW_SIZE = 500

# This is only a starting threshold.
# We will tune it on validation data later.
THRESHOLD = 0.10

DATA_PATH = "Datasets/regime_change_a1.csv"


# ==================================================
# Load data
# ==================================================

def load_data(path):
    df = pd.read_csv(path)

    X = df[VARIABLES].to_numpy(dtype=np.float64)
    time = df["time"].to_numpy()

    return X, time


# ==================================================
# Create lagged features
# ==================================================

def create_lagged_features(
    X,
    start,
    end,
    max_lag,
):
    """
    Create features:

        X(t-2), X(t-1)

    for every valid target time t.

    Returns
    -------
    features:
        shape (samples, n_vars * max_lag)

    targets:
        shape (samples, n_vars)

    times:
        target times
    """

    first_t = max(
        start,
        max_lag,
    )

    features = []
    targets = []
    times = []

    for t in range(first_t, end):

        row = []

        # Oldest lag first.
        for lag in range(
            max_lag,
            0,
            -1,
        ):
            row.extend(X[t - lag])

        features.append(row)
        targets.append(X[t])
        times.append(t)

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(times),
    )


# ==================================================
# Infer contemporaneous ordering
# ==================================================

def infer_order(
    residual_correlations,
):
    """
    Infer a simple ordering from a residual-correlation
    score.

    This is a heuristic baseline, not a guaranteed
    causal-ordering algorithm.

    Lower total residual correlation is placed later.

    Returns
    -------
    list[int]
        Variable indices in estimated causal order.
    """

    scores = residual_correlations.sum(axis=0)

    return list(
        np.argsort(scores)
    )


# ==================================================
# Estimate one graph
# ==================================================

def estimate_graph(
    X,
    estimate_time,
    window_size,
    max_lag,
    threshold,
):
    """
    Estimate a full graph containing:

        lagged edges
        contemporaneous edges

    Output shape:

        (source, target, lag)
    """

    A_pred = np.zeros(
        (
            N_VARS,
            N_VARS,
            max_lag + 1,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Select historical estimation window
    # --------------------------------------------------

    window_start = max(
        max_lag,
        estimate_time - window_size + 1,
    )

    window_end = estimate_time + 1

    (
        features,
        targets,
        _,
    ) = create_lagged_features(
        X,
        window_start,
        window_end,
        max_lag,
    )

    if len(features) < 10:
        return A_pred

    # --------------------------------------------------
    # Stage 1:
    # Estimate lagged relationships
    # --------------------------------------------------

    lag_coefficients = np.zeros(
        (
            N_VARS,
            N_VARS,
            max_lag,
        ),
        dtype=np.float64,
    )

    lag_only_predictions = np.zeros_like(
        targets
    )

    lag_models = []

    for target_idx in range(N_VARS):

        y = targets[:, target_idx]

        model = LinearRegression()

        model.fit(
            features,
            y,
        )

        lag_models.append(model)

        prediction = model.predict(
            features
        )

        lag_only_predictions[:, target_idx] = (
            prediction
        )

        coefficients = model.coef_

        for lag_position, lag in enumerate(
            range(max_lag, 0, -1)
        ):

            start_idx = (
                lag_position * N_VARS
            )

            end_idx = (
                start_idx + N_VARS
            )

            lag_coefficients[
                :,
                target_idx,
                lag - 1,
            ] = coefficients[
                start_idx:end_idx
            ]

            for source_idx in range(N_VARS):

                coefficient = coefficients[
                    start_idx + source_idx
                ]

                if abs(coefficient) >= threshold:

                    A_pred[
                        source_idx,
                        target_idx,
                        lag,
                    ] = coefficient

    # --------------------------------------------------
    # Stage 2:
    # Estimate contemporaneous relationships
    # --------------------------------------------------

    residuals = (
        targets
        - lag_only_predictions
    )

    # Residual correlation matrix
    residual_corr = np.corrcoef(
        residuals,
        rowvar=False,
    )

    residual_corr = np.nan_to_num(
        residual_corr,
        nan=0.0,
    )

    # Ignore self-correlation
    np.fill_diagonal(
        residual_corr,
        0.0,
    )

    # Infer an ordering.
    #
    # This is intentionally a heuristic.
    order = infer_order(
        np.abs(residual_corr)
    )

    # --------------------------------------------------
    # Regress each variable on:
    #
    #   previous-time information
    #   +
    #   contemporaneous variables earlier in order
    # --------------------------------------------------

    position = {
        variable_idx: position_idx
        for position_idx, variable_idx
        in enumerate(order)
    }

    for target_idx in order:

        earlier_variables = [
            idx
            for idx in order
            if position[idx]
            < position[target_idx]
        ]

        if not earlier_variables:
            continue

        # Historical features
        X_lag = features

        # Current-time contemporaneous features
        X_current = targets[
            :,
            earlier_variables,
        ]

        X_combined = np.column_stack(
            [
                X_lag,
                X_current,
            ]
        )

        y = targets[
            :,
            target_idx,
        ]

        model = LinearRegression()

        model.fit(
            X_combined,
            y,
        )

        coefficients = model.coef_

        # Current-time coefficients are
        # located after all lagged features.
        current_start = (
            N_VARS * max_lag
        )

        for j, source_idx in enumerate(
            earlier_variables
        ):

            coefficient = coefficients[
                current_start + j
            ]

            if abs(coefficient) >= threshold:

                A_pred[
                    source_idx,
                    target_idx,
                    0,
                ] = coefficient

    return A_pred


# ==================================================
# Print graph
# ==================================================

def print_graph(
    A,
    title,
):
    print(f"\n===== {title} =====")

    found = False

    for source in range(N_VARS):

        for target in range(N_VARS):

            for lag in range(
                MAX_LAG + 1
            ):

                weight = A[
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
# Main
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    X, time = load_data(
        DATA_PATH
    )

    print(
        "===== Full Regression Baseline ====="
    )

    print(
        "Dataset shape:",
        X.shape,
    )

    print(
        "Window size:",
        WINDOW_SIZE,
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    # --------------------------------------------------
    # Estimate graph BEFORE change
    # --------------------------------------------------

    A_before = estimate_graph(
        X=X,
        estimate_time=2499,
        window_size=WINDOW_SIZE,
        max_lag=MAX_LAG,
        threshold=THRESHOLD,
    )

    # --------------------------------------------------
    # Estimate graph AFTER change
    # --------------------------------------------------

    A_after = estimate_graph(
        X=X,
        estimate_time=4999,
        window_size=WINDOW_SIZE,
        max_lag=MAX_LAG,
        threshold=THRESHOLD,
    )

    # --------------------------------------------------
    # Print predictions
    # --------------------------------------------------

    print_graph(
        A_before,
        "Estimated BEFORE Change",
    )

    print_graph(
        A_after,
        "Estimated AFTER Change",
    )

    # --------------------------------------------------
    # Build time-indexed predictions
    # --------------------------------------------------

    A_true, _, _ = (
        build_time_indexed_ground_truth()
    )

    A_pred = np.zeros_like(
        A_true
    )

    # For this first baseline test:
    #
    # use the estimated pre-change graph
    # for all pre-change times
    #
    # and the estimated post-change graph
    # for all post-change times.

    A_pred[:2500] = A_before
    A_pred[2500:] = A_after

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    results = evaluate_model(
        A_pred,
        A_true,
        change_point=2500,
        threshold=THRESHOLD,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        "\n===== Static Performance BEFORE ====="
    )

    for key, value in results[
        "static_before"
    ].items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n===== Static Performance AFTER ====="
    )

    for key, value in results[
        "static_after"
    ].items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n===== Adaptation Performance ====="
    )

    print(
        "Stable-edge preservation:",
        results[
            "stable_edge_preservation"
        ],
    )

    print(
        "Change-detection accuracy:",
        results[
            "change_detection_accuracy"
        ],
    )

    print(
        "Adaptation delay:",
        results[
            "adaptation_delay"
        ],
    )

    print(
        "Unnecessary change rate:",
        results[
            "unnecessary_change_rate"
        ],
    )
