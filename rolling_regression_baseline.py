import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# ==================================================
# Configuration
# ==================================================

VARIABLES = ["X1", "X2", "X3", "X4"]

N_VARS = len(VARIABLES)

MAX_LAG = 2

WINDOW_SIZE = 500

# Selected using validation experiments.
THRESHOLD = 0.20

# Estimate a graph every 25 time steps.
STEP_SIZE = 25

DATA_PATH = "Datasets/regime_change_a1.csv"

OUTPUT_PATH = "Datasets/regime_change_A_pred_baseline.npy"


# ==================================================
# Load data
# ==================================================

def load_data(path):
    df = pd.read_csv(path)

    X = df[VARIABLES].to_numpy(
        dtype=np.float64
    )

    time = df["time"].to_numpy()

    return X, time


# ==================================================
# Create lagged regression features
# ==================================================

def create_lagged_features(
    X,
    start,
    end,
    max_lag,
):
    """
    Build features for target times:

        X(t-2), X(t-1)

    The current-time variables X(t) are not used
    here because these are lagged predictors.
    """

    first_t = max(
        start,
        max_lag,
    )

    features = []
    targets = []
    target_times = []

    for t in range(
        first_t,
        end,
    ):

        row = []

        # Oldest lag first:
        #
        # lag 2:
        # X1(t-2), X2(t-2), ...
        #
        # lag 1:
        # X1(t-1), X2(t-1), ...

        for lag in range(
            max_lag,
            0,
            -1,
        ):
            row.extend(
                X[t - lag]
            )

        features.append(row)
        targets.append(X[t])
        target_times.append(t)

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(target_times),
    )


# ==================================================
# Infer a contemporaneous ordering
# ==================================================

def infer_order(
    residual_correlations,
):
    """
    Heuristic ordering for contemporaneous
    relationships.

    This is a simple baseline heuristic, not a
    guaranteed causal-ordering algorithm.
    """

    scores = residual_correlations.sum(
        axis=0
    )

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
    Estimate a full causal graph around one
    target time.

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

    # --------------------------------------------
    # Select rolling window
    # --------------------------------------------

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

    # --------------------------------------------
    # Stage 1:
    # Estimate lagged effects
    # --------------------------------------------

    lag_only_predictions = np.zeros_like(
        targets
    )

    for target_idx in range(N_VARS):

        y = targets[:, target_idx]

        model = LinearRegression()

        model.fit(
            features,
            y,
        )

        prediction = model.predict(
            features
        )

        lag_only_predictions[
            :,
            target_idx
        ] = prediction

        coefficients = model.coef_

        for lag_position, lag in enumerate(
            range(max_lag, 0, -1)
        ):

            start_idx = (
                lag_position * N_VARS
            )

            for source_idx in range(
                N_VARS
            ):

                coefficient = coefficients[
                    start_idx + source_idx
                ]

                if abs(coefficient) >= threshold:

                    A_pred[
                        source_idx,
                        target_idx,
                        lag,
                    ] = coefficient

    # --------------------------------------------
    # Stage 2:
    # Estimate contemporaneous effects
    # --------------------------------------------

    residuals = (
        targets
        - lag_only_predictions
    )

    residual_corr = np.corrcoef(
        residuals,
        rowvar=False,
    )

    residual_corr = np.nan_to_num(
        residual_corr,
        nan=0.0,
    )

    np.fill_diagonal(
        residual_corr,
        0.0,
    )

    # Infer a heuristic ordering.
    order = infer_order(
        np.abs(residual_corr)
    )

    position = {
        variable_idx: position_idx
        for position_idx, variable_idx
        in enumerate(order)
    }

    # --------------------------------------------
    # Fit contemporaneous effects
    # --------------------------------------------

    for target_idx in order:

        earlier_variables = [
            idx
            for idx in order
            if position[idx]
            < position[target_idx]
        ]

        if not earlier_variables:
            continue

        X_lag = features

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
    print(
        f"\n===== {title} ====="
    )

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

    # --------------------------------------------
    # Load
    # --------------------------------------------

    X, time = load_data(
        DATA_PATH
    )

    n_points = len(X)

    print(
        "===== Rolling Regression Baseline ====="
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
        "Step size:",
        STEP_SIZE,
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    # --------------------------------------------
    # Allocate predictions
    #
    # Shape:
    # (time, source, target, lag)
    # --------------------------------------------

    A_pred = np.zeros(
        (
            n_points,
            N_VARS,
            N_VARS,
            MAX_LAG + 1,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------
    # Estimate graphs over time
    # --------------------------------------------

    estimated_times = []

    for t in range(
        MAX_LAG,
        n_points,
        STEP_SIZE,
    ):

        # Need enough history for rolling window.
        if t < MAX_LAG:
            continue

        graph = estimate_graph(
            X=X,
            estimate_time=t,
            window_size=WINDOW_SIZE,
            max_lag=MAX_LAG,
            threshold=THRESHOLD,
        )

        A_pred[t] = graph

        estimated_times.append(t)

    # --------------------------------------------
    # Fill timestamps between estimates
    #
    # Each time point receives the most recent
    # graph estimate.
    # --------------------------------------------

    last_graph = np.zeros(
        (
            N_VARS,
            N_VARS,
            MAX_LAG + 1,
        ),
        dtype=np.float32,
    )

    estimate_set = set(
        estimated_times
    )

    for t in range(n_points):

        if t in estimate_set:

            last_graph = A_pred[t].copy()

        else:

            A_pred[t] = last_graph

    # --------------------------------------------
    # Save predictions
    # --------------------------------------------

    np.save(
        OUTPUT_PATH,
        A_pred,
    )

    print(
        "\nSaved predictions to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nPrediction shape:"
    )

    print(
        A_pred.shape
    )

    # --------------------------------------------
    # Inspect estimates around change point
    # --------------------------------------------

    inspection_times = [
        2400,
        2450,
        2475,
        2500,
        2525,
        2550,
        2600,
        2700,
        3000,
        3500,
        4000,
        4999,
    ]

    for t in inspection_times:

        if t >= n_points:
            continue

        print_graph(
            A_pred[t],
            f"Prediction at t={t}",
        )
