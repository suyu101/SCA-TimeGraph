import os
import numpy as np
import pandas as pd


# ============================================================
# SCA MODEL
# Selective Causal Adaptation for Evolving Time-Series Graphs
# ============================================================

DATA_PATH = "Datasets/regime_change_a1.csv"

N_VARIABLES = 4
MAX_LAG = 2

# Prediction window
WINDOW_SIZE = 100

# Two-window change detection
DETECT_WINDOW = 100

# Memory parameters
MEMORY_RATE = 0.02
PLASTICITY_RATE = 0.50

# Change detection thresholds
CHANGE_THRESHOLD = 0.02
EDGE_CHANGE_THRESHOLD = 0.02

VARIABLES = ["X1", "X2", "X3", "X4"]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def calculate_r2(y, y_pred):
    """
    Calculate R^2 without using future information.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot < 1e-12:
        return 0.0

    return max(0.0, 1.0 - ss_res / ss_tot)


def standardize(X):
    """
    Standardize using only the supplied local window.
    No global dataset statistics are used.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std < 1e-8] = 1.0

    return (X - mean) / std


# ============================================================
# PAST-ONLY PREDICTORS
# ============================================================

def build_past_predictors(X):
    """
    Construct predictors using only historical observations.

    For every target time t:

        lag 1 -> X(t-1)
        lag 2 -> X(t-2)

    Current target X(t) is NEVER included.

    This prevents target leakage.
    """

    n = len(X)

    if n <= MAX_LAG:
        return np.empty((0, N_VARIABLES * MAX_LAG))

    rows = []

    for t in range(MAX_LAG, n):
        row = []

        # lag 1
        row.extend(X[t - 1])

        # lag 2
        row.extend(X[t - 2])

        rows.append(row)

    return np.asarray(rows, dtype=float)


# ============================================================
# LAG SCORE
# ============================================================

def calculate_lag_score(X, source, target, lag):
    """
    Calculate incremental predictive contribution of:

        source(t-lag) -> target(t)

    using only information available up to target time t.

    The score is:

        R2(full model) - R2(baseline model)

    where the baseline contains other historical predictors,
    but NOT the candidate edge itself.
    """

    n = len(X)

    if n <= MAX_LAG + 5:
        return 0.0

    X_std = standardize(X)

    past = build_past_predictors(X_std)

    # Target values correspond to t = MAX_LAG ... n-1
    y = X_std[MAX_LAG:, target]

    candidate = X_std[MAX_LAG - lag:n - lag, source]

    # Candidate must have exactly the same number of rows
    if len(candidate) != len(y):
        return 0.0

    # --------------------------------------------------------
    # Baseline predictors
    # --------------------------------------------------------

    baseline_columns = []

    for historical_lag in range(1, MAX_LAG + 1):

        for variable in range(N_VARIABLES):

            # Skip the candidate edge when it is one of
            # the historical predictors.
            if historical_lag == lag and variable == source:
                continue

            column_index = (
                (historical_lag - 1) * N_VARIABLES
                + variable
            )

            baseline_columns.append(past[:, column_index])

    if len(baseline_columns) > 0:
        X_base = np.column_stack(baseline_columns)
    else:
        X_base = np.empty((len(y), 0))

    # --------------------------------------------------------
    # Add candidate
    # --------------------------------------------------------

    if X_base.shape[1] > 0:
        X_full = np.column_stack([X_base, candidate])
    else:
        X_full = candidate.reshape(-1, 1)

    # --------------------------------------------------------
    # Fit baseline
    # --------------------------------------------------------

    if X_base.shape[1] > 0:
        X_base_design = np.column_stack([
            np.ones(len(y)),
            X_base
        ])

        beta_base = np.linalg.lstsq(
            X_base_design,
            y,
            rcond=None
        )[0]

        pred_base = X_base_design @ beta_base

    else:
        pred_base = np.full(
            len(y),
            np.mean(y)
        )

    # --------------------------------------------------------
    # Fit full model
    # --------------------------------------------------------

    X_full_design = np.column_stack([
        np.ones(len(y)),
        X_full
    ])

    beta_full = np.linalg.lstsq(
        X_full_design,
        y,
        rcond=None
    )[0]

    pred_full = X_full_design @ beta_full

    r2_base = calculate_r2(y, pred_base)
    r2_full = calculate_r2(y, pred_full)

    improvement = r2_full - r2_base

    return max(0.0, float(improvement))


# ============================================================
# LOCAL GRAPH ESTIMATION
# ============================================================

def estimate_local_graph(X_window):
    """
    Estimate the local causal graph from the current
    historical window.

    Output:

        graph[source, target, lag]
    """

    graph = np.zeros(
        (N_VARIABLES, N_VARIABLES, MAX_LAG + 1),
        dtype=np.float32
    )

    for source in range(N_VARIABLES):

        for target in range(N_VARIABLES):

            # No self-causal edges
            if source == target:
                continue

            for lag in range(MAX_LAG + 1):

                score = calculate_lag_score(
                    X_window,
                    source,
                    target,
                    lag
                )

                graph[source, target, lag] = score

    return graph


# ============================================================
# DIRECT TWO-WINDOW CHANGE DETECTOR
# ============================================================

def detect_change(previous_graph, current_graph):
    """
    Detect whether the recent causal graph has changed.

    IMPORTANT:

    The detector compares:

        previous local graph
                    VS
        current local graph

    rather than comparing the current graph against the
    smoothed memory.

    This prevents memory smoothing from hiding the regime change.
    """

    difference = np.abs(
        current_graph - previous_graph
    )

    # Ignore self edges
    for i in range(N_VARIABLES):
        difference[i, i, :] = 0.0

    flattened = difference.flatten()

    # Look at the strongest graph changes rather than averaging
    # across all edges.
    top_k = min(4, len(flattened))

    strongest = np.sort(flattened)[-top_k:]

    change_score = float(
        np.mean(strongest)
    )

    detected = change_score >= CHANGE_THRESHOLD

    return change_score, detected


# ============================================================
# MEMORY UPDATE
# ============================================================

def update_memory(
    memory,
    current_graph,
    change_detected
):
    """
    Selective causal adaptation.

    Stable edges:
        slow update

    Changed edges:
        fast update when change is detected
    """

    updated = memory.copy()

    difference = np.abs(
        current_graph - memory
    )

    for source in range(N_VARIABLES):

        for target in range(N_VARIABLES):

            if source == target:
                continue

            for lag in range(MAX_LAG + 1):

                diff = difference[
                    source,
                    target,
                    lag
                ]

                # Stable edge -> slow memory update
                rate = MEMORY_RATE

                # Changed edge -> fast plastic update
                if (
                    change_detected
                    and diff >= EDGE_CHANGE_THRESHOLD
                ):
                    rate = PLASTICITY_RATE

                updated[
                    source,
                    target,
                    lag
                ] = (
                    (1.0 - rate)
                    * memory[source, target, lag]
                    +
                    rate
                    * current_graph[source, target, lag]
                )

    return updated


# ============================================================
# MAIN SCA PREDICTION PIPELINE
# ============================================================

def predict(X):
    """
    Generate A_pred for the entire time series.

    Output shape:

        (time, source, target, lag)

        (5000, 4, 4, 3)

    No future information is used.
    """

    n = len(X)

    A_pred = np.zeros(
        (
            n,
            N_VARIABLES,
            N_VARIABLES,
            MAX_LAG + 1
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Memory graph
    # --------------------------------------------------------

    memory = None

    # Previous graph used ONLY for direct change detection
    previous_detection_graph = None

    change_scores = np.zeros(n)

    detected_flags = np.zeros(
        n,
        dtype=bool
    )

    # --------------------------------------------------------
    # Chronological processing
    # --------------------------------------------------------

    for t in range(MAX_LAG, n):

        # ----------------------------------------------------
        # Current local prediction window
        # ----------------------------------------------------

        start = max(
            0,
            t - WINDOW_SIZE + 1
        )

        X_window = X[start:t + 1]

        current_graph = estimate_local_graph(
            X_window
        )

        # ----------------------------------------------------
        # Initialize memory
        # ----------------------------------------------------

        if memory is None:

            memory = current_graph.copy()

            previous_detection_graph = current_graph.copy()

            A_pred[t] = memory

            continue

        # ----------------------------------------------------
        # CHANGE DETECTION
        # ----------------------------------------------------

        if t >= DETECT_WINDOW * 2:

            previous_start = (
                t - 2 * DETECT_WINDOW + 1
            )

            previous_end = (
                t - DETECT_WINDOW + 1
            )

            current_start = (
                t - DETECT_WINDOW + 1
            )

            current_end = t + 1

            X_previous = X[
                previous_start:previous_end
            ]

            X_current = X[
                current_start:current_end
            ]

            previous_graph = estimate_local_graph(
                X_previous
            )

            detection_graph = estimate_local_graph(
                X_current
            )

            change_score, detected = detect_change(
                previous_graph,
                detection_graph
            )

            # The detection graph is based only on recent data.
            previous_detection_graph = detection_graph

        else:

            change_score = 0.0
            detected = False

        change_scores[t] = change_score
        detected_flags[t] = detected

        # ----------------------------------------------------
        # MEMORY + PLASTICITY
        # ----------------------------------------------------

        memory = update_memory(
            memory,
            current_graph,
            detected
        )

        # ----------------------------------------------------
        # Store prediction
        # ----------------------------------------------------

        A_pred[t] = memory

    return A_pred, change_scores, detected_flags


# ============================================================
# DATA LOADING
# ============================================================

def load_regime_change_data():
    """
    Load the controlled regime-change dataset.
    """

    df = pd.read_csv(DATA_PATH)

    X = df[
        VARIABLES
    ].to_numpy(
        dtype=np.float32
    )

    time = df[
        "time"
    ].to_numpy()

    return X, time


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("===== SCA MODEL =====")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("Loading regime-change dataset...")

    X, time = load_regime_change_data()

    print(
        "Input shape:",
        X.shape
    )

    print(
        "Time shape:",
        time.shape
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating SCA model...")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "Generating SCA causal predictions..."
    )

    A_pred, change_scores, detected_flags = predict(
        X
    )

    # --------------------------------------------------------
    # Interface check
    # --------------------------------------------------------

    print()
    print("A_pred shape:")
    print(A_pred.shape)

    print()
    print("Expected:")
    print((5000, 4, 4, 3))

    print()
    print("Minimum prediction:")
    print(float(A_pred.min()))

    print()
    print("Maximum prediction:")
    print(float(A_pred.max()))

    # --------------------------------------------------------
    # CHANGE DETECTOR
    # --------------------------------------------------------

    print()
    print("===== CHANGE DETECTOR =====")

    for t in [
        2490,
        2499,
        2500,
        2510,
        2525,
        2550,
        2600,
        2650,
        2699,
        2750
    ]:

        print(
            f"t={t}: "
            f"change_score={change_scores[t]:.6f}, "
            f"detected={detected_flags[t]}"
        )

    # --------------------------------------------------------
    # REGIME CHANGE CHECK
    # --------------------------------------------------------

    print()
    print("===== REGIME CHANGE CHECK =====")

    x4_x3 = A_pred[
        :,
        3,
        2,
        0
    ]

    x2_x3 = A_pred[
        :,
        1,
        2,
        1
    ]

    print(
        "X4 -> X3 before:",
        x4_x3[2300:2500:50]
    )

    print(
        "X4 -> X3 after:",
        x4_x3[2600:2800:50]
    )

    print(
        "X2 -> X3 before:",
        x2_x3[2300:2500:50]
    )

    print(
        "X2 -> X3 after:",
        x2_x3[2600:2800:50]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/sca_A_pred.npy"
    )

    np.save(
        output_path,
        A_pred
    )

    print()
    print("Saved predictions to:")
    print(output_path)

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    assert A_pred.shape == (
        5000,
        4,
        4,
        3
    )

    assert np.all(
        np.isfinite(A_pred)
    )

    print()
    print(
        "SUCCESS: SCA prediction interface is correct!"
    )