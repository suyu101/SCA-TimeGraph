import numpy as np
import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "Datasets/regime_change_a1.csv"

df = pd.read_csv(DATA_PATH)

VARIABLES = [
    "X1",
    "X2",
    "X3",
    "X4",
]

X = df[
    VARIABLES
].to_numpy(
    dtype=np.float64
)


# ============================================================
# R2 FUNCTION
# ============================================================

def calculate_r2(
    predictors,
    target,
):
    """
    Calculate R2 for a linear regression.
    """

    if predictors.shape[1] == 0:
        return 0.0

    design = np.column_stack(
        [
            np.ones(
                len(target)
            ),
            predictors,
        ]
    )

    beta = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )[0]

    prediction = (
        design @ beta
    )

    residual = (
        target - prediction
    )

    ss_res = np.sum(
        residual ** 2
    )

    ss_tot = np.sum(
        (
            target
            -
            np.mean(target)
        ) ** 2
    )

    if ss_tot < 1e-12:
        return 0.0

    return max(
        0.0,
        1.0 - ss_res / ss_tot,
    )


# ============================================================
# STANDARDIZE
# ============================================================

def standardize(
    values
):
    mean = values.mean(
        axis=0
    )

    std = values.std(
        axis=0
    )

    std[
        std < 1e-10
    ] = 1.0

    return (
        values - mean
    ) / std


# ============================================================
# TEST LAGGED EDGE
# ============================================================

def test_lagged_edge(
    X_window,
    source,
    target,
    lag,
):
    """
    Test:

        source(t-lag) -> target(t)

    using only past information.

    Returns:

        baseline R2
        full R2
        improvement
    """

    max_lag = 2

    if len(X_window) <= max_lag + 10:
        return 0.0, 0.0, 0.0

    # --------------------------------------------------------
    # Target X(t)
    # --------------------------------------------------------

    target_values = X_window[
        max_lag:,
        target,
    ]

    # --------------------------------------------------------
    # Build all past predictors.
    #
    # lag 1:
    # X(t-1)
    #
    # lag 2:
    # X(t-2)
    # --------------------------------------------------------

    predictors = []

    for lag_value in [
        1,
        2,
    ]:

        predictors.append(
            X_window[
                max_lag - lag_value:
                len(X_window) - lag_value,
                :
            ]
        )

    predictors = np.concatenate(
        predictors,
        axis=1,
    )

    # --------------------------------------------------------
    # Candidate column
    #
    # First 4 columns = lag 1
    # Next 4 columns  = lag 2
    # --------------------------------------------------------

    candidate_column = (
        (lag - 1) * 4
        +
        source
    )

    baseline = np.delete(
        predictors,
        candidate_column,
        axis=1,
    )

    full = predictors

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    baseline = standardize(
        baseline
    )

    full = standardize(
        full
    )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    baseline_r2 = calculate_r2(
        baseline,
        target_values,
    )

    full_r2 = calculate_r2(
        full,
        target_values,
    )

    improvement = (
        full_r2
        -
        baseline_r2
    )

    return (
        baseline_r2,
        full_r2,
        max(
            0.0,
            improvement,
        ),
    )


# ============================================================
# TEST CONTEMPORANEOUS EDGE
# ============================================================

def test_contemporaneous_edge(
    X_window,
    source,
    target,
):
    """
    Test:

        source(t) -> target(t)

    Baseline:
        past variables

    Full:
        past variables + source(t)
    """

    max_lag = 2

    target_values = X_window[
        max_lag:,
        target,
    ]

    # --------------------------------------------------------
    # Past predictors
    # --------------------------------------------------------

    past_predictors = []

    for lag_value in [
        1,
        2,
    ]:

        past_predictors.append(
            X_window[
                max_lag - lag_value:
                len(X_window) - lag_value,
                :
            ]
        )

    past_predictors = np.concatenate(
        past_predictors,
        axis=1,
    )

    # --------------------------------------------------------
    # Current source
    # --------------------------------------------------------

    current_source = X_window[
        max_lag:,
        source,
    ]

    current_source = (
        current_source
        -
        current_source.mean()
    )

    source_std = (
        current_source.std()
    )

    if source_std > 1e-10:

        current_source = (
            current_source
            /
            source_std
        )

    # --------------------------------------------------------
    # Baseline and full
    # --------------------------------------------------------

    baseline = standardize(
        past_predictors
    )

    full = np.column_stack(
        [
            baseline,
            current_source,
        ]
    )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    baseline_r2 = calculate_r2(
        baseline,
        target_values,
    )

    full_r2 = calculate_r2(
        full,
        target_values,
    )

    improvement = (
        full_r2
        -
        baseline_r2
    )

    return (
        baseline_r2,
        full_r2,
        max(
            0.0,
            improvement,
        ),
    )


# ============================================================
# PRINT EDGE RESULT
# ============================================================

def print_result(
    name,
    before,
    after,
):
    print(
        f"\n{name}"
    )

    print(
        "Before:"
    )

    print(
        f"  baseline R2 = {before[0]:.6f}"
    )

    print(
        f"  full R2     = {before[1]:.6f}"
    )

    print(
        f"  improvement = {before[2]:.6f}"
    )

    print(
        "After:"
    )

    print(
        f"  baseline R2 = {after[0]:.6f}"
    )

    print(
        f"  full R2     = {after[1]:.6f}"
    )

    print(
        f"  improvement = {after[2]:.6f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "===== CANDIDATE EDGE DIAGNOSTIC ====="
    )

    print(
        "\nDataset shape:",
        X.shape,
    )

    print(
        "\nExpected regime change:"
    )

    print(
        "t = 2500"
    )

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    before_window = X[
        2300:2500
    ]

    after_window = X[
        2500:2700
    ]

    print(
        "\nBefore window:",
        before_window.shape,
    )

    print(
        "After window:",
        after_window.shape,
    )

    # ========================================================
    # EDGE 1
    #
    # X4(t) -> X3(t)
    # lag 0
    # ========================================================

    before_x4_x3 = (
        test_contemporaneous_edge(
            before_window,
            source=3,
            target=2,
        )
    )

    after_x4_x3 = (
        test_contemporaneous_edge(
            after_window,
            source=3,
            target=2,
        )
    )

    print_result(
        "X4(t) -> X3(t) [lag 0]",
        before_x4_x3,
        after_x4_x3,
    )

    # ========================================================
    # EDGE 2
    #
    # X2(t-1) -> X3(t)
    # lag 1
    # ========================================================

    before_x2_x3 = (
        test_lagged_edge(
            before_window,
            source=1,
            target=2,
            lag=1,
        )
    )

    after_x2_x3 = (
        test_lagged_edge(
            after_window,
            source=1,
            target=2,
            lag=1,
        )
    )

    print_result(
        "X2(t-1) -> X3(t) [lag 1]",
        before_x2_x3,
        after_x2_x3,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n===== SUMMARY ====="
    )

    print(
        "\nX4 -> X3:"
    )

    print(
        f"Before = "
        f"{before_x4_x3[2]:.6f}"
    )

    print(
        f"After  = "
        f"{after_x4_x3[2]:.6f}"
    )

    print(
        "\nX2(t-1) -> X3(t):"
    )

    print(
        f"Before = "
        f"{before_x2_x3[2]:.6f}"
    )

    print(
        f"After  = "
        f"{after_x2_x3[2]:.6f}"
    )

    print(
        "\n===== DIAGNOSTIC COMPLETE ====="
    )