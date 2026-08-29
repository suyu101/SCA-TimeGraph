import os
import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ============================================================
# Configuration
# ============================================================

DATA_PATH = (
    "Datasets/A1/Gaussian/4 variable/Lag 2/"
    "linear_ts_n5000_vars4_lag2.csv"
)

OUTPUT_PATH = "results/pcmci_a1_A_pred.npy"

VARIABLES = ["X1", "X2", "X3", "X4"]

N_VARS = len(VARIABLES)

MAX_LAG = 2

ALPHA = 0.05


# ============================================================
# Load data
# ============================================================

def load_timegraph(path):
    """Load TimeGraph CSV data."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = VARIABLES + ["time"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    X = df[VARIABLES].to_numpy(
        dtype=np.float64
    )

    time = df["time"].to_numpy()

    return X, time


# ============================================================
# Run PCMCI+
# ============================================================

def run_pcmci_plus(
    X,
    tau_min=0,
    tau_max=2,
    alpha_level=0.05,
):
    """
    Run the PCMCI+ algorithm using ParCorr.
    """

    dataframe = pp.DataFrame(
        X,
        var_names=VARIABLES,
    )

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(
            significance="analytic"
        ),
        verbosity=1,
    )

    results = pcmci.run_pcmciplus(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=alpha_level,
    )

    return results


# ============================================================
# Convert PCMCI+ graph into our tensor format
# ============================================================

def pcmci_graph_to_tensor(
    results,
    n_vars,
    max_lag,
):
    """
    Convert Tigramite's PCMCI+ graph representation into:

        A_pred[source, target, lag]

    Our convention:

        lag=0
            source(t) -> target(t)

        lag=1
            source(t-1) -> target(t)

        lag=2
            source(t-2) -> target(t)

    Important:
    Tigramite represents lagged graph entries using
    the target variable first and the source variable
    second. The orientation characters must therefore
    be interpreted relative to that convention.
    """

    graph = results["graph"]

    A_pred = np.zeros(
        (
            n_vars,
            n_vars,
            max_lag + 1,
        ),
        dtype=np.float32,
    )

    print(
        "\n===== Directed PCMCI+ Links ====="
    )

    # --------------------------------------------------------
    # LAGGED LINKS
    # --------------------------------------------------------
    #
    # For a relationship:
    #
    #     X_source(t-lag) -> X_target(t)
    #
    # Tigramite stores the relevant graph entry as:
    #
    #     graph[target, source, lag]
    #
    # For lagged links:
    #
    #     "<--"
    #
    # corresponds to:
    #
    #     source -> target
    #
    # while:
    #
    #     "-->"
    #
    # corresponds to:
    #
    #     target -> source
    # --------------------------------------------------------

    for target in range(n_vars):

        for source in range(n_vars):

            if source == target:
                continue

            for lag in range(
                1,
                max_lag + 1,
            ):

                value = graph[
                    target,
                    source,
                    lag,
                ]

                if value == "<--":

                    A_pred[
                        source,
                        target,
                        lag,
                    ] = 1.0

                    print(
                        f"{VARIABLES[source]} -> "
                        f"{VARIABLES[target]} "
                        f"(lag={lag})"
                    )

                elif value == "-->":

                    A_pred[
                        target,
                        source,
                        lag,
                    ] = 1.0

                    print(
                        f"{VARIABLES[target]} -> "
                        f"{VARIABLES[source]} "
                        f"(lag={lag})"
                    )

    # --------------------------------------------------------
    # CONTEMPORANEOUS LINKS
    # --------------------------------------------------------
    #
    # At lag=0, PCMCI+ returns orientation information
    # for contemporaneous links.
    #
    # We only record explicitly directed links.
    # Ambiguous/undirected orientations are ignored.
    # --------------------------------------------------------

    for source in range(n_vars):

        for target in range(n_vars):

            if source == target:
                continue

            value = graph[
                source,
                target,
                0,
            ]

            if value == "-->":

                A_pred[
                    source,
                    target,
                    0,
                ] = 1.0

                print(
                    f"{VARIABLES[source]} -> "
                    f"{VARIABLES[target]} "
                    f"(lag=0)"
                )

    return A_pred


# ============================================================
# Print prediction tensor
# ============================================================

def print_predicted_graph(
    A_pred,
    variables,
):
    """Print all detected directed edges."""

    print(
        "\n===== Predicted Graph ====="
    )

    found = False

    n_vars = len(variables)

    max_lag = A_pred.shape[2] - 1

    for source in range(n_vars):

        for target in range(n_vars):

            if source == target:
                continue

            for lag in range(
                max_lag + 1
            ):

                if A_pred[
                    source,
                    target,
                    lag,
                ] != 0:

                    found = True

                    print(
                        f"{variables[source]} -> "
                        f"{variables[target]} "
                        f"(lag={lag})"
                    )

    if not found:
        print(
            "No directed edges detected."
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "===== PCMCI+ Baseline ====="
    )

    print(
        "Dataset:",
        DATA_PATH,
    )

    print(
        "Variables:",
        VARIABLES,
    )

    print(
        "Maximum lag:",
        MAX_LAG,
    )

    print(
        "Alpha:",
        ALPHA,
    )

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, time = load_timegraph(
        DATA_PATH
    )

    print(
        "\nData shape:",
        X.shape,
    )

    print(
        "Time range:",
        time[0],
        "->",
        time[-1],
    )

    # --------------------------------------------------------
    # Run PCMCI+
    # --------------------------------------------------------

    results = run_pcmci_plus(
        X=X,
        tau_min=0,
        tau_max=MAX_LAG,
        alpha_level=ALPHA,
    )

    print(
        "\nPCMCI+ finished."
    )

    # --------------------------------------------------------
    # Show result keys
    # --------------------------------------------------------

    print(
        "\nResult keys:"
    )

    print(
        list(results.keys())
    )

    # --------------------------------------------------------
    # Convert graph
    # --------------------------------------------------------

    A_pred = pcmci_graph_to_tensor(
        results=results,
        n_vars=N_VARS,
        max_lag=MAX_LAG,
    )

    # --------------------------------------------------------
    # Print prediction
    # --------------------------------------------------------

    print_predicted_graph(
        A_pred,
        VARIABLES,
    )

    # --------------------------------------------------------
    # Print shape
    # --------------------------------------------------------

    print(
        "\nA_pred shape:"
    )

    print(
        A_pred.shape
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        OUTPUT_PATH,
        A_pred,
    )

    print(
        "\nSaved prediction to:"
    )

    print(
        OUTPUT_PATH
    )

