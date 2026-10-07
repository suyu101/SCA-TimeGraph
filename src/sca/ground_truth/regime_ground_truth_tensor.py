import numpy as np


VARIABLES = [
    "X1",
    "X2",
    "X3",
    "X4",
]

CHANGE_POINT = 2500
N_POINTS = 5000
MAX_LAG = 2


# --------------------------------------------------
# Regime 1: before change
# --------------------------------------------------

REGIME_1_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X4",  0, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2",  0, "X1"): 0.40,
}


# --------------------------------------------------
# Regime 2: after change
# --------------------------------------------------

REGIME_2_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X2", -1, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2",  0, "X1"): 0.40,
}


def build_graph(true_links, variables, max_lag):
    """
    Convert causal links into:

        (source, target, lag)

    graph tensor.
    """

    n_vars = len(variables)

    graph = np.zeros(
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

        graph[
            source_idx,
            target_idx,
            lag_idx,
        ] = weight

    return graph


def build_time_indexed_ground_truth():
    """
    Create A_true[t, source, target, lag].
    """

    graph_before = build_graph(
        REGIME_1_LINKS,
        VARIABLES,
        MAX_LAG,
    )

    graph_after = build_graph(
        REGIME_2_LINKS,
        VARIABLES,
        MAX_LAG,
    )

    A_true = np.zeros(
        (
            N_POINTS,
            len(VARIABLES),
            len(VARIABLES),
            MAX_LAG + 1,
        ),
        dtype=np.float32,
    )

    # Before change point
    A_true[:CHANGE_POINT] = graph_before

    # After change point
    A_true[CHANGE_POINT:] = graph_after

    return A_true, graph_before, graph_after


if __name__ == "__main__":

    A_true, A_before, A_after = (
        build_time_indexed_ground_truth()
    )

    print("===== Time-Indexed Ground Truth =====")

    print("\nA_true shape:")
    print(A_true.shape)

    print("\nExpected shape:")
    print(
        (
            N_POINTS,
            len(VARIABLES),
            len(VARIABLES),
            MAX_LAG + 1,
        )
    )

    # --------------------------------------------
    # Check before change
    # --------------------------------------------

    print("\nGraph at t=2499:")

    print(A_true[2499])

    # --------------------------------------------
    # Check after change
    # --------------------------------------------

    print("\nGraph at t=2500:")

    print(A_true[2500])

    # --------------------------------------------
    # Check stable graphs
    # --------------------------------------------

    print("\nStable edge checks:")

    # X1 -> X4, lag 2
    print(
        "X1 -> X4, lag 2:",
        A_true[2499, 0, 3, 2],
        "before /",
        A_true[2500, 0, 3, 2],
        "after",
    )

    # X3 -> X2, lag 1
    print(
        "X3 -> X2, lag 1:",
        A_true[2499, 2, 1, 1],
        "before /",
        A_true[2500, 2, 1, 1],
        "after",
    )

    # X2 -> X1, lag 0
    print(
        "X2 -> X1, lag 0:",
        A_true[2499, 1, 0, 0],
        "before /",
        A_true[2500, 1, 0, 0],
        "after",
    )

    # --------------------------------------------
    # Check changed edges
    # --------------------------------------------

    print("\nChanged edges:")

    print(
        "X4 -> X3, lag 0:",
        A_true[2499, 3, 2, 0],
        "before /",
        A_true[2500, 3, 2, 0],
        "after",
    )

    print(
        "X2 -> X3, lag 1:",
        A_true[2499, 1, 2, 1],
        "before /",
        A_true[2500, 1, 2, 1],
        "after",
    )
