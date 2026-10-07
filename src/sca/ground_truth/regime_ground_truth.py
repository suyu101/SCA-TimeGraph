import numpy as np


VARIABLES = [
    "X1",
    "X2",
    "X3",
    "X4",
]


def build_graph(true_links, variables, max_lag):
    """
    Convert causal links into a graph tensor.

    Shape:
        (source, target, lag)
    """

    n_vars = len(variables)

    A = np.zeros(
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

        A[
            source_idx,
            target_idx,
            lag_idx,
        ] = weight

    return A


# --------------------------------------------------
# Regime 1
# --------------------------------------------------

REGIME_1_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X4",  0, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2",  0, "X1"): 0.40,
}


# --------------------------------------------------
# Regime 2
# --------------------------------------------------

REGIME_2_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X2", -1, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2",  0, "X1"): 0.40,
}


if __name__ == "__main__":

    A_before = build_graph(
        REGIME_1_LINKS,
        VARIABLES,
        max_lag=2,
    )

    A_after = build_graph(
        REGIME_2_LINKS,
        VARIABLES,
        max_lag=2,
    )

    print("===== Regime Ground Truth =====")

    print("\nRegime 1:")

    for link, weight in REGIME_1_LINKS.items():
        print(link, "=>", weight)

    print("\nRegime 2:")

    for link, weight in REGIME_2_LINKS.items():
        print(link, "=>", weight)

    print("\nGraph shapes:")
    print("A_before:", A_before.shape)
    print("A_after :", A_after.shape)

    # --------------------------------------------------
    # Find changed entries
    # --------------------------------------------------

    changed = np.argwhere(
        A_before != A_after
    )

    print("\nChanged graph entries:")

    for source_idx, target_idx, lag_idx in changed:

        before = A_before[
            source_idx,
            target_idx,
            lag_idx,
        ]

        after = A_after[
            source_idx,
            target_idx,
            lag_idx,
        ]

        print(
            f"{VARIABLES[source_idx]} -> "
            f"{VARIABLES[target_idx]} "
            f"(lag={lag_idx}): "
            f"{before} -> {after}"
        )
