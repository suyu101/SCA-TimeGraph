
import numpy as np


# ==================================================
# Ground-truth edge definitions
# ==================================================

# Variable indices:
#
# X1 = 0
# X2 = 1
# X3 = 2
# X4 = 3

STABLE_EDGES = [
    (0, 3, 2),   # X1 -> X4, lag 2
    (2, 1, 1),   # X3 -> X2, lag 1
    (1, 0, 0),   # X2 -> X1, lag 0
]

OLD_EDGE = (3, 2, 0)      # X4 -> X3, lag 0
NEW_EDGE = (1, 2, 1)      # X2 -> X3, lag 1

CHANGE_POINT = 2500


# ==================================================
# Basic helper
# ==================================================

def edge_present(
    graph,
    edge,
    threshold=0.0,
):
    """
    Check whether a specific causal edge is present.

    Parameters
    ----------
    graph : np.ndarray
        Shape:
            (source, target, lag)

    edge : tuple
        (source, target, lag)

    threshold : float
        Edge is considered present if its absolute
        weight is greater than the threshold.
    """

    source, target, lag = edge

    return (
        abs(graph[source, target, lag])
        > threshold
    )


# ==================================================
# Metric 1: Stable-edge preservation
# ==================================================

def stable_edge_preservation(
    A_pred,
    change_point,
    stable_edges,
    threshold=0.0,
):
    """
    Measure how often stable edges remain present
    after the causal change.

    Returns
    -------
    float
        Value between 0 and 1.

        1.0 = every stable edge was preserved
        0.0 = no stable edges were preserved
    """

    after = A_pred[change_point:]

    total = 0
    preserved = 0

    for edge in stable_edges:

        for graph in after:

            total += 1

            if edge_present(
                graph,
                edge,
                threshold,
            ):
                preserved += 1

    if total == 0:
        return 0.0

    return preserved / total


# ==================================================
# Metric 2: Unnecessary stable-edge change rate
# ==================================================

def unnecessary_change_rate(
    A_pred,
    change_point,
    stable_edges,
    threshold=0.0,
):
    """
    Measure how often stable edges are incorrectly
    removed after the true causal change.

    Returns
    -------
    float
        Value between 0 and 1.

        0.0 = no unnecessary changes
        1.0 = every stable edge was incorrectly changed
    """

    after = A_pred[change_point:]

    total = 0
    unnecessary = 0

    for edge in stable_edges:

        for graph in after:

            total += 1

            if not edge_present(
                graph,
                edge,
                threshold,
            ):
                unnecessary += 1

    if total == 0:
        return 0.0

    return unnecessary / total


# ==================================================
# Metric 3: Adaptation delay
# ==================================================

def adaptation_delay(
    A_pred,
    change_point,
    old_edge,
    new_edge,
    threshold=0.0,
):
    """
    Measure how many time steps are required to
    reach the correct post-change state.

    Correct post-change state:

        old edge = absent
        new edge = present

    Returns
    -------
    int

        0   = adapted immediately
        >0  = number of time steps needed
        -1  = never correctly adapted
    """

    for t in range(
        change_point,
        len(A_pred),
    ):

        graph = A_pred[t]

        old_present = edge_present(
            graph,
            old_edge,
            threshold,
        )

        new_present = edge_present(
            graph,
            new_edge,
            threshold,
        )

        if (
            not old_present
            and new_present
        ):
            return t - change_point

    return -1


# ==================================================
# Metric 4: Change-detection accuracy
# ==================================================

def change_detection_accuracy(
    A_pred,
    change_point,
    old_edge,
    new_edge,
    threshold=0.0,
):
    """
    Fraction of post-change time points at which
    the graph has the correct changed-edge state.

    Correct state:

        old edge = absent
        new edge = present

    Returns
    -------
    float
        Value between 0 and 1.
    """

    after = A_pred[change_point:]

    correct = 0

    for graph in after:

        old_present = edge_present(
            graph,
            old_edge,
            threshold,
        )

        new_present = edge_present(
            graph,
            new_edge,
            threshold,
        )

        if (
            not old_present
            and new_present
        ):
            correct += 1

    if len(after) == 0:
        return 0.0

    return correct / len(after)


# ==================================================
# Helper: create a prediction sequence
# ==================================================

def create_prediction_sequence(
    mode,
    n_points=5000,
):
    """
    Create controlled fake predictions for metric
    sanity testing.

    Modes
    -----

    perfect:
        Correctly preserve stable edges and
        correctly adapt.

    incomplete:
        Correctly add the new edge but incorrectly
        keep the old edge.

    unnecessary:
        Correctly adapt but incorrectly remove
        one stable edge.
    """

    A_pred = np.zeros(
        (n_points, 4, 4, 3),
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Before change
    #
    # Correct state:
    #
    # stable edges +
    # old edge
    # --------------------------------------------------

    for t in range(CHANGE_POINT):

        # Stable: X1 -> X4
        A_pred[t, 0, 3, 2] = 0.25

        # Stable: X3 -> X2
        A_pred[t, 2, 1, 1] = 0.30

        # Stable: X2 -> X1
        A_pred[t, 1, 0, 0] = 0.40

        # Old edge: X4 -> X3
        A_pred[t, 3, 2, 0] = 0.35

    # --------------------------------------------------
    # After change
    # --------------------------------------------------

    for t in range(
        CHANGE_POINT,
        n_points,
    ):

        # Stable edge 1
        A_pred[t, 0, 3, 2] = 0.25

        # Stable edge 2
        A_pred[t, 2, 1, 1] = 0.30

        if mode == "unnecessary":

            # Stable edge 3 is incorrectly removed
            # X2 -> X1 remains zero

            pass

        else:

            # Stable edge 3 correctly preserved
            A_pred[t, 1, 0, 0] = 0.40

        # --------------------------------------------------
        # New edge
        # --------------------------------------------------

        A_pred[t, 1, 2, 1] = 0.35

        # --------------------------------------------------
        # Old edge
        # --------------------------------------------------

        if mode == "incomplete":

            # Incorrectly keep old edge
            A_pred[t, 3, 2, 0] = 0.35

        # In perfect and unnecessary modes,
        # old edge correctly disappears.

    return A_pred


# ==================================================
# Run sanity tests
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # TEST 1: Perfect adaptation
    # --------------------------------------------------

    A_perfect = create_prediction_sequence(
        mode="perfect"
    )

    stable = stable_edge_preservation(
        A_perfect,
        CHANGE_POINT,
        STABLE_EDGES,
    )

    detection = change_detection_accuracy(
        A_perfect,
        CHANGE_POINT,
        OLD_EDGE,
        NEW_EDGE,
    )

    delay = adaptation_delay(
        A_perfect,
        CHANGE_POINT,
        OLD_EDGE,
        NEW_EDGE,
    )

    unnecessary = unnecessary_change_rate(
        A_perfect,
        CHANGE_POINT,
        STABLE_EDGES,
    )

    print("===== TEST 1: Perfect Adaptation =====")

    print(
        "Stable-edge preservation:",
        stable,
    )

    print(
        "Change-detection accuracy:",
        detection,
    )

    print(
        "Adaptation delay:",
        delay,
    )

    print(
        "Unnecessary change rate:",
        unnecessary,
    )


    # --------------------------------------------------
    # TEST 2: Incomplete adaptation
    # --------------------------------------------------

    A_incomplete = create_prediction_sequence(
        mode="incomplete"
    )

    stable = stable_edge_preservation(
        A_incomplete,
        CHANGE_POINT,
        STABLE_EDGES,
    )

    detection = change_detection_accuracy(
        A_incomplete,
        CHANGE_POINT,
        OLD_EDGE,
        NEW_EDGE,
    )

    delay = adaptation_delay(
        A_incomplete,
        CHANGE_POINT,
        OLD_EDGE,
        NEW_EDGE,
    )

    unnecessary = unnecessary_change_rate(
        A_incomplete,
        CHANGE_POINT,
        STABLE_EDGES,
    )

    print("\n===== TEST 2: Incomplete Adaptation =====")

    print(
        "Stable-edge preservation:",
        stable,
    )

    print(
        "Change-detection accuracy:",
        detection,
    )

    print(
        "Adaptation delay:",
        delay,
    )

    print(
        "Unnecessary change rate:",
        unnecessary,
    )


    # --------------------------------------------------
    # TEST 3: Unnecessary stable-edge change
    # --------------------------------------------------

    A_unnecessary = create_prediction_sequence(
        mode="unnecessary"
    )

    stable = stable_edge_preservation(
        A_unnecessary,
        CHANGE_POINT,
        STABLE_EDGES,
    )

    detection = change_detection_accuracy(
        A_unnecessary,
        CHANGE_POINT,
        OLD_EDGE,
        NEW_EDGE,
    )

    delay = adaptation_delay(
        A_unnecessary,
        CHANGE_POINT,
        OLD_EDGE,
        NEW_EDGE,
    )

    unnecessary = unnecessary_change_rate(
        A_unnecessary,
        CHANGE_POINT,
        STABLE_EDGES,
    )

    print(
        "\n===== TEST 3: Unnecessary Stable-Edge Change ====="
    )

    print(
        "Stable-edge preservation:",
        stable,
    )

    print(
        "Change-detection accuracy:",
        detection,
    )

    print(
        "Adaptation delay:",
        delay,
    )

    print(
        "Unnecessary change rate:",
        unnecessary,
    )

