import numpy as np


def binary_graph(A, threshold=0.0):
    """
    Convert a weighted causal graph tensor into a binary graph.

    Parameters
    ----------
    A : np.ndarray
        Shape:
        (source, target, lag)

    threshold : float
        Edges with absolute weight > threshold
        are considered present.

    Returns
    -------
    np.ndarray
        Binary graph with the same shape as A.
    """

    return (np.abs(A) > threshold).astype(int)


def precision_recall_f1(A_pred, A_true, threshold=0.0):
    """
    Compute precision, recall, and F1 for a causal graph.

    """

    pred = binary_graph(A_pred, threshold)
    true = binary_graph(A_true, threshold)

    tp = np.logical_and(
        pred == 1,
        true == 1,
    ).sum()

    fp = np.logical_and(
        pred == 1,
        true == 0,
    ).sum()

    fn = np.logical_and(
        pred == 0,
        true == 1,
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def structural_hamming_distance(
    A_pred,
    A_true,
    threshold=0.0,
):
    """
    Compute a simple Hamming-style structural distance
    over all source-target-lag entries.

    Lower is better.
    """

    pred = binary_graph(A_pred, threshold)
    true = binary_graph(A_true, threshold)

    return int(np.sum(pred != true))


if __name__ == "__main__":

    # --------------------------------------------
    # Example ground-truth graph
    # --------------------------------------------

    A_true = np.zeros(
        (4, 4, 3),
        dtype=np.float32,
    )

    A_true[0, 3, 2] = 0.25
    A_true[3, 2, 0] = 0.35
    A_true[2, 1, 1] = 0.30
    A_true[1, 0, 0] = 0.40

    # --------------------------------------------
    # Test 1:
    # Perfect prediction
    # --------------------------------------------

    A_perfect = A_true.copy()

    result = precision_recall_f1(
        A_perfect,
        A_true,
    )

    shd = structural_hamming_distance(
        A_perfect,
        A_true,
    )

    print("===== Perfect Prediction =====")

    print("TP:", result["tp"])
    print("FP:", result["fp"])
    print("FN:", result["fn"])
    print("Precision:", result["precision"])
    print("Recall:", result["recall"])
    print("F1:", result["f1"])
    print("SHD:", shd)

    # --------------------------------------------
    # Test 2:
    # Empty prediction
    # --------------------------------------------

    A_empty = np.zeros_like(A_true)

    result = precision_recall_f1(
        A_empty,
        A_true,
    )

    shd = structural_hamming_distance(
        A_empty,
        A_true,
    )

    print("\n===== Empty Prediction =====")

    print("TP:", result["tp"])
    print("FP:", result["fp"])
    print("FN:", result["fn"])
    print("Precision:", result["precision"])
    print("Recall:", result["recall"])
    print("F1:", result["f1"])
    print("SHD:", shd)
    # --------------------------------------------
    # Test 3:
    # Partially correct prediction
    # --------------------------------------------

    A_partial = np.zeros_like(A_true)

    # Correct edges
    A_partial[0, 3, 2] = 0.25
    A_partial[3, 2, 0] = 0.35
    A_partial[2, 1, 1] = 0.30

    # Wrong edge:
    # X1 -> X2 at lag 0
    A_partial[0, 1, 0] = 0.50

    result = precision_recall_f1(
        A_partial,
        A_true,
    )

    shd = structural_hamming_distance(
        A_partial,
        A_true,
    )

    print("\n===== Partially Correct Prediction =====")

    print("TP:", result["tp"])
    print("FP:", result["fp"])
    print("FN:", result["fn"])
    print("Precision:", result["precision"])
    print("Recall:", result["recall"])
    print("F1:", result["f1"])
    print("SHD:", shd)
