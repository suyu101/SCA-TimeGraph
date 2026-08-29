import numpy as np

from evaluation import (
    precision_recall_f1,
    structural_hamming_distance,
)

from adaptation_metrics import (
    stable_edge_preservation,
    change_detection_accuracy,
    adaptation_delay,
    unnecessary_change_rate,
    STABLE_EDGES,
    OLD_EDGE,
    NEW_EDGE,
    CHANGE_POINT,
)


def evaluate_static_graph(
    A_pred,
    A_true,
    threshold=0.0,
):
    """
    Evaluate one causal graph.

    Returns:
        precision
        recall
        f1
        shd
    """

    scores = precision_recall_f1(
        A_pred,
        A_true,
        threshold=threshold,
    )

    shd = structural_hamming_distance(
        A_pred,
        A_true,
        threshold=threshold,
    )

    return {
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "shd": shd,
    }


def evaluate_adaptation(
    A_pred,
    change_point=CHANGE_POINT,
    threshold=0.0,
):
    """
    Evaluate time-varying causal adaptation.
    """

    stable = stable_edge_preservation(
        A_pred,
        change_point,
        STABLE_EDGES,
        threshold=threshold,
    )

    detection = change_detection_accuracy(
        A_pred,
        change_point,
        OLD_EDGE,
        NEW_EDGE,
        threshold=threshold,
    )

    delay = adaptation_delay(
        A_pred,
        change_point,
        OLD_EDGE,
        NEW_EDGE,
        threshold=threshold,
    )

    unnecessary = unnecessary_change_rate(
        A_pred,
        change_point,
        STABLE_EDGES,
        threshold=threshold,
    )

    return {
        "stable_edge_preservation": stable,
        "change_detection_accuracy": detection,
        "adaptation_delay": delay,
        "unnecessary_change_rate": unnecessary,
    }


def evaluate_model(
    A_pred,
    A_true,
    change_point=CHANGE_POINT,
    threshold=0.0,
):
    """
    Complete evaluation for a time-indexed prediction.

    Expected shape:

        A_pred:
        (time, source, target, lag)

        A_true:
        (time, source, target, lag)
    """

    if A_pred.shape != A_true.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"A_pred={A_pred.shape}, "
            f"A_true={A_true.shape}"
        )

    # --------------------------------------------
    # Graph before the change
    # --------------------------------------------

    static_before = evaluate_static_graph(
        A_pred[change_point - 1],
        A_true[change_point - 1],
        threshold=threshold,
    )

    # --------------------------------------------
    # Graph after the change
    # --------------------------------------------

    static_after = evaluate_static_graph(
        A_pred[-1],
        A_true[-1],
        threshold=threshold,
    )

    # --------------------------------------------
    # Dynamic adaptation
    # --------------------------------------------

    adaptation = evaluate_adaptation(
        A_pred,
        change_point=change_point,
        threshold=threshold,
    )

    return {
        "static_before": static_before,
        "static_after": static_after,
        **adaptation,
    }


if __name__ == "__main__":

    # --------------------------------------------
    # Use perfect ground truth as prediction
    # for a sanity check
    # --------------------------------------------

    from regime_ground_truth_tensor import (
        build_time_indexed_ground_truth,
    )

    A_true, _, _ = (
        build_time_indexed_ground_truth()
    )

    A_pred = A_true.copy()

    # --------------------------------------------
    # Evaluate
    # --------------------------------------------

    results = evaluate_model(
        A_pred,
        A_true,
    )

    print("===== COMPLETE MODEL EVALUATION =====")

    print("\nStatic performance BEFORE change:")

    for key, value in results["static_before"].items():
        print(f"{key}: {value}")

    print("\nStatic performance AFTER change:")

    for key, value in results["static_after"].items():
        print(f"{key}: {value}")

    print("\nAdaptation performance:")

    print(
        "Stable-edge preservation:",
        results["stable_edge_preservation"],
    )

    print(
        "Change-detection accuracy:",
        results["change_detection_accuracy"],
    )

    print(
        "Adaptation delay:",
        results["adaptation_delay"],
    )

    print(
        "Unnecessary change rate:",
        results["unnecessary_change_rate"],
    )
