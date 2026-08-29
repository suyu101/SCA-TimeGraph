import numpy as np

from regime_ground_truth_tensor import (
    build_time_indexed_ground_truth,
)

from evaluate_model import (
    evaluate_model,
)


PREDICTION_PATH = (
    "Datasets/regime_change_A_pred_baseline.npy"
)

THRESHOLD = 0.20

CHANGE_POINT = 2500


if __name__ == "__main__":

    # --------------------------------------------
    # Load true graph
    # --------------------------------------------

    A_true, _, _ = (
        build_time_indexed_ground_truth()
    )

    # --------------------------------------------
    # Load baseline prediction
    # --------------------------------------------

    A_pred = np.load(
        PREDICTION_PATH
    )

    print(
        "===== Baseline Evaluation ====="
    )

    print(
        "A_true shape:",
        A_true.shape,
    )

    print(
        "A_pred shape:",
        A_pred.shape,
    )

    # --------------------------------------------
    # Verify shape
    # --------------------------------------------

    if A_pred.shape != A_true.shape:

        raise ValueError(
            "Prediction and ground truth shapes "
            "do not match."
        )

    # --------------------------------------------
    # Evaluate
    # --------------------------------------------

    results = evaluate_model(
        A_pred=A_pred,
        A_true=A_true,
        change_point=CHANGE_POINT,
        threshold=THRESHOLD,
    )

    # --------------------------------------------
    # Static performance
    # --------------------------------------------

    print(
        "\n===== BEFORE CHANGE ====="
    )

    for key, value in results[
        "static_before"
    ].items():

        print(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print(
        "\n===== AFTER CHANGE ====="
    )

    for key, value in results[
        "static_after"
    ].items():

        print(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    # --------------------------------------------
    # Adaptation performance
    # --------------------------------------------

    print(
        "\n===== ADAPTATION ====="
    )

    print(
        "Stable-edge preservation:",
        f"{results['stable_edge_preservation']:.4f}",
    )

    print(
        "Change-detection accuracy:",
        f"{results['change_detection_accuracy']:.4f}",
    )

    print(
        "Adaptation delay:",
        results["adaptation_delay"],
    )

    print(
        "Unnecessary change rate:",
        f"{results['unnecessary_change_rate']:.4f}",
    )
