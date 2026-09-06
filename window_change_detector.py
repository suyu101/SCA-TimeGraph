import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

A_PRED_PATH = "results/sca_A_pred.npy"

CHANGE_POINT = 2500

# Number of graphs used on each side of the comparison
WINDOW_SIZE = 50

# Threshold will initially be calibrated from stable data
CALIBRATION_END = 2000

# Percentile used to determine threshold
THRESHOLD_PERCENTILE = 95


# ============================================================
# WINDOW-AVERAGED CHANGE SCORE
# ============================================================

def compute_change_score(
    A_pred,
    t,
    window_size,
):
    """
    Compare the average causal graph immediately before t
    with the average causal graph immediately after t.

    A_pred shape:
        (time, source, target, lag)
    """

    if t - window_size < 0:
        return 0.0

    if t + window_size > len(A_pred):
        return 0.0

    previous_graph = np.mean(
        A_pred[
            t - window_size:t
        ],
        axis=0,
    )

    current_graph = np.mean(
        A_pred[
            t:t + window_size
        ],
        axis=0,
    )

    difference = np.abs(
        current_graph - previous_graph
    )

    # Ignore self edges
    n_vars = difference.shape[0]

    for i in range(n_vars):
        difference[i, i, :] = 0.0

    flattened = difference.flatten()

    # Use strongest causal changes
    top_k = min(4, len(flattened))

    strongest = np.sort(flattened)[-top_k:]

    return float(np.mean(strongest))


# ============================================================
# COMPUTE ALL CHANGE SCORES
# ============================================================

def compute_all_scores(
    A_pred,
    window_size,
):
    n = len(A_pred)

    scores = np.zeros(n)

    for t in range(
        window_size,
        n - window_size,
    ):

        scores[t] = compute_change_score(
            A_pred,
            t,
            window_size,
        )

    return scores


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def calibrate_threshold(
    scores,
    calibration_end,
    percentile,
):
    """
    Determine threshold using only the stable
    pre-change calibration region.
    """

    stable_scores = scores[
        100:calibration_end
    ]

    threshold = np.percentile(
        stable_scores,
        percentile,
    )

    return float(threshold)


# ============================================================
# DETECTION
# ============================================================

def detect_changes(
    scores,
    threshold,
):
    return scores >= threshold


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    print("===== WINDOW-AVERAGED CHANGE DETECTOR =====")

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    A_pred = np.load(
        A_PRED_PATH
    )

    print(
        "A_pred shape:",
        A_pred.shape,
    )

    print(
        "Window size:",
        WINDOW_SIZE,
    )

    # --------------------------------------------------------
    # Compute scores
    # --------------------------------------------------------

    scores = compute_all_scores(
        A_pred,
        WINDOW_SIZE,
    )

    # --------------------------------------------------------
    # Calibrate threshold
    # --------------------------------------------------------

    threshold = calibrate_threshold(
        scores,
        CALIBRATION_END,
        THRESHOLD_PERCENTILE,
    )

    print(
        "Calibrated threshold:",
        threshold,
    )

    # --------------------------------------------------------
    # Detect
    # --------------------------------------------------------

    detected = detect_changes(
        scores,
        threshold,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    before = detected[:CHANGE_POINT]

    after = detected[CHANGE_POINT:]

    false_alarm_rate = np.mean(
        before
    )

    detection_rate = np.mean(
        after
    )

    # First detection
    detection_indices = np.where(
        detected
    )[0]

    if len(detection_indices) > 0:
        first_detection = detection_indices[0]
    else:
        first_detection = None

    # First detection AFTER the true change
    after_indices = np.where(
        detected[CHANGE_POINT:]
    )[0]

    if len(after_indices) > 0:
        first_after_detection = (
            CHANGE_POINT
            + after_indices[0]
        )

        adaptation_delay = (
            first_after_detection
            - CHANGE_POINT
        )

    else:
        first_after_detection = None
        adaptation_delay = None

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("===== RESULTS =====")

    print(
        "False-alarm rate before change:",
        false_alarm_rate,
    )

    print(
        "Detection rate after change:",
        detection_rate,
    )

    print(
        "First detection:",
        first_detection,
    )

    print(
        "First detection after true change:",
        first_after_detection,
    )

    print(
        "Adaptation delay:",
        adaptation_delay,
    )

    # --------------------------------------------------------
    # Inspect around change point
    # --------------------------------------------------------

    print()
    print("===== AROUND CHANGE POINT =====")

    for t in [
        2400,
        2450,
        2490,
        2499,
        2500,
        2501,
        2510,
        2550,
        2600,
    ]:

        print(
            f"t={t}: "
            f"score={scores[t]:.6f}, "
            f"detected={detected[t]}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    np.save(
        "results/window_change_scores.npy",
        scores,
    )

    np.save(
        "results/window_detected_flags.npy",
        detected,
    )

    print()
    print(
        "Saved:"
    )

    print(
        "results/window_change_scores.npy"
    )

    print(
        "results/window_detected_flags.npy"
    )