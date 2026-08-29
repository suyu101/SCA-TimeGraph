import numpy as np
import matplotlib.pyplot as plt


PREDICTION_PATH = (
    "Datasets/regime_change_A_pred_baseline.npy"
)

CHANGE_POINT = 2500


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

A_pred = np.load(
    PREDICTION_PATH
)


# --------------------------------------------------
# Extract the two changing edges
#
# X4 -> X3, lag 0
# X2 -> X3, lag 1
# --------------------------------------------------

old_edge_strength = A_pred[
    :,
    3,
    2,
    0,
]

new_edge_strength = A_pred[
    :,
    1,
    2,
    1,
]


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    old_edge_strength,
    label="Old edge: X4 -> X3",
)

plt.plot(
    new_edge_strength,
    label="New edge: X2(t-1) -> X3",
)

plt.axvline(
    CHANGE_POINT,
    linestyle="--",
    label="True change point",
)

plt.xlabel("Time")
plt.ylabel("Estimated edge strength")

plt.title(
    "Rolling Regression Adaptation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results_baseline_adaptation.png",
    dpi=200,
)

