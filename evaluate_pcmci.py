import numpy as np

from evaluate_model import evaluate_static_graph


# ============================================================
# Paths
# ============================================================

PREDICTION_PATH = (
    "results/pcmci_a1_A_pred.npy"
)


# ============================================================
# Build TimeGraph A1 ground truth
# ============================================================

VARIABLES = ["X1", "X2", "X3", "X4"]

MAX_LAG = 2

A_true = np.zeros(
    (
        len(VARIABLES),
        len(VARIABLES),
        MAX_LAG + 1,
    ),
    dtype=np.float32,
)


# X1(t-2) -> X4(t)
A_true[0, 3, 2] = 0.25

# X4(t) -> X3(t)
A_true[3, 2, 0] = 0.35

# X3(t-1) -> X2(t)
A_true[2, 1, 1] = 0.30

# X2(t) -> X1(t)
A_true[1, 0, 0] = 0.40


# ============================================================
# Load PCMCI+ prediction
# ============================================================

A_pred = np.load(
    PREDICTION_PATH
)


print("===== PCMCI+ Evaluation =====")

print(
    "A_pred shape:",
    A_pred.shape,
)

print(
    "A_true shape:",
    A_true.shape,
)


# ============================================================
# Evaluate
# ============================================================

results = evaluate_static_graph(
    A_pred,
    A_true,
    threshold=0.0,
)


# ============================================================
# Print
# ============================================================

print("\n===== Results =====")

print(
    "Precision:",
    f"{results['precision']:.4f}",
)

print(
    "Recall:",
    f"{results['recall']:.4f}",
)

print(
    "F1:",
    f"{results['f1']:.4f}",
)

print(
    "SHD:",
    results["shd"],
)
