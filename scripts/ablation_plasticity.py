import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sca.evaluation.evaluate_model import evaluate_model
from sca.ground_truth.regime_ground_truth_tensor import build_time_indexed_ground_truth



# ============================================================
# ABLATION STUDY
# Selective Causal Plasticity
# ============================================================

A_true, _, _ = build_time_indexed_ground_truth()


# ------------------------------------------------------------
# Load the already generated SCA outputs
# ------------------------------------------------------------

A_full = np.load("results/sca_A_pred.npy")
plasticity = np.load("results/sca_plasticity_rates.npy")


# ------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------

def evaluate(name, A_pred):

    results = evaluate_model(
        A_pred,
        A_true,
        threshold=0.02
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Before F1:",
        results["static_before"]["f1"]
    )

    print(
        "After F1:",
        results["static_after"]["f1"]
    )

    print(
        "Before SHD:",
        results["static_before"]["shd"]
    )

    print(
        "After SHD:",
        results["static_after"]["shd"]
    )

    print(
        "Stable-edge preservation:",
        results["stable_edge_preservation"]
    )

    print(
        "Change detection accuracy:",
        results["change_detection_accuracy"]
    )

    print(
        "Adaptation delay:",
        results["adaptation_delay"]
    )

    print(
        "Unnecessary change rate:",
        results["unnecessary_change_rate"]
    )

    return results


# ============================================================
# FULL SCA
# ============================================================

full_results = evaluate(
    "FULL SCA — Selective Causal Plasticity",
    A_full
)


# ============================================================
# PLASTICITY STATISTICS
# ============================================================

print()
print("=" * 60)
print("PLASTICITY STATISTICS")
print("=" * 60)

print(
    "Minimum:",
    plasticity.min()
)

print(
    "Maximum:",
    plasticity.max()
)

print(
    "Mean:",
    plasticity.mean()
)

print(
    "Fraction above minimum:",
    np.mean(plasticity > 0.020001)
)