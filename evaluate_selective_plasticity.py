import os
import numpy as np

from regime_ground_truth_tensor import build_time_indexed_ground_truth
from evaluate_model import evaluate_model


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = "results"

CHANGE_POINT = 2500

EVAL_THRESHOLD = 0.02

# Edges identified from the regime-change experiment.
#
# These are the edges we have already observed changing
# in the current experiment:
#
# X4 -> X3 lag 0  : decreases strongly
# X2 -> X3 lag 1  : increases strongly
#
# IMPORTANT:
# We will NOT assume all other edges are stable until
# comparing against the actual time-indexed ground truth.
#
# indices:
# X1=0, X2=1, X3=2, X4=3

CHANGED_EDGES = [
    (3, 2, 0),   # X4 -> X3 lag 0
    (1, 2, 1),   # X2 -> X3 lag 1
]


# ============================================================
# HELPERS
# ============================================================

def edge_name(source, target, lag):
    return f"X{source+1}->X{target+1}_lag{lag}"


def load_variant(name, prediction_file, plasticity_file):
    prediction_path = os.path.join(
        RESULTS_DIR,
        prediction_file
    )

    plasticity_path = os.path.join(
        RESULTS_DIR,
        plasticity_file
    )

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"Missing prediction file: {prediction_path}"
        )

    if not os.path.exists(plasticity_path):
        raise FileNotFoundError(
            f"Missing plasticity file: {plasticity_path}"
        )

    A = np.load(prediction_path)
    P = np.load(plasticity_path)

    return {
        "name": name,
        "A": A,
        "P": P
    }


def mean_edge_plasticity(P, edges, start, end):
    values = []

    for source, target, lag in edges:
        values.append(
            np.mean(
                P[start:end, source, target, lag]
            )
        )

    if not values:
        return np.nan

    return float(np.mean(values))


def mean_edge_adaptation(A, edges, start, end):
    values = []

    for source, target, lag in edges:
        values.append(
            np.mean(
                A[start:end, source, target, lag]
            )
        )

    if not values:
        return np.nan

    return float(np.mean(values))


def print_edge_plasticity(P, edges):

    for source, target, lag in edges:

        before = np.mean(
            P[
                CHANGE_POINT - 200:
                CHANGE_POINT,
                source,
                target,
                lag
            ]
        )

        after = np.mean(
            P[
                CHANGE_POINT + 100:
                CHANGE_POINT + 300,
                source,
                target,
                lag
            ]
        )

        print(
            f"{edge_name(source,target,lag):20s} "
            f"before={before:.4f} "
            f"after={after:.4f} "
            f"ratio="
            f"{after / max(before, 1e-8):.2f}"
        )


# ============================================================
# GROUND TRUTH
# ============================================================

print()
print("=" * 70)
print("SELECTIVE CAUSAL PLASTICITY EVALUATION")
print("=" * 70)

print()
print("Loading time-indexed ground truth...")

A_true, _, _ = build_time_indexed_ground_truth()

print(
    "Ground truth shape:",
    A_true.shape
)


# ============================================================
# LOAD VARIANTS
# ============================================================

# IMPORTANT:
#
# At the moment your three scripts overwrite
# sca_A_pred.npy and sca_plasticity_rates.npy.
#
# Therefore this script expects you to rename/copy
# the outputs after each run.
#
# We first look for these filenames.

VARIANTS = [
    (
        "Full SCA",
        "sca_full_A_pred.npy",
        "sca_full_plasticity_rates.npy"
    ),

    (
        "No Persistence",
        "sca_no_persistence_A_pred.npy",
        "sca_no_persistence_plasticity_rates.npy"
    ),

    (
        "Global Only",
        "sca_global_only_A_pred.npy",
        "sca_global_only_plasticity_rates.npy"
    ),
]


# ============================================================
# CHECK FILES
# ============================================================

print()
print("Checking experiment files...")

missing = False

for name, prediction_file, plasticity_file in VARIANTS:

    p1 = os.path.join(
        RESULTS_DIR,
        prediction_file
    )

    p2 = os.path.join(
        RESULTS_DIR,
        plasticity_file
    )

    print()
    print(name)

    print(
        " prediction:",
        "OK" if os.path.exists(p1) else "MISSING"
    )

    print(
        " plasticity:",
        "OK" if os.path.exists(p2) else "MISSING"
    )

    if not os.path.exists(p1) or not os.path.exists(p2):
        missing = True


if missing:

    print()
    print("=" * 70)
    print("MISSING EXPERIMENT FILES")
    print("=" * 70)

    print()
    print("We need to rename the outputs from each ablation.")
    print("I will give you the exact PowerShell commands below.")

    raise SystemExit


# ============================================================
# EVALUATION
# ============================================================

results = []

for name, prediction_file, plasticity_file in VARIANTS:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    A = np.load(
        os.path.join(
            RESULTS_DIR,
            prediction_file
        )
    )

    P = np.load(
        os.path.join(
            RESULTS_DIR,
            plasticity_file
        )
    )

    print(
        "A_pred shape:",
        A.shape
    )

    print(
        "Plasticity shape:",
        P.shape
    )

    # --------------------------------------------------------
    # Standard causal evaluation
    # --------------------------------------------------------

    evaluation = evaluate_model(
        A,
        A_true,
        threshold=EVAL_THRESHOLD
    )

    before_f1 = evaluation["static_before"]["f1"]
    after_f1 = evaluation["static_after"]["f1"]

    before_shd = evaluation["static_before"]["shd"]
    after_shd = evaluation["static_after"]["shd"]

    stable_preservation = (
        evaluation["stable_edge_preservation"]
    )

    detection_accuracy = (
        evaluation["change_detection_accuracy"]
    )

    adaptation_delay = (
        evaluation["adaptation_delay"]
    )

    unnecessary_rate = (
        evaluation["unnecessary_change_rate"]
    )

    # --------------------------------------------------------
    # Plasticity
    # --------------------------------------------------------

    changed_before = mean_edge_plasticity(
        P,
        CHANGED_EDGES,
        CHANGE_POINT - 200,
        CHANGE_POINT
    )

    changed_after = mean_edge_plasticity(
        P,
        CHANGED_EDGES,
        CHANGE_POINT + 100,
        CHANGE_POINT + 300
    )

    # All non-self edges excluding known changed edges
    stable_edges = []

    for source in range(4):
        for target in range(4):

            if source == target:
                continue

            for lag in range(3):

                edge = (
                    source,
                    target,
                    lag
                )

                if edge not in CHANGED_EDGES:
                    stable_edges.append(edge)

    stable_before = mean_edge_plasticity(
        P,
        stable_edges,
        CHANGE_POINT - 200,
        CHANGE_POINT
    )

    stable_after = mean_edge_plasticity(
        P,
        stable_edges,
        CHANGE_POINT + 100,
        CHANGE_POINT + 300
    )

    selectivity_ratio = (
        changed_after /
        max(stable_after, 1e-8)
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("CAUSAL PERFORMANCE")
    print("------------------")

    print(
        f"Before F1:              {before_f1:.4f}"
    )

    print(
        f"After F1:               {after_f1:.4f}"
    )

    print(
        f"Before SHD:             {before_shd}"
    )

    print(
        f"After SHD:              {after_shd}"
    )

    print(
        f"Stable preservation:   {stable_preservation:.4f}"
    )

    print(
        f"Detection accuracy:    {detection_accuracy:.4f}"
    )

    print(
        f"Adaptation delay:      {adaptation_delay}"
    )

    print(
        f"Unnecessary rate:      {unnecessary_rate:.4f}"
    )

    print()
    print("PLASTICITY SELECTIVITY")
    print("----------------------")

    print(
        f"Changed edges BEFORE:  {changed_before:.4f}"
    )

    print(
        f"Changed edges AFTER:   {changed_after:.4f}"
    )

    print(
        f"Stable edges BEFORE:   {stable_before:.4f}"
    )

    print(
        f"Stable edges AFTER:    {stable_after:.4f}"
    )

    print(
        f"SELECTIVITY RATIO:     {selectivity_ratio:.2f}x"
    )

    print()
    print("EDGE-LEVEL PLASTICITY")
    print("---------------------")

    print_edge_plasticity(
        P,
        CHANGED_EDGES
    )

    results.append({
        "Model": name,
        "Before F1": before_f1,
        "After F1": after_f1,
        "Before SHD": before_shd,
        "After SHD": after_shd,
        "Stable Preservation": stable_preservation,
        "Detection Accuracy": detection_accuracy,
        "Adaptation Delay": adaptation_delay,
        "Unnecessary Rate": unnecessary_rate,
        "Changed Plasticity": changed_after,
        "Stable Plasticity": stable_after,
        "Selectivity Ratio": selectivity_ratio
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL ABLATION COMPARISON")
print("=" * 70)

print()

header = (
    f"{'Model':20s}"
    f"{'After F1':>10s}"
    f"{'SHD':>8s}"
    f"{'Stable':>10s}"
    f"{'Delay':>8s}"
    f"{'P_changed':>12s}"
    f"{'P_stable':>12s}"
    f"{'Selectivity':>14s}"
)

print(header)
print("-" * len(header))

for r in results:

    print(
        f"{r['Model']:20s}"
        f"{r['After F1']:10.4f}"
        f"{r['After SHD']:8}"
        f"{r['Stable Preservation']:10.4f}"
        f"{r['Adaptation Delay']:8}"
        f"{r['Changed Plasticity']:12.4f}"
        f"{r['Stable Plasticity']:12.4f}"
        f"{r['Selectivity Ratio']:14.2f}x"
    )


# ============================================================
# SAVE CSV
# ============================================================

import csv

output_file = os.path.join(
    RESULTS_DIR,
    "selective_plasticity_ablation.csv"
)

with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print()
print("Saved:")
print(output_file)

print()
print("=" * 70)
print("DONE")
print("=" * 70)